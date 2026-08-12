"""
ingestion/embed_and_upload.py

Gün 9 — Embedding ve Qdrant'a Yükleme (Hibrit: dense + sparse, tamamen yerel)

Akış:
    1. loader + chunker'dan chunk listesini al (chunks_cache.json varsa oradan)
    2. Her grup chunk için BGE-M3'ün TEK encode() çağrısıyla hem DENSE hem
       SPARSE embedding'i aynı anda üret (API yok, rate limit yok)
    3. Qdrant koleksiyonunu (dense 1024 + sparse named vector'larla) oluştur
    4. Batch halinde upsert et

NOT: Gemini tamamen kaldırıldı. Artık tüm embedding yerel BGE-M3 ile üretiliyor.
Rate limit / TPM bütçesi kavramları yok — sınır artık sadece bilgisayarının
(GPU'nun) hızı.
"""

import json
import time
import uuid
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)

from config import settings
from logging_config import get_logger
from ingestion.loader import tum_dokumanlari_yukle
from ingestion.chunker import kayitlari_chunkla

logger = get_logger(__name__)

COLLECTION_NAME = "documents"
DENSE_VECTOR_SIZE = 1024   # BGE-M3'ün dense çıktısı sabit 1024 boyutlu
BATCH_SIZE = 64            # tek encode() çağrısında kaç chunk işlensin (GPU belleğine göre ayarlanabilir)

CHECKPOINT_FILE = Path("embed_progress.json")
CHUNKS_CACHE_FILE = Path("chunks_cache.json")

_UUID_NAMESPACE = uuid.UUID("6f3e9c1a-1a2b-4b2e-9f3a-8d2f4e6a9c1b")


def _chunk_id_to_qdrant_id(chunk_id: str) -> str:
    """chunk_id (dosya::sayfa::index) string'inden SABİT bir UUID üretir.
    Aynı chunk_id her zaman aynı UUID'ye eşlenir — bir dosyayı düzeltip
    yeniden işlediğinde SADECE o dosyanın id'leri değişir, diğerleri etkilenmez.
    """
    return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))

_bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


# ------------------------------------------------------------------
# Checkpoint yardımcıları (crash/kesinti durumunda kaldığı yerden devam)
# ------------------------------------------------------------------
def _checkpoint_oku() -> int:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))["son_tamamlanan_batch"]
        except Exception as e:
            logger.warning("Checkpoint dosyası okunamadı, 0'dan başlanıyor: %s", e)
            return 0
    return 0


def _checkpoint_yaz(batch_no: int) -> None:
    CHECKPOINT_FILE.write_text(json.dumps({"son_tamamlanan_batch": batch_no}), encoding="utf-8")


def _checkpoint_sil() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


# ------------------------------------------------------------------
# BGE-M3 ile TEK ÇAĞRIDA hem dense hem sparse üretimi
# ------------------------------------------------------------------
def bge_m3_hibrit_embed_batch(metinler: list[str]) -> tuple[list[list[float]], list[dict]]:
    """
    Bir grup metin için BGE-M3 ile aynı anda dense + sparse embedding üretir.
    Dönüş: (dense_listesi, sparse_listesi)
    """
    cikti = _bge_model.encode(
        metinler,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_listesi = [vektor.tolist() for vektor in cikti["dense_vecs"]]

    sparse_listesi = []
    for lexical_weights in cikti["lexical_weights"]:
        sparse_listesi.append({
            "indices": [int(k) for k in lexical_weights.keys()],
            "values": [float(v) for v in lexical_weights.values()],
        })

    return dense_listesi, sparse_listesi


# ------------------------------------------------------------------
# Qdrant koleksiyonu oluşturma
# ------------------------------------------------------------------
def qdrant_koleksiyon_olustur(client: QdrantClient) -> None:
    """'documents' koleksiyonunu dense(1024)+sparse named vector'larla oluşturur (idempotent)."""
    if client.collection_exists(COLLECTION_NAME):
        logger.info("Koleksiyon zaten var: %s", COLLECTION_NAME)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    logger.info("Koleksiyon oluşturuldu: %s (dense boyutu: %d)", COLLECTION_NAME, DENSE_VECTOR_SIZE)


# ------------------------------------------------------------------
# Batch upsert
# ------------------------------------------------------------------
def chunklari_yukle(client: QdrantClient, chunk_kayitlari: list[dict]) -> None:
    """chunk_kayitlari listesini BATCH_SIZE'lık gruplar halinde embed edip Qdrant'a yükler."""
    toplam = len(chunk_kayitlari)
    gruplar = [chunk_kayitlari[i:i + BATCH_SIZE] for i in range(0, toplam, BATCH_SIZE)]

    baslangic_batch = _checkpoint_oku()
    if baslangic_batch > 0:
        logger.info("Checkpoint bulundu: batch %d/%d'den devam ediliyor.", baslangic_batch + 1, len(gruplar))

    islenen = baslangic_batch * BATCH_SIZE

    for batch_no in range(baslangic_batch, len(gruplar)):
        batch = gruplar[batch_no]
        metinler = [chunk["metin"] for chunk in batch]

        baslangic_zaman = time.monotonic()
        dense_listesi, sparse_listesi = bge_m3_hibrit_embed_batch(metinler)

        point_id_baslangic = batch_no * BATCH_SIZE
        points = []
        for i, chunk in enumerate(batch):
            points.append(PointStruct(
                id=_chunk_id_to_qdrant_id(chunk["chunk_id"]),
                vector={
                    "dense": dense_listesi[i],
                    "sparse": SparseVector(
                        indices=sparse_listesi[i]["indices"],
                        values=sparse_listesi[i]["values"],
                    ),
                },
                payload={
                    "dosya_adi": chunk["dosya_adi"],
                    "kategori": chunk["kategori"],
                    "sayfa_no": chunk["sayfa_no"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"],
                    "metin": chunk["metin"],
                },
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        islenen += len(batch)
        sure = time.monotonic() - baslangic_zaman
        logger.info("Yüklendi: %d / %d (batch %d/%d, %.1f sn)", islenen, toplam, batch_no + 1, len(gruplar), sure)

        _checkpoint_yaz(batch_no + 1)

    _checkpoint_sil()
    logger.info("Checkpoint temizlendi, tüm chunk'lar yüklendi.")


def _var_olan_id_leri_al(client: QdrantClient) -> set[str]:
    """Qdrant'taki mevcut tüm point id'lerini döndürür (scroll ile, sayfa sayfa)."""
    id_ler = set()
    sonraki_sayfa = None
    while True:
        sonuc, sonraki_sayfa = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=False,
            with_vectors=False,
            offset=sonraki_sayfa,
        )
        id_ler.update(str(point.id) for point in sonuc)
        if sonraki_sayfa is None:
            break
    return id_ler


def eski_int_idleri_yeni_uuidye_tasi(client: QdrantClient) -> None:
    """Eski (sayısal, sıra bazlı) id'lerle yüklenmiş point'leri, payload'daki
    chunk_id'den türetilen SABİT UUID id'lere taşır.

    ÖNEMLİ: embedding'ler YENİDEN HESAPLANMAZ — sadece var olan vektör/payload
    aynı içerikle, yeni id altında tekrar yazılır, sonra eski id silinir.
    Bu, geçmişte ürettiğimiz 37778 embedding'i kaybetmeden yeni,
    içerik-bazlı id şemasına geçmemizi sağlıyor.
    """
    eski_pointler = []
    sonraki = None
    while True:
        sonuc, sonraki = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=True,
            offset=sonraki,
        )
        eski_pointler.extend(p for p in sonuc if isinstance(p.id, int))
        if sonraki is None:
            break

    if not eski_pointler:
        logger.info("Eski (sayısal id'li) point yok, taşımaya gerek yok.")
        return

    logger.info("%d eski point, yeni UUID şemasına taşınıyor (embedding tekrar hesaplanmıyor)...", len(eski_pointler))

    yeni_points = [
        PointStruct(id=_chunk_id_to_qdrant_id(p.payload["chunk_id"]), vector=p.vector, payload=p.payload)
        for p in eski_pointler
    ]
    eski_id_ler = [p.id for p in eski_pointler]

    for i in range(0, len(yeni_points), 500):
        client.upsert(collection_name=COLLECTION_NAME, points=yeni_points[i:i + 500])
    for i in range(0, len(eski_id_ler), 500):
        client.delete(collection_name=COLLECTION_NAME, points_selector=eski_id_ler[i:i + 500])

    logger.info("Taşıma tamamlandı: %d point yeni id ile yazıldı, eski id'ler silindi.", len(yeni_points))


def eksikleri_tamamla(client: QdrantClient, chunk_kayitlari: list[dict]) -> None:
    """Qdrant'ta olması gerekip de olmayan chunk'ları bulup sadece onları embed edip yükler."""
    logger.info("Mevcut point id'leri Qdrant'tan okunuyor...")
    var_olanlar = _var_olan_id_leri_al(client)
    logger.info("Qdrant'ta %d point var.", len(var_olanlar))

    eksik_chunklar = [
        chunk for chunk in chunk_kayitlari
        if _chunk_id_to_qdrant_id(chunk["chunk_id"]) not in var_olanlar
    ]
    logger.info("%d chunk eksik, sadece bunlar işlenecek.", len(eksik_chunklar))

    if not eksik_chunklar:
        logger.info("Eksik yok, her şey tamam.")
        return

    for start in range(0, len(eksik_chunklar), BATCH_SIZE):
        batch = eksik_chunklar[start:start + BATCH_SIZE]
        metinler = [chunk["metin"] for chunk in batch]

        dense_listesi, sparse_listesi = bge_m3_hibrit_embed_batch(metinler)

        points = []
        for i, chunk in enumerate(batch):
            points.append(PointStruct(
                id=_chunk_id_to_qdrant_id(chunk["chunk_id"]),
                vector={
                    "dense": dense_listesi[i],
                    "sparse": SparseVector(
                        indices=sparse_listesi[i]["indices"],
                        values=sparse_listesi[i]["values"],
                    ),
                },
                payload={
                    "dosya_adi": chunk["dosya_adi"],
                    "kategori": chunk["kategori"],
                    "sayfa_no": chunk["sayfa_no"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"],
                    "metin": chunk["metin"],
                },
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("Tamamlandı: %d / %d eksik chunk", min(start + BATCH_SIZE, len(eksik_chunklar)), len(eksik_chunklar))

    logger.info("Eksikler tamamlandı.")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    if CHUNKS_CACHE_FILE.exists():
        logger.info("Chunk cache bulundu (%s), dosyalar yeniden okunmuyor.", CHUNKS_CACHE_FILE)
        chunk_kayitlari = json.loads(CHUNKS_CACHE_FILE.read_text(encoding="utf-8"))
    else:
        logger.info("Chunk cache yok, dosyalar okunup chunk'lanıyor (bu adım birkaç dakika sürebilir)...")
        kayitlar = tum_dokumanlari_yukle()
        chunk_kayitlari = kayitlari_chunkla(kayitlar)
        CHUNKS_CACHE_FILE.write_text(json.dumps(chunk_kayitlari, ensure_ascii=False), encoding="utf-8")
        logger.info("Chunk cache kaydedildi (%s).", CHUNKS_CACHE_FILE)

    logger.info("%d chunk hazır.", len(chunk_kayitlari))

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    qdrant_koleksiyon_olustur(client)
    eski_int_idleri_yeni_uuidye_tasi(client)
    eksikleri_tamamla(client, chunk_kayitlari)

    logger.info("Tamamlandı.")


if __name__ == "__main__":
    main()
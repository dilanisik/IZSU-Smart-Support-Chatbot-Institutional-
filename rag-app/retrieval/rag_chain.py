"""
retrieval/rag_chain.py

Gün 10 — Semantik Arama ve RAG Zinciri

Akış:
    soru -> soru_embedle (BGE-M3, dense+sparse) -> qdranttan_ara (hibrit, RRF)
    -> esik_ile_filtrele -> prompt_olustur -> ollama_ile_yanitla -> yanit + kaynak atıfı.

embed_and_upload.py ile aynı BGE-M3 modelini, aynı encode() ayarlarıyla,
aynı "dense"/"sparse" named vector isimleriyle ve "documents" koleksiyonunu
kullanır — bu ikisi arasında herhangi bir tutarsızlık olursa arama sonuç
döndürmez ya da anlamsız sonuç döner.

Ollama sunucusu (10.100.17.144) soğuk başlarken modeli belleğe yüklemek tek
başına ~150-200 sn sürebiliyor (ölçüldü: total_duration 189sn ve bunun 177sn'si
load_duration). Bu yüzden hem timeout yüksek tutuluyor hem de keep_alive ile
model yüklendikten sonra bellekte tutuluyor (30m) ki art arda sorularda bu maliyet
tekrar ödenmesin.

Kullanım:
    from retrieval.rag_chain import soruyu_yanitla
    sonuc = soruyu_yanitla("......?")
"""

import requests
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
COLLECTION_NAME = "documents"          # embed_and_upload.py ile aynı
DENSE_VECTOR_NAME = "dense"            # embed_and_upload.py ile aynı
SPARSE_VECTOR_NAME = "sparse"          # embed_and_upload.py ile aynı

OLLAMA_URL = "http://10.100.17.144:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:27b"
OLLAMA_TIMEOUT_SN = 300     # soğuk yükleme (180sn) + üretim süresi payı
OLLAMA_KEEP_ALIVE = "30m"   # model yüklendikten sonra bellekte kalma süresi

TOP_K_RETRIEVE = 5         # Qdrant'tan çekilecek ve doğrudan prompt'a verilecek chunk sayısı

#  Gün 9'da düşük skorlu/bozuk chunk'ların
# 0.60 civarında kaldığı gözlemlenmişti, o yüzden başlangıç eşiği 0.55.
SCORE_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Model — modül import edilirken bir kere yüklenir (her sorguda değil)
# ---------------------------------------------------------------------------
_bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
_qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def soru_embedle(soru: str):
    """Soruyu, embed_and_upload.py'deki chunk embedding ile AYNI ayarlarla
    dense + sparse embed eder. Ayarlar (fp16, return_colbert_vecs=False)
    ikisinde de birebir aynı olmak zorunda."""
    cikti = _bge_model.encode(
        [soru],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense_vec = cikti["dense_vecs"][0].tolist()
    lexical_weights = cikti["lexical_weights"][0]
    sparse_vec = SparseVector(
        indices=[int(k) for k in lexical_weights.keys()],
        values=[float(v) for v in lexical_weights.values()],
    )
    return dense_vec, sparse_vec


def qdranttan_ara(dense_vec, sparse_vec, top_k: int = TOP_K_RETRIEVE):
    """Hibrit arama: dense ve sparse aramayı ayrı ayrı prefetch edip RRF
    (Reciprocal Rank Fusion) ile birleştirir. qdrant-client >= 1.10 gerekir."""
    sonuc = _qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, limit=top_k),
            Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, limit=top_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return sonuc.points  # her biri .score ve .payload (dosya_adi, kategori, sayfa_no, metin, ...) içerir


def esik_ile_filtrele(points, esik: float = SCORE_THRESHOLD):
    filtrelenmis = [p for p in points if p.score >= esik]
    logger.info("Eşik filtrelemesi: %d/%d chunk kaldı (eşik=%.2f)", len(filtrelenmis), len(points), esik)
    return filtrelenmis


def prompt_olustur(soru: str, points) -> str:
    baglam_parcalari = []
    for i, p in enumerate(points, start=1):
        pl = p.payload
        sayfa = pl["sayfa_no"] if pl["sayfa_no"] is not None else "-"
        baglam_parcalari.append(
            f"[Kaynak {i} — {pl['dosya_adi']}, sayfa {sayfa}, kategori: {pl['kategori']}]\n{pl['metin']}"
        )
    baglam = "\n\n".join(baglam_parcalari)

    return f"""Aşağıda İZSU kurumsal dokümanlarından alınmış bağlam parçaları var.
Yalnızca bu bağlama dayanarak soruyu yanıtla. Bağlamda yeterli bilgi yoksa
"Bu konuda elimde yeterli bilgi yok." de — bağlamda olmayan hiçbir bilgi uydurma.

Yanıtının sonuna, kullandığın kaynak(lar)ı "Kaynak: <dosya adı>, sayfa <no>"
formatında ekle.

--- BAĞLAM ---
{baglam}
--- BAĞLAM SONU ---

Soru: {soru}
Yanıt:"""


def ollama_ile_yanitla(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # Model sunucuda soğuk (bellekte değil) olduğunda yükleme tek başına
        # 150-200 sn sürebiliyor (ölçüldü). keep_alive ile yükledikten sonra
        # bellekte tutulmasını istiyoruz ki art arda sorularda bu yükleme
        # maliyeti tekrar ödenmesin.
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    yanit = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SN)
    yanit.raise_for_status()
    return yanit.json()["message"]["content"]


def soruyu_yanitla(soru: str) -> dict:
    """
    Router (Gün 14) tarafından çağrılacak ana giriş noktası.
    Dönen dict: {"yanit": str, "kaynaklar": list[dict], "kullanilan_chunk_sayisi": int}
    """
    dense_vec, sparse_vec = soru_embedle(soru)
    adaylar = qdranttan_ara(dense_vec, sparse_vec)
    secilenler = esik_ile_filtrele(adaylar)

    if not secilenler:
        return {
            "yanit": "Bu konuda elimde yeterli bilgi yok.",
            "kaynaklar": [],
            "kullanilan_chunk_sayisi": 0,
        }

    prompt = prompt_olustur(soru, secilenler)
    yanit = ollama_ile_yanitla(prompt)

    kaynaklar = [
        {
            "dosya_adi": p.payload["dosya_adi"],
            "sayfa_no": p.payload["sayfa_no"],
            "kategori": p.payload["kategori"],
            "score": p.score,
        }
        for p in secilenler
    ]

    return {"yanit": yanit, "kaynaklar": kaynaklar, "kullanilan_chunk_sayisi": len(secilenler)}


if __name__ == "__main__":
    soru = "labaratuvar işlemleri nelerdir?"
    sonuc = soruyu_yanitla(soru)
    print(sonuc["yanit"])
    print("\nKaynaklar:")
    for k in sonuc["kaynaklar"]:
        print(f"  - {k['dosya_adi']} (sayfa {k['sayfa_no']}, skor {k['score']:.3f})")

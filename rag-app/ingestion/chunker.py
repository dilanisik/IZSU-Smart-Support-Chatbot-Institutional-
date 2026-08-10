"""
ingestion/chunker.py

Metni token bazlı, örtüşmeli (overlap'li) parçalara böler.
Plan hedefi: 300-500 token / chunk, %10-15 overlap.

Token sayımı için tiktoken (cl100k_base) kullanılır — bu, gerçek embedding
modelinin tokenizer'ıyla birebir aynı olmasa da (OpenAI/Gemini farklı
tokenizer kullanabilir), tutarlı ve öngörülebilir bir chunk boyutu
sağlamak için yeterli bir yaklaşımdır.

Kullanım:
    from ingestion.chunker import chunkla, kayitlari_chunkla

    parcalar = chunkla("uzun bir metin ...")

    # loader'dan gelen kayıtları (dosya_adi, kategori, sayfa_no, metin)
    # chunk metadata'sıyla birlikte üretir:
    chunk_kayitlari = kayitlari_chunkla(kayitlar)
"""

import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

HEDEF_TOKEN = 400          # 300-500 aralığının ortası
OVERLAP_ORANI = 0.125      # %10-15 aralığının ortası


def token_say(metin: str) -> int:
    return len(encoding.encode(metin))


def chunkla(metin: str, hedef_token: int = HEDEF_TOKEN, overlap_orani: float = OVERLAP_ORANI) -> list[str]:
    """Metni hedef_token büyüklüğünde, overlap_orani kadar örtüşen parçalara böler."""
    tokenlar = encoding.encode(metin)
    if not tokenlar:
        return []

    overlap = int(hedef_token * overlap_orani)
    adim = hedef_token - overlap

    parcalar = []
    start = 0
    while start < len(tokenlar):
        parca_tokenlar = tokenlar[start:start + hedef_token]
        parcalar.append(encoding.decode(parca_tokenlar))
        if start + hedef_token >= len(tokenlar):
            break
        start += adim

    return parcalar


def kayitlari_chunkla(kayitlar: list[dict]) -> list[dict]:
    """
    loader.tum_dokumanlari_yukle() çıktısını alır, her kaydın metnini
    chunk'lara böler ve her chunk için metadata + chunk_id üretir.
    Bu liste doğrudan Gün 9'da embedding + Qdrant upload'a verilecek.
    """
    chunk_kayitlari = []
    for kayit in kayitlar:
        parcalar = chunkla(kayit["metin"])
        for i, parca in enumerate(parcalar):
            chunk_kayitlari.append({
                "dosya_adi": kayit["dosya_adi"],
                "kategori": kayit["kategori"],
                "sayfa_no": kayit["sayfa_no"],
                "chunk_index": i,
                "chunk_id": f"{kayit['dosya_adi']}::p{kayit['sayfa_no'] or 0}::c{i}",
                "metin": parca,
                "token_sayisi": token_say(parca),
            })
    return chunk_kayitlari


if __name__ == "__main__":
    from ingestion.loader import tum_dokumanlari_yukle

    kayitlar = tum_dokumanlari_yukle()
    chunk_kayitlari = kayitlari_chunkla(kayitlar)

    print(f"{len(kayitlar)} kayıt -> {len(chunk_kayitlari)} chunk")
    if chunk_kayitlari:
        ort_token = sum(c["token_sayisi"] for c in chunk_kayitlari) / len(chunk_kayitlari)
        print(f"Ortalama chunk boyutu: {ort_token:.0f} token")
        print("\nÖrnek chunk:")
        print(chunk_kayitlari[0])
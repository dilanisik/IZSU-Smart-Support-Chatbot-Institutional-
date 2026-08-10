"""
ingestion/loader.py

documents/ klasöründeki .docx ve .pdf dosyalarını okur, her biri için
metin + metadata (dosya adı, kategori, sayfa no) içeren kayıtlar üretir.

Kategori, dosya adının ilk alt çizgiden önceki kısmından çıkarılır
(örn. "kurumsal_izsu-tarihcesi.docx" -> kategori: "kurumsal").

Kullanım:
    from ingestion.loader import tum_dokumanlari_yukle
    kayitlar = tum_dokumanlari_yukle()
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document

from logging_config import get_logger

logger = get_logger(__name__)

DOCUMENTS_DIR = Path("documents")


def kategori_cikar(dosya_adi: str) -> str:
    if "_" not in dosya_adi:
        return "diger"
    return dosya_adi.split("_", 1)[0]


def docx_oku(path: Path) -> list[dict]:
    """Bir .docx dosyasını tek bir kayıt olarak döndürür (sayfa kavramı yok)."""
    doc = Document(path)
    metin = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if not metin.strip():
        logger.warning("Boş içerik, atlandı: %s", path.name)
        return []

    return [{
        "dosya_adi": path.name,
        "kategori": kategori_cikar(path.name),
        "sayfa_no": None,
        "metin": metin,
    }]


def pdf_oku(path: Path) -> list[dict]:
    """Bir .pdf dosyasını sayfa sayfa kayıtlara böler."""
    try:
        reader = PdfReader(path)
    except Exception as e:
        logger.error("PDF açılamadı, atlandı: %s -> %s", path.name, e)
        return []

    kayitlar = []
    for i, sayfa in enumerate(reader.pages, start=1):
        metin = sayfa.extract_text() or ""
        if metin.strip():
            kayitlar.append({
                "dosya_adi": path.name,
                "kategori": kategori_cikar(path.name),
                "sayfa_no": i,
                "metin": metin,
            })
    if not kayitlar:
        logger.warning("Metin çıkarılamadı (görsel/taranmış olabilir): %s", path.name)
    return kayitlar


def tum_dokumanlari_yukle(klasor: Path = DOCUMENTS_DIR) -> list[dict]:
    """documents/ altındaki tüm dosyaları okuyup tek bir liste halinde döndürür."""
    kayitlar = []
    for path in sorted(klasor.glob("*.docx")):
        kayitlar.extend(docx_oku(path))
    for path in sorted(klasor.glob("*.pdf")):
        kayitlar.extend(pdf_oku(path))

    logger.info("Toplam %d kayıt yüklendi (%d dosyadan)",
                len(kayitlar), len(list(klasor.glob("*.docx"))) + len(list(klasor.glob("*.pdf"))))
    return kayitlar


if __name__ == "__main__":
    kayitlar = tum_dokumanlari_yukle()
    print(f"{len(kayitlar)} kayıt yüklendi.")
    kategoriler = {}
    for k in kayitlar:
        kategoriler[k["kategori"]] = kategoriler.get(k["kategori"], 0) + 1
    for kat, sayi in sorted(kategoriler.items(), key=lambda x: -x[1]):
        print(f"  {kat}: {sayi}")

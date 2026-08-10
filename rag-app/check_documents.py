"""
documents/ klasöründeki tüm .docx ve .pdf dosyalarını tarar,
her birinden ne kadar metin çıkarılabildiğini raporlar.

Değişiklikler:
- Her dosya işlenmeden önce adı yazdırılır (hangi dosyada takıldığını görebilmek için)
- Her dosyaya 15 saniyelik zaman aşımı konuldu (büyük/bozuk PDF'ler script'i kilitlemesin diye)
- PDF'lerde ilk 5 sayfa yeterli (dosyanın metin içerip içermediğini anlamak için tüm sayfaları taramaya gerek yok)
- Bozuk dosyalar artık script'i durdurmuyor, listeye "HATA" olarak ekleniyor

Kullanım:
    python check_documents.py
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pypdf import PdfReader
from docx import Document

DOCUMENTS_DIR = Path("documents")
MIN_CHAR_ESIGI = 50
PDF_MAX_SAYFA = 5       # sadece ilk 5 sayfaya bak, hız için
DOSYA_TIMEOUT_SN = 15


def docx_metin_uzunlugu(path: Path) -> int:
    doc = Document(path)
    return sum(len(p.text) for p in doc.paragraphs)


def pdf_metin_uzunlugu(path: Path) -> int:
    reader = PdfReader(path)
    sayfalar = reader.pages[:PDF_MAX_SAYFA]
    return sum(len(page.extract_text() or "") for page in sayfalar)


def dosya_isle(path: Path) -> int:
    if path.suffix == ".docx":
        return docx_metin_uzunlugu(path)
    return pdf_metin_uzunlugu(path)


def kategori_cikar(dosya_adi: str) -> str:
    if "_" not in dosya_adi:
        return "!! KATEGORI YOK (alt çizgi bulunamadı)"
    return dosya_adi.split("_", 1)[0]


def main():
    dosyalar = sorted(DOCUMENTS_DIR.glob("*.docx")) + sorted(DOCUMENTS_DIR.glob("*.pdf"))
    supheli = []
    hatali = []
    kategoriler = {}

    print(f"Toplam {len(dosyalar)} dosya taranıyor...\n")

    for i, path in enumerate(dosyalar, 1):
        print(f"[{i}/{len(dosyalar)}] {path.name}")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(dosya_isle, path)
            try:
                uzunluk = future.result(timeout=DOSYA_TIMEOUT_SN)
            except FutureTimeoutError:
                uzunluk = -1
                hatali.append((path.name, "TIMEOUT (15sn içinde bitmedi)"))
                print(f"    !! ZAMAN AŞIMI")
            except Exception as e:
                uzunluk = -1
                hatali.append((path.name, str(e)))
                print(f"    !! HATA: {e}")

        kategori = kategori_cikar(path.name)
        kategoriler[kategori] = kategoriler.get(kategori, 0) + 1

        if 0 <= uzunluk < MIN_CHAR_ESIGI:
            supheli.append((path.name, uzunluk))

    print("\n=== Kategori dağılımı ===")
    for kat, sayi in sorted(kategoriler.items(), key=lambda x: -x[1]):
        print(f"  {kat}: {sayi}")

    print(f"\n=== Hatalı/açılamayan dosyalar ===")
    if not hatali:
        print("  Yok")
    else:
        for ad, hata in hatali:
            print(f"  {ad}  ->  {hata}")

    print(f"\n=== Şüpheli dosyalar ({MIN_CHAR_ESIGI} karakterden az metin, ilk {PDF_MAX_SAYFA} sayfa) ===")
    if not supheli:
        print("  Yok")
    else:
        for ad, uzunluk in supheli:
            print(f"  {ad}  ->  {uzunluk} karakter")


if __name__ == "__main__":
    main()
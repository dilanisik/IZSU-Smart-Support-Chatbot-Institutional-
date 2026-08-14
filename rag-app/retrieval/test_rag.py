"""
test_rag.py (retrieval/ klasöründe)

Gün 11 — Doküman RAG Testleri

test_sorular.json içindeki soruları retrieval/rag_chain.py'deki soruyu_yanitla()
üzerinden tek tek çalıştırır, her birinin yanıtını + kaynaklarını + süresini kaydeder.

Devam/tekrar deneme mantığı: bir soru daha önce BAŞARIYLA (hata=None, yanit dolu)
tamamlanmışsa tekrar sorulmaz. Ama timeout/hata almış sorular "tamamlanmamış"
sayılır ve script'i tekrar çalıştırdığında OTOMATİK olarak yeniden denenir

Kullanım (rag-app kök dizininden, venv aktifken):
    python -m retrieval.test_rag

Çıktı:
    retrieval/test_sonuclari_gun11.json — ham sonuçlar
"""

import json
import time
from pathlib import Path

from retrieval.rag_chain import soruyu_yanitla

BASE_DIR = Path(__file__).resolve().parent

SORULAR_DOSYASI = BASE_DIR / "test_sorular.json"
SONUC_DOSYASI = BASE_DIR / "test_sonuclari_gun11.json"


def sonuclari_yukle() -> list[dict]:
    if SONUC_DOSYASI.exists():
        return json.loads(SONUC_DOSYASI.read_text(encoding="utf-8"))
    return []


def sonuclari_kaydet(sonuclar: list[dict]) -> None:
    SONUC_DOSYASI.write_text(json.dumps(sonuclar, ensure_ascii=False, indent=2), encoding="utf-8")


def basarili_mi(kayit: dict) -> bool:
    """Bir sonuç kaydının 'başarıyla tamamlanmış' sayılıp sayılmayacağını belirler.
    hata varsa ya da yanıt hiç üretilmemişse (None) başarısız sayılır -> tekrar denenir."""
    return kayit.get("hata") is None and kayit.get("yanit") is not None


def main():
    if not SORULAR_DOSYASI.exists():
        raise FileNotFoundError(
            f"test_sorular.json bulunamadı: {SORULAR_DOSYASI}\n"
            f"Bu dosyayı test_rag.py ile AYNI klasöre ({BASE_DIR}) koymalısın."
        )

    sorular = json.loads(SORULAR_DOSYASI.read_text(encoding="utf-8"))
    onceki_sonuclar = sonuclari_yukle()

    # id -> eski kayıt (varsa) sözlüğü; başarılı olanları koruyup, başarısızları elemek.
    eski_kayit_map = {k["id"]: k for k in onceki_sonuclar}
    basarili_id_ler = {i for i, k in eski_kayit_map.items() if basarili_mi(k)}

    # Sonuç listesini, başarılı eski kayıtlarla başlat (sırası korunsun diye soru sırasına göre).
    sonuclar = [eski_kayit_map[s["id"]] for s in sorular if s["id"] in basarili_id_ler]

    tekrar_denenecek = len(eski_kayit_map) - len(basarili_id_ler)
    if tekrar_denenecek > 0:
        print(f"{tekrar_denenecek} soru önceki denemede başarısız olmuştu (timeout/hata), yeniden denenecek.\n")

    print(f"{len(sorular)} soru toplam, {len(basarili_id_ler)} tanesi zaten başarıyla tamamlanmış.\n")

    for soru_kaydi in sorular:
        if soru_kaydi["id"] in basarili_id_ler:
            print(f"[{soru_kaydi['id']}] atlandı (zaten başarılı): {soru_kaydi['soru']}")
            continue

        print(f"[{soru_kaydi['id']}/{len(sorular)}] soruluyor: {soru_kaydi['soru']}")
        baslangic = time.monotonic()

        try:
            cikti = soruyu_yanitla(soru_kaydi["soru"])
            hata = None
        except Exception as e:
            cikti = {"yanit": None, "kaynaklar": [], "kullanilan_chunk_sayisi": 0}
            hata = str(e)
            print(f"  HATA: {hata}")

        sure_sn = time.monotonic() - baslangic
        print(f"  bitti ({sure_sn:.1f} sn, {cikti['kullanilan_chunk_sayisi']} chunk kullanıldı)\n")

        # Eğer bu id için daha önce (başarısız da olsa) bir kayıt varsa, elle girilmiş
        # retrieval_dogru/halusinasyon_var_mi/manuel_not alanlarını koru.
        eski = eski_kayit_map.get(soru_kaydi["id"], {})

        sonuc = {
            "id": soru_kaydi["id"],
            "kategori": soru_kaydi["kategori"],
            "soru": soru_kaydi["soru"],
            "beklenen": soru_kaydi["beklenen"],
            "not": soru_kaydi.get("not", ""),
            "yanit": cikti["yanit"],
            "kaynaklar": cikti["kaynaklar"],
            "kullanilan_chunk_sayisi": cikti["kullanilan_chunk_sayisi"],
            "sure_sn": round(sure_sn, 1),
            "hata": hata,
            "retrieval_dogru": eski.get("retrieval_dogru"),
            "halusinasyon_var_mi": eski.get("halusinasyon_var_mi"),
            "manuel_not": eski.get("manuel_not", ""),
        }
        sonuclar.append(sonuc)
        sonuclari_kaydet(sonuclar)  # her soru sonrası hemen kaydet

    basarisiz_kalan = sum(1 for s in sonuclar if not basarili_mi(s))
    print(f"\nTamamlandı: {len(sonuclar)}/{len(sorular)} soru işlendi.")
    if basarisiz_kalan > 0:
        print(f"UYARI: {basarisiz_kalan} soru hâlâ başarısız (muhtemelen tekrar timeout aldı). "
              f"Scripti tekrar çalıştırınca sadece bunlar denenecek.")
    print(f"Sonuçlar: {SONUC_DOSYASI}")
    print("\nŞimdi bu dosyayı aç, her satırın 'retrieval_dogru' ve 'halusinasyon_var_mi' "
          "alanlarını true/false olarak doldur, sonra Gün 11 raporunu buna göre yaz.")


if __name__ == "__main__":
    main()
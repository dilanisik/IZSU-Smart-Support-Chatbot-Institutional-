"""
sql_engine/result_formatter.py

Gün 13 — SQL Sonuçlarının Doğal Dile Dönüştürülmesi

Akış:
    DataFrame -> df_metne_cevir (LLM'in okuyabileceği metne çevir)
    -> sonucu_dogal_dile_cevir (LLM ile okunabilir cümle üret)

Ayrıca Streamlit (Gün 17) için bir ara katman: grafik_onerisi(df) — sonuç
verisinin yapısına bakarak tablo/grafik gösterimi için basit bir öneri üretir.
Gerçek çizim işini bu dosya yapmıyor, sadece "ne çizilmeli" kararını veriyor.

text_to_sql.py'deki LangGraph akışına, sql_calistir sonrası çalışan yeni bir
node (dogal_dile_cevir) olarak bağlanır — SqlAgentState'teki dogal_dil_yanit
alanını bu modül doldurur.

Gün 13 içi düzeltme: grafik_onerisi(), "yil"/"ay" gibi zaman belirten ama
düz tamsayı (int) olarak saklanan kolonları (örn. EXTRACT(YEAR ...) sonucu)
başta tanıyamıyordu — yalnızca gerçek datetime tipindeki ya da adında
"tarih"/"donem"/"date" geçen kolonları zaman kolonu sayıyordu. Test sırasında
"yil" adlı bir kolonun hem zaman hem sayısal kolon listesine de girmediği
(ikisi arasında "kaybolduğu") ve sonuç olarak line grafik yerine yanlışlıkla
"tablo" önerildiği görüldü. Anahtar kelime listesine "yil"/"yıl"/"ay" eklendi.

Kullanım:
    from sql_engine.result_formatter import sonucu_dogal_dile_cevir
    yanit = sonucu_dogal_dile_cevir("Ödenmemiş faturalar kimde?", df)
"""

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

# DataFrame bu satır sayısını aşarsa LLM'e tamamı yerine özet + ilk/son N satır
# gönderilir — hem prompt'u şişirmemek hem de zaten yavaş olan modele (bkz.
# Gün 12'deki 4 dakikalık SQL üretim süreleri) gereksiz yük bindirmemek için.
MAKS_TAM_SATIR = 30
OZET_ORNEK_SATIR = 10

_llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    keep_alive=settings.ollama_keep_alive,
    timeout=settings.ollama_timeout_sn,
)

_PROMPT = ChatPromptTemplate.from_template("""Aşağıda bir SQL sorgusunun sonucu (tablo verisi) var. Bu veriye
dayanarak kullanıcının sorusunu, sade ve anlaşılır bir Türkçe cümle/paragrafla
yanıtla. Sayısal değerleri (tutar, adet, oran vb.) doğru şekilde aktar,
veride olmayan hiçbir bilgi uydurma. Tablo çok satırlıysa, tamamını tek tek
sayma; genel bir özet ver (toplam kaç kayıt olduğunu belirt) ve varsa dikkat
çekici birkaç örneği vurgula.

Soru: {soru}

--- SORGU SONUCU ---
{veri}
--- SORGU SONUCU SONU ---

Yanıt:""")

_dogal_dil_zinciri = _PROMPT | _llm | StrOutputParser()


def df_metne_cevir(df: pd.DataFrame) -> str:
    """DataFrame'i LLM'in okuyabileceği düz bir metne çevirir. Küçük
    tablolarda tamamını, büyük tablolarda özet + örnek satırları verir."""
    satir_sayisi = len(df)

    if satir_sayisi == 0:
        return "Sorgu sonucu boş — hiçbir kayıt bulunamadı."

    if satir_sayisi <= MAKS_TAM_SATIR:
        return f"Toplam {satir_sayisi} satır:\n{df.to_string(index=False)}"

    # Büyük tablo: özet istatistik + ilk/son birkaç örnek satır.
    parcalar = [f"Toplam {satir_sayisi} satır var (tamamı gösterilmiyor, örnekleniyor)."]

    sayisal_kolonlar = df.select_dtypes(include="number").columns.tolist()
    if sayisal_kolonlar:
        ozet = df[sayisal_kolonlar].describe().loc[["mean", "min", "max"]]
        parcalar.append(f"Sayısal kolonların özeti:\n{ozet.to_string()}")

    parcalar.append(f"İlk {OZET_ORNEK_SATIR} satır:\n{df.head(OZET_ORNEK_SATIR).to_string(index=False)}")
    parcalar.append(f"Son {OZET_ORNEK_SATIR} satır:\n{df.tail(OZET_ORNEK_SATIR).to_string(index=False)}")

    return "\n\n".join(parcalar)


def sonucu_dogal_dile_cevir(soru: str, df: pd.DataFrame) -> str:
    """SQL sonucunu (DataFrame) + orijinal soruyu LLM'e vererek okunabilir
    bir Türkçe yanıt üretir."""
    veri_metni = df_metne_cevir(df)
    yanit = _dogal_dil_zinciri.invoke({"soru": soru, "veri": veri_metni})
    logger.info("Doğal dil yanıtı üretildi (%d satırlık sonuçtan).", len(df))
    return yanit


# Kolon adında bu kelimelerden biri geçiyorsa, kolon "zaman ekseni" olarak
# kabul edilir — hem gerçek datetime hem de EXTRACT(YEAR ...) gibi düz
# tamsayı/int dönen zaman kolonlarını (yil, ay vb.) yakalamak için.
_ZAMAN_ANAHTAR_KELIMELERI = ["tarih", "donem", "dönem", "date", "yil", "yıl", "ay", "hafta"]


def grafik_onerisi(df: pd.DataFrame) -> dict:
    """
    Streamlit (Gün 17) için basit bir gösterim önerisi üretir. Gerçek çizim
    burada yapılmaz — yalnızca "ne tür bir gösterim uygun" kararı verilir.

    Dönüş: {"tip": "tablo" | "bar" | "line", "x": str | None, "y": str | None}
    """
    if df.empty or len(df.columns) == 0:
        return {"tip": "tablo", "x": None, "y": None}

    sayisal_kolonlar = df.select_dtypes(include="number").columns.tolist()

    # Önce gerçek datetime tipindeki kolonlara bak.
    tarih_kolonlari = df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns.tolist()

    # Sonra isim bazlı sezgisel kontrol — "yil", "donem" gibi düz int/str
    # olarak saklanan ama aslında zaman ekseni olan kolonları yakalar.
    # NOT: sayısal olsa bile (örn. "yil" int'tir) burada zaman kolonu sayılır,
    # bu yüzden aşağıda kategorik/sayısal ayrımından ÖNCE çıkarılıyor.
    isim_bazli_zaman_kolonlari = [
        k for k in df.columns
        if k not in tarih_kolonlari and any(anahtar in k.lower() for anahtar in _ZAMAN_ANAHTAR_KELIMELERI)
    ]
    tarih_kolonlari = tarih_kolonlari + isim_bazli_zaman_kolonlari

    # Zaman kolonu olarak sayılanları, sayısal kolon listesinden çıkar ki
    # "y ekseni" olarak yanlışlıkla kendisi seçilmesin (örn. "yil" hem
    # sayısal hem zaman kolonu olabilir, y ekseni olarak toplam_tuketim
    # gibi GERÇEK bir ölçüm kolonu seçilmeli).
    olculebilir_sayisal_kolonlar = [k for k in sayisal_kolonlar if k not in tarih_kolonlari]

    kategorik_kolonlar = [
        k for k in df.columns
        if k not in sayisal_kolonlar and k not in tarih_kolonlari
    ]

    # Tek satırlık ya da hiç ölçülebilir sayısal veri içermeyen sonuçlar için grafik anlamsız.
    if len(df) <= 1 or not olculebilir_sayisal_kolonlar:
        return {"tip": "tablo", "x": None, "y": None}

    # Zaman kolonu varsa -> zaman serisi -> çizgi grafik.
    if tarih_kolonlari:
        return {"tip": "line", "x": tarih_kolonlari[0], "y": olculebilir_sayisal_kolonlar[0]}

    # Kategorik bir kolon (örn. isim, kategori) + sayısal kolon varsa -> bar grafik.
    if kategorik_kolonlar and len(df) <= 50:  # çok fazla kategori bar grafiği okunaksız yapar
        return {"tip": "bar", "x": kategorik_kolonlar[0], "y": olculebilir_sayisal_kolonlar[0]}

    return {"tip": "tablo", "x": None, "y": None}


if __name__ == "__main__":
    # Hızlı manuel test — text_to_sql.py'nin ürettiği gerçek bir sonuçla dener.
    from sql_engine.text_to_sql import soruyu_sqlle_yanitla

    soru = "Faturasını ödememiş müşterilerin adı, soyadı ve borç tutarı nedir?"
    sql_sonucu = soruyu_sqlle_yanitla(soru)

    if sql_sonucu["hata"]:
        print("SQL HATASI:", sql_sonucu["hata"])
    else:
        print("SQL:", sql_sonucu["sql"])
        print("\nDoğal dil yanıtı:")
        print(sonucu_dogal_dile_cevir(soru, sql_sonucu["sonuc"]))
        print("\nGrafik önerisi:", grafik_onerisi(sql_sonucu["sonuc"]))
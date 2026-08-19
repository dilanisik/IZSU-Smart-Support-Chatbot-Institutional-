"""
sql_engine/text_to_sql.py

Gün 12 — Text-to-SQL Motoru (LangChain + LangGraph)
Gün 13 güncellemesi: sonucu doğal dile çeviren node eklendi.
Gün 14 düzeltmesi: calistir node'u hata verdiğinde (örn. modelin var olmayan
bir kolon/tablo uydurup SQL çalıştırma hatası vermesi durumunda),
dogal_dile_cevir node'u YİNE DE tetikleniyordu — çünkü calistir'den sonra
koşulsuz bir kenar vardı. Bu, None bir DataFrame'i doğal dile çevirmeye
çalışıp ikinci bir hataya ("NoneType has no len()") yol açıyordu. Artık
calistir'den sonra da guvenlik'teki gibi koşullu bir dallanma var: hata
varsa direkt END'e gidiliyor, yoksa dogal_dile_cevir'e geçiliyor.

Akış:
    soru -> uret -> guvenlik -> [güvenli mi?]
        değilse -> END (hata state'te)
        güvenliyse -> calistir -> [hata var mı?]
            varsa -> END (hata state'te)
            yoksa -> dogal_dile_cevir -> END


Şema, db/init.sql (Gün 4) ile BİREBİR eşleştirildi:
    musteriler(musteri_id, ad, soyad, tc_no, abonelik_no, sayac_no, kayit_tarihi)
    adresler(adres_id, musteri_id, il, ilce, mahalle, sokak_cadde, bina_no,
             daire_no, posta_kodu)
    su_tuketimi(tuketim_id, musteri_id, ilk_endeks, son_endeks, tuketim_m3,
                okuma_tarihi)
    faturalar(fatura_id, musteri_id, tuketim_id, donem, tuketim_m3, tutar,
              son_odeme_tarihi, odendi_mi)

ÖNEMLİ: sql_guvenli_mi() yalnızca TABLO isimlerini whitelist ile kontrol
ediyor, KOLON isimlerini doğrulamıyor. Model bazen var olmayan bir kolon
uydurabiliyor (örn. faturalar.abonelik_no gibi — gerçekte yok) — bu durumda
güvenlik kontrolünden geçer ama gerçek veritabanında çalıştırma hatası
alır. Bu hata düzgün yakalanıp state'e yazılıyor (çökme yok), ama ileride
kolon bazlı bir whitelist eklemek halüsinasyonu daha erken yakalayabilir —
bu bir geliştirme önerisi olarak not edildi.

Kullanım:
    from sql_engine.text_to_sql import soruyu_sqlle_yanitla
    sonuc = soruyu_sqlle_yanitla("Uğur EREN'in son 3 aylık faturası nedir?")
"""

import re
from typing import TypedDict, Optional

import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from config import settings
from logging_config import get_logger
from sql_engine.result_formatter import sonucu_dogal_dile_cevir

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Güvenlik — whitelist (init.sql ile birebir)
# ---------------------------------------------------------------------------
IZINLI_TABLOLAR = {"musteriler", "adresler", "su_tuketimi", "faturalar"}

YASAKLI_ANAHTAR_KELIMELER = [
    "insert", "update", "delete", "drop", "alter", "truncate", "create",
    "grant", "revoke", "exec", "execute", "merge", "attach", "pragma",
    "copy", "vacuum", "call",
]

# ---------------------------------------------------------------------------
# LangChain bileşenleri
# ---------------------------------------------------------------------------
_db = SQLDatabase.from_uri(
    settings.postgres_url,
    include_tables=list(IZINLI_TABLOLAR),
    sample_rows_in_table_info=2,
)

_llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    keep_alive=settings.ollama_keep_alive,
    timeout=settings.ollama_timeout_sn,
)

FEW_SHOT_ORNEKLER = """
Örnek 1:
Soru: Uğur EREN adlı kişinin son 3 aylık faturası nedir?
SQL:
SELECT f.donem, f.tuketim_m3, f.tutar, f.odendi_mi
FROM faturalar f
JOIN musteriler m ON f.musteri_id = m.musteri_id
WHERE m.ad = 'Uğur' AND m.soyad = 'EREN'
ORDER BY f.donem DESC
LIMIT 3;

Örnek 2:
Soru: Faturasını ödememiş müşterilerin adı, soyadı ve borç tutarı nedir?
SQL:
SELECT m.ad, m.soyad, f.tutar, f.donem
FROM faturalar f
JOIN musteriler m ON f.musteri_id = m.musteri_id
WHERE f.odendi_mi = false
ORDER BY f.donem DESC;

Örnek 3:
Soru: 2025 yılında en çok su tüketen 5 müşteri kimdir?
SQL:
SELECT m.ad, m.soyad, SUM(s.tuketim_m3) AS toplam_tuketim
FROM su_tuketimi s
JOIN musteriler m ON s.musteri_id = m.musteri_id
WHERE EXTRACT(YEAR FROM s.okuma_tarihi) = 2025
GROUP BY m.musteri_id, m.ad, m.soyad
ORDER BY toplam_tuketim DESC
LIMIT 5;

Örnek 4:
Soru: İzmir Konak'ta oturan müşterilerin sayacı hangi tarihte okunmuş?
SQL:
SELECT m.ad, m.soyad, s.okuma_tarihi, s.tuketim_m3
FROM su_tuketimi s
JOIN musteriler m ON s.musteri_id = m.musteri_id
JOIN adresler a ON a.musteri_id = m.musteri_id
WHERE a.il = 'İzmir' AND a.ilce = 'Konak'
ORDER BY s.okuma_tarihi DESC;

Örnek 5:
Soru: Son 3 ay içinde okunmuş su tüketim kayıtları nelerdir?
SQL:
SELECT m.ad, m.soyad, s.okuma_tarihi, s.tuketim_m3
FROM su_tuketimi s
JOIN musteriler m ON s.musteri_id = m.musteri_id
WHERE s.okuma_tarihi >= CURRENT_DATE - INTERVAL '3 months'
ORDER BY s.okuma_tarihi DESC;

Örnek 6:
Soru: Soyadı Yılmaz olan müşterileri listeler misin?
SQL:
SELECT m.ad, m.soyad, m.abonelik_no
FROM musteriler m
WHERE m.soyad ILIKE '%Yılmaz%';

Örnek 7:
Soru: Son ödeme tarihi geçmiş ama hâlâ ödenmemiş faturalar hangileri?
SQL:
SELECT m.ad, m.soyad, f.tutar, f.son_odeme_tarihi
FROM faturalar f
JOIN musteriler m ON f.musteri_id = m.musteri_id
WHERE f.odendi_mi = false AND f.son_odeme_tarihi < CURRENT_DATE
ORDER BY f.son_odeme_tarihi ASC;

Örnek 8:
Soru: Hiç faturası kesilmemiş müşteriler var mı?
SQL:
SELECT m.ad, m.soyad, m.abonelik_no
FROM musteriler m
LEFT JOIN faturalar f ON f.musteri_id = m.musteri_id
WHERE f.fatura_id IS NULL;

Örnek 9:
Soru: Toplam kaç müşteri kayıtlı?
SQL:
SELECT COUNT(*) AS toplam_musteri
FROM musteriler;

Örnek 10:
Soru: Ortalama fatura tutarı ne kadar?
SQL:
SELECT AVG(f.tutar) AS ortalama_tutar
FROM faturalar f;

Örnek 11:
Soru: En yüksek su tüketimine sahip 3 okuma kaydı hangisi (endeks farkına göre)?
SQL:
SELECT m.ad, m.soyad, s.ilk_endeks, s.son_endeks, (s.son_endeks - s.ilk_endeks) AS endeks_farki
FROM su_tuketimi s
JOIN musteriler m ON s.musteri_id = m.musteri_id
ORDER BY endeks_farki DESC
LIMIT 3;
""".strip()

_PROMPT = ChatPromptTemplate.from_template("""Sen bir PostgreSQL uzmanısın. Aşağıdaki veritabanı şemasına göre,
kullanıcının doğal dil sorusunu TEK BİR SELECT sorgusuna çevir.

KURALLAR:
- Yalnızca SELECT sorgusu üret. INSERT/UPDATE/DELETE/DROP gibi hiçbir
  değiştirme komutu üretme.
- Yalnızca aşağıda tanımlanan tabloları ve kolonları kullan. Var olmayan bir
  tablo/kolon UYDURMA.
- Sonucunda SADECE SQL sorgusunu, ```sql ... ``` kod bloğu içinde döndür,
  başka açıklama ekleme.
- Sorgunun sonunda noktalı virgül (;) kullan, birden fazla komut ekleme.

--- ŞEMA ---
{sema}
--- ŞEMA SONU ---

--- ÖRNEKLER ---
{ornekler}
--- ÖRNEKLER SONU ---

Soru: {soru}
SQL:""")

_sql_uretme_zinciri = _PROMPT | _llm | StrOutputParser()


def _sql_bloğunu_ayikla(ham_metin: str) -> str:
    kod_blok = re.search(r"```(?:sql)?\s*(.*?)```", ham_metin, re.DOTALL | re.IGNORECASE)
    if kod_blok:
        return kod_blok.group(1).strip()
    select_konumu = ham_metin.upper().find("SELECT")
    if select_konumu == -1:
        return ham_metin.strip()
    return ham_metin[select_konumu:].strip()


def sql_guvenli_mi(sql: str) -> tuple[bool, str]:
    temiz_sql = sql.strip().rstrip(";").strip()
    sql_kucuk = temiz_sql.lower()

    if not sql_kucuk.startswith("select"):
        return False, "Sorgu SELECT ile başlamıyor."

    if ";" in temiz_sql:
        return False, "Sorguda birden fazla komut olabilir (noktalı virgül tespit edildi)."

    for kelime in YASAKLI_ANAHTAR_KELIMELER:
        if re.search(rf"\b{kelime}\b", sql_kucuk):
            return False, f"Yasaklı anahtar kelime tespit edildi: {kelime}"

    if "--" in temiz_sql or "/*" in temiz_sql:
        return False, "Sorguda yorum satırı (potansiyel injection) tespit edildi."

    tablo_taramasi_icin = re.sub(r"extract\s*\([^)]*\)", " ", sql_kucuk)

    kullanilan_tablolar = set(re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", tablo_taramasi_icin))
    bilinmeyen_tablolar = kullanilan_tablolar - IZINLI_TABLOLAR
    if bilinmeyen_tablolar:
        return False, f"Bilinmeyen/izinsiz tablo(lar): {bilinmeyen_tablolar}"

    if not kullanilan_tablolar:
        return False, "Sorguda hiçbir tablo referansı bulunamadı."

    return True, "Güvenli."


# ---------------------------------------------------------------------------
# LangGraph — State ve node'lar
# ---------------------------------------------------------------------------
class SqlAgentState(TypedDict):
    soru: str
    sql: Optional[str]
    guvenli_mi: Optional[bool]
    hata: Optional[str]
    sonuc: Optional[pd.DataFrame]
    dogal_dil_yanit: Optional[str]


def _sql_uret_node(state: SqlAgentState) -> dict:
    ham_metin = _sql_uretme_zinciri.invoke({
        "sema": _db.get_table_info(),
        "ornekler": FEW_SHOT_ORNEKLER,
        "soru": state["soru"],
    })
    sql = _sql_bloğunu_ayikla(ham_metin)
    logger.info("Üretilen SQL: %s", sql)
    return {"sql": sql}


def _guvenlik_kontrolu_node(state: SqlAgentState) -> dict:
    guvenli_mi, sebep = sql_guvenli_mi(state["sql"])
    if not guvenli_mi:
        logger.warning("SQL reddedildi: %s | SQL: %s", sebep, state["sql"])
        return {"guvenli_mi": False, "hata": f"Güvenlik kontrolünden geçemedi: {sebep}"}
    return {"guvenli_mi": True}


def _sql_calistir_node(state: SqlAgentState) -> dict:
    try:
        with _db._engine.connect() as conn:
            sonuc = pd.read_sql(state["sql"], conn)
        return {"sonuc": sonuc}
    except Exception as e:
        logger.error("SQL çalıştırma hatası: %s | SQL: %s", e, state["sql"])
        return {"hata": f"Sorgu çalıştırılamadı: {e}"}


def _dogal_dile_cevir_node(state: SqlAgentState) -> dict:
    try:
        yanit = sonucu_dogal_dile_cevir(state["soru"], state["sonuc"])
        return {"dogal_dil_yanit": yanit}
    except Exception as e:
        logger.error("Doğal dile çevirme hatası: %s", e)
        return {"hata": f"Sonuç doğal dile çevrilemedi: {e}"}


def _guvenlik_sonrasi_yonlendirme(state: SqlAgentState) -> str:
    return "calistir" if state.get("guvenli_mi") else END


def _calistirma_sonrasi_yonlendirme(state: SqlAgentState) -> str:
    # Gün 14 düzeltmesi: calistir hata verdiyse (state["hata"] dolu, state["sonuc"]
    # None kaldı) dogal_dile_cevir'e HİÇ gitme — None bir DataFrame'i doğal dile
    # çevirmeye çalışmak ikinci bir hataya yol açıyordu.
    return END if state.get("hata") else "dogal_dile_cevir"


_graph_builder = StateGraph(SqlAgentState)
_graph_builder.add_node("uret", _sql_uret_node)
_graph_builder.add_node("guvenlik", _guvenlik_kontrolu_node)
_graph_builder.add_node("calistir", _sql_calistir_node)
_graph_builder.add_node("dogal_dile_cevir", _dogal_dile_cevir_node)

_graph_builder.set_entry_point("uret")
_graph_builder.add_edge("uret", "guvenlik")
_graph_builder.add_conditional_edges("guvenlik", _guvenlik_sonrasi_yonlendirme, {"calistir": "calistir", END: END})
_graph_builder.add_conditional_edges(
    "calistir", _calistirma_sonrasi_yonlendirme, {"dogal_dile_cevir": "dogal_dile_cevir", END: END}
)
_graph_builder.add_edge("dogal_dile_cevir", END)

_sql_agent_graph = _graph_builder.compile()


def soruyu_sqlle_yanitla(soru: str) -> dict:
    """
    Router (Gün 14) tarafından çağrılacak ana giriş noktası.
    Dönen dict: {"sql": str, "sonuc": DataFrame | None, "dogal_dil_yanit": str | None, "hata": str | None}
    """
    baslangic_state = {
        "soru": soru, "sql": None, "guvenli_mi": None,
        "hata": None, "sonuc": None, "dogal_dil_yanit": None,
    }
    sonuc_state = _sql_agent_graph.invoke(baslangic_state)
    return {
        "sql": sonuc_state.get("sql"),
        "sonuc": sonuc_state.get("sonuc"),
        "dogal_dil_yanit": sonuc_state.get("dogal_dil_yanit"),
        "hata": sonuc_state.get("hata"),
    }


if __name__ == "__main__":
    soru = "Faturasını ödememiş müşterilerin adı, soyadı ve borç tutarı nedir?"
    sonuc = soruyu_sqlle_yanitla(soru)
    print("SQL:", sonuc["sql"])
    if sonuc["hata"]:
        print("HATA:", sonuc["hata"])
    else:
        print("\nDoğal dil yanıtı:")
        print(sonuc["dogal_dil_yanit"])
        print("\nHam sonuç (ilk 5 satır):")
        print(sonuc["sonuc"].head())

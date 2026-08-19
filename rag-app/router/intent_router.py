"""
router/intent_router.py

Gün 14 — Intent Router (LangGraph)

Akış:
    soru -> siniflandir (structured_query / document_query / hybrid_query)
    -> [hybrid_query ise] soru_ayir (SQL alt sorusu + RAG alt sorusu üretir)
    -> sql_dali (gerekliyse) -> rag_dali (gerekliyse) -> birlestir/bitir

Gün 14 içi düzeltme (bu değişiklik): hybrid sorularda RAG'a ve SQL'e ORİJİNAL
soru olduğu gibi gönderiliyordu. Test sırasında şu görüldü:
"Öztek Ülker'in son faturası ne kadar ve personelin etik ilkeleri konusunda
uyarılma duyurusunda neler bulunmaktadır?" sorusunda SQL kısmı doğru çalıştı,
ama RAG kısmı "bilgi yok" dedi — halbuki ilgili "Genelge" belgesi Qdrant'ta
tek başına sorulduğunda skor 1.0 ile birinci sırada çıkıyordu.

debug_ara.py ile karşılaştırma yapıldı:
  - Sadece "personelin etik ilkeleri konusunda uyarılma duyurusu" sorulunca
    -> doğru chunk skor 1.0 ile 1. sırada.
  - Tam hibrit soru (...faturası ne kadar VE etik ilkeleri...) sorulunca
    -> doğru chunk skor 0.25'e düşüp 6. sıraya geriliyor (eşik altı), yerine
       alakasız chunk'lar (mali rapor vb.) tesadüfen 0.50 skorla öne geçiyor.

Sebep: BGE-M3 embedding'i, iki farklı konuyu (fatura + etik duyurusu) içeren
tek bir soruyu vektörleştirirken anlamı "sulandırıyor" — vektör ikisinin
ortasında bir yerde kalıyor ve hiçbir konuyla tam eşleşmiyor.

Çözüm: hybrid_query sınıflandığında, LLM ile soru SQL'e özel alt soru ve
RAG'a özel alt soru olarak ikiye ayrılıyor (_soru_ayir_node). Her motor artık
sadece kendi konusuyla ilgili, "temiz" bir soru alıyor. structured_query ve
document_query (tek taraflı) sorularda ayırma adımı atlanır, orijinal soru
doğrudan kullanılır — çünkü zaten tek konulular, sulanma riski yok.

Kullanım:
    from router.intent_router import soruyu_yonlendir
    sonuc = soruyu_yonlendir("......?")
"""

import re
from typing import TypedDict, Optional, Literal

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from config import settings
from logging_config import get_logger
from sql_engine.text_to_sql import soruyu_sqlle_yanitla
from retrieval.rag_chain import soruyu_yanitla as rag_ile_yanitla

logger = get_logger(__name__)

Intent = Literal["structured_query", "document_query", "hybrid_query"]

_llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    keep_alive=settings.ollama_keep_alive,
    timeout=settings.ollama_timeout_sn,
)

# ---------------------------------------------------------------------------
# 1) Sınıflandırma
# ---------------------------------------------------------------------------
FEW_SHOT_SINIFLANDIRMA = """
Örnek 1:
Soru: Ahmet Yılmaz'ın son 3 faturası ne kadar?
Sınıf: structured_query

Örnek 2:
Soru: Bu ay kaç yeni müşteri kayıt oldu?
Sınıf: structured_query

Örnek 3:
Soru: Ödenmemiş faturası olan müşteriler kimler?
Sınıf: structured_query

Örnek 4:
Soru: Personelin etik ilkeleri konusunda hangi kurallara uyulması gerekiyor?
Sınıf: document_query

Örnek 5:
Soru: İZSU'nun stratejik planındaki ana hedefler nelerdir?
Sınıf: document_query

Örnek 6:
Soru: Su kesintisi duyurusu ne zaman yayınlandı ve içeriği nedir?
Sınıf: document_query

Örnek 7:
Soru: Ahmet Yılmaz'ın son faturası ne kadar ve itiraz süreci nasıl işler?
Sınıf: hybrid_query

Örnek 8:
Soru: En çok su tüketen 5 müşteri kim ve su tasarrufu için kurumun önerileri neler?
Sınıf: hybrid_query

Örnek 9:
Soru: Öztek Ülker'in son faturası ne kadar ve personelin etik ilkeleri konusunda uyarılma duyurusunda neler bulunmaktadır?
Sınıf: hybrid_query
""".strip()

_SINIFLANDIRMA_PROMPT = ChatPromptTemplate.from_template("""Aşağıdaki soruyu üç sınıftan birine ata:
- structured_query: Yalnızca veritabanındaki (müşteri, fatura, su tüketimi, adres) somut/sayısal
  bilgiyle yanıtlanabilir.
- document_query: Yalnızca kurumsal dokümanlardaki (yönetmelik, duyuru, rapor, plan vb.) metinsel
  bilgiyle yanıtlanabilir.
- hybrid_query: Sorunun bir kısmı veritabanı bilgisi, bir kısmı doküman bilgisi gerektiriyor.

Sadece sınıf adını yaz, başka hiçbir şey ekleme.

--- ÖRNEKLER ---
{ornekler}
--- ÖRNEKLER SONU ---

Soru: {soru}
Sınıf:""")

_siniflandirma_zinciri = _SINIFLANDIRMA_PROMPT | _llm | StrOutputParser()


def _intent_ayikla(ham_metin: str) -> Intent:
    metin = ham_metin.strip().lower()
    if "hybrid" in metin:
        return "hybrid_query"
    if "structured" in metin:
        return "structured_query"
    if "document" in metin:
        return "document_query"
    # Belirsiz çıktı: en güvenli varsayım hybrid — hem SQL hem RAG denenir,
    # yanlışlıkla bir bilgiyi atlamaktansa fazladan çalışmak tercih edilir.
    logger.warning("Sınıflandırma belirsiz, hybrid_query varsayılıyor. Ham çıktı: %r", ham_metin)
    return "hybrid_query"


# ---------------------------------------------------------------------------
# 2) Hibrit soru ayırma (Gün 14 düzeltmesi)
# ---------------------------------------------------------------------------
_SORU_AYIRMA_PROMPT = ChatPromptTemplate.from_template("""Aşağıdaki soru iki farklı konuyu birden içeriyor: biri veritabanı
(müşteri/fatura/su tüketimi) ile ilgili, diğeri kurumsal doküman (yönetmelik/duyuru/rapor/plan)
ile ilgili. Soruyu bu iki konuya göre ayrı ayrı, kendi başına anlamlı iki alt soruya böl.

KURALLAR:
- Her alt soru, orijinal sorudaki ilgili kısmın anlamını tam korumalı.
- Alt sorulara orijinal soruda GEÇMEYEN hiçbir bilgi ekleme.
- Tam olarak şu formatta yanıt ver, başka hiçbir şey ekleme:
SQL_SORU: <veritabanı ile ilgili alt soru>
RAG_SORU: <doküman ile ilgili alt soru>

Soru: {soru}
Yanıt:""")

_soru_ayirma_zinciri = _SORU_AYIRMA_PROMPT | _llm | StrOutputParser()


def _soru_ayikla(ham_metin: str, orijinal_soru: str) -> tuple[str, str]:
    sql_eslesme = re.search(r"SQL_SORU:\s*(.+)", ham_metin)
    rag_eslesme = re.search(r"RAG_SORU:\s*(.+)", ham_metin)
    sql_sorusu = sql_eslesme.group(1).strip() if sql_eslesme else orijinal_soru
    rag_sorusu = rag_eslesme.group(1).strip() if rag_eslesme else orijinal_soru
    return sql_sorusu, rag_sorusu


# ---------------------------------------------------------------------------
# 3) Hibrit birleştirme
# ---------------------------------------------------------------------------
_BIRLESTIRME_PROMPT = ChatPromptTemplate.from_template("""Aşağıda iki farklı kaynaktan gelen yanıt var: biri veritabanı sorgusundan
(SQL), diğeri kurumsal dokümanlardan (RAG). Bu ikisini, orijinal soruyu tek
ve akıcı bir Türkçe yanıt olarak cevaplayacak şekilde birleştir. Hiçbir
kaynakta olmayan bilgi uydurma.

Orijinal soru: {soru}

Veritabanı yanıtı: {sql_yanit}

Doküman yanıtı: {rag_yanit}

Birleştirilmiş yanıt:""")

_birlestirme_zinciri = _BIRLESTIRME_PROMPT | _llm | StrOutputParser()


# ---------------------------------------------------------------------------
# LangGraph — State ve node'lar
# ---------------------------------------------------------------------------
class RouterState(TypedDict):
    soru: str
    intent: Optional[Intent]
    sql_sorusu: Optional[str]
    rag_sorusu: Optional[str]
    sql_sonuc: Optional[dict]
    rag_sonuc: Optional[dict]
    nihai_yanit: Optional[str]


def _siniflandir_node(state: RouterState) -> dict:
    ham_metin = _siniflandirma_zinciri.invoke({"ornekler": FEW_SHOT_SINIFLANDIRMA, "soru": state["soru"]})
    intent = _intent_ayikla(ham_metin)
    logger.info("Router sınıflandırması: %s -> %s", state["soru"], intent)
    return {"intent": intent}


def _soru_ayir_node(state: RouterState) -> dict:
    ham_metin = _soru_ayirma_zinciri.invoke({"soru": state["soru"]})
    sql_sorusu, rag_sorusu = _soru_ayikla(ham_metin, state["soru"])
    logger.info("Soru ayrıldı -> SQL: %r | RAG: %r", sql_sorusu, rag_sorusu)
    return {"sql_sorusu": sql_sorusu, "rag_sorusu": rag_sorusu}


def _sql_dalini_calistir_node(state: RouterState) -> dict:
    soru = state.get("sql_sorusu") or state["soru"]
    return {"sql_sonuc": soruyu_sqlle_yanitla(soru)}


def _rag_dalini_calistir_node(state: RouterState) -> dict:
    soru = state.get("rag_sorusu") or state["soru"]
    return {"rag_sonuc": rag_ile_yanitla(soru)}


def _sadece_sql_bitir_node(state: RouterState) -> dict:
    sonuc = state["sql_sonuc"]
    yanit = sonuc.get("dogal_dil_yanit") or sonuc.get("hata") or "Yanıt üretilemedi."
    return {"nihai_yanit": yanit}


def _sadece_rag_bitir_node(state: RouterState) -> dict:
    return {"nihai_yanit": state["rag_sonuc"]["yanit"]}


def _hibrit_birlestir_node(state: RouterState) -> dict:
    sql_sonuc = state["sql_sonuc"]
    sql_yanit = sql_sonuc.get("dogal_dil_yanit") or sql_sonuc.get("hata") or "Veritabanından yanıt alınamadı."
    rag_yanit = state["rag_sonuc"]["yanit"]
    birlesik = _birlestirme_zinciri.invoke({"soru": state["soru"], "sql_yanit": sql_yanit, "rag_yanit": rag_yanit})
    return {"nihai_yanit": birlesik}


def _siniflandirma_sonrasi_yonlendirme(state: RouterState) -> str:
    if state["intent"] == "structured_query":
        return "sql_dali"
    if state["intent"] == "document_query":
        return "rag_dali"
    return "soru_ayir"  # hybrid_query


def _sql_dali_sonrasi_yonlendirme(state: RouterState) -> str:
    return "rag_dali" if state["intent"] == "hybrid_query" else "sadece_sql_bitir"


def _rag_dali_sonrasi_yonlendirme(state: RouterState) -> str:
    return "hibrit_birlestir" if state["intent"] == "hybrid_query" else "sadece_rag_bitir"


_graph_builder = StateGraph(RouterState)
_graph_builder.add_node("siniflandir", _siniflandir_node)
_graph_builder.add_node("soru_ayir", _soru_ayir_node)
_graph_builder.add_node("sql_dali", _sql_dalini_calistir_node)
_graph_builder.add_node("rag_dali", _rag_dalini_calistir_node)
_graph_builder.add_node("sadece_sql_bitir", _sadece_sql_bitir_node)
_graph_builder.add_node("sadece_rag_bitir", _sadece_rag_bitir_node)
_graph_builder.add_node("hibrit_birlestir", _hibrit_birlestir_node)

_graph_builder.set_entry_point("siniflandir")

_graph_builder.add_conditional_edges(
    "siniflandir",
    _siniflandirma_sonrasi_yonlendirme,
    {"sql_dali": "sql_dali", "rag_dali": "rag_dali", "soru_ayir": "soru_ayir"},
)
_graph_builder.add_edge("soru_ayir", "sql_dali")
_graph_builder.add_conditional_edges(
    "sql_dali", _sql_dali_sonrasi_yonlendirme, {"rag_dali": "rag_dali", "sadece_sql_bitir": "sadece_sql_bitir"}
)
_graph_builder.add_conditional_edges(
    "rag_dali", _rag_dali_sonrasi_yonlendirme, {"hibrit_birlestir": "hibrit_birlestir", "sadece_rag_bitir": "sadece_rag_bitir"}
)
_graph_builder.add_edge("sadece_sql_bitir", END)
_graph_builder.add_edge("sadece_rag_bitir", END)
_graph_builder.add_edge("hibrit_birlestir", END)

_router_graph = _graph_builder.compile()


def soruyu_yonlendir(soru: str) -> dict:
    """
    Ana giriş noktası (Streamlit arayüzü Gün 16-18'de bunu çağıracak).
    Dönen dict: {"intent", "nihai_yanit", "sql_sonuc", "rag_sonuc"}
    """
    baslangic_state = {
        "soru": soru, "intent": None, "sql_sorusu": None, "rag_sorusu": None,
        "sql_sonuc": None, "rag_sonuc": None, "nihai_yanit": None,
    }
    sonuc_state = _router_graph.invoke(baslangic_state)
    return {
        "intent": sonuc_state.get("intent"),
        "nihai_yanit": sonuc_state.get("nihai_yanit"),
        "sql_sonuc": sonuc_state.get("sql_sonuc"),
        "rag_sonuc": sonuc_state.get("rag_sonuc"),
    }


if __name__ == "__main__":
    soru = "Öztek Ülker'in son faturası ne kadar ve personelin etik ilkeleri konusunda uyarılma duyurusunda neler bulunmaktadır?"
    print("Soru:", soru)
    print("=" * 70)
    sonuc = soruyu_yonlendir(soru)
    print("Sınıf:", sonuc["intent"])
    print("Yanıt:", sonuc["nihai_yanit"])
    print("\n[Ham SQL sonucu]:", sonuc["sql_sonuc"])
    print("\n[Ham RAG sonucu]:", sonuc["rag_sonuc"]["yanit"] if sonuc["rag_sonuc"] else None)
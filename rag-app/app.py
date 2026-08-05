"""
Streamlit giris noktasi.
Gun 3 itibariyla bu dosya sadece iskelet -- config ve logging
altyapisinin dogru calistigini gostermek icin minimal bir arayuz
render ediyor. Router/SQL/RAG entegrasyonu ileriki gunlerde buraya
baglanacak.

Calistirma: streamlit run app.py
"""

import streamlit as st

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="İZSU Hibrit RAG Chatbot", page_icon="💧")

logger.info("Uygulama baslatildi (ortam: %s)", settings.app_env)

st.title("💧 İZSU Kurum İçi Akıllı Destek Chatbot")
st.caption("Hibrit RAG uygulaması — proje iskeleti (Gün 3)")

st.info(
    "Bu ekran şu an yalnızca bir yer tutucudur. "
    "Router, SQL motoru ve RAG bileşenleri sonraki günlerde buraya bağlanacaktır."
)

query = st.text_input("Sorgunuzu yazın:", placeholder="Örn. Uğur Eren'in son 3 aylık tüketimi nedir?")

if query:
    logger.info("Kullanıcı sorgusu alındı: %s", query)
    st.warning("Router henüz bağlanmadı — bu sorgu şu an işlenmiyor.")

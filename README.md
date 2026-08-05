# IZSU-Smart-Support-Chatbot-Institutional-
This project is an AI-powered internal support chatbot developed for İZSU employees. It leverages a Hybrid Retrieval-Augmented Generation (Hybrid RAG) architecture by combining structured SQL queries with semantic document retrieval to provide accurate and context-aware responses.



<img width="956" height="685" alt="image" src="https://github.com/user-attachments/assets/49fcae73-6590-4936-95d6-8e28b1d21bc8" />



# Hibrit RAG Uygulaması — İZSU Kurum İçi Akıllı Destek Chatbot

Yapılandırılmış veri (PostgreSQL) sorgulama ile doküman tabanlı RAG'i (Qdrant) birleştiren,
kullanıcı sorgusunu otomatik yönlendiren (router) hibrit bir destek chatbot'u.

> Staj projesi — Dilan Işık

## Proje Durumu

| Gün | Konu | Durum |
|---|---|---|
| 1 | Gereksinim analizi, mimari tasarım, veri şeması | ✅ Tamamlandı |
| 2 | Docker Compose, PostgreSQL, Qdrant, Python ortamı | ✅ Tamamlandı |
| 3 | Proje iskeleti, config/logging altyapısı | ✅ Tamamlandı |
| 4 | Sentetik veri üretimi | 🔄 Devam ediyor |

## Klasör Yapısı🔄 

```
rag-app/
├── data_generation/   # Faker ile sentetik veri üretimi (Gün 3-4)
├── db/                # SQLAlchemy modelleri, DB erişim katmanı
├── ingestion/         # Dokümanları Qdrant'a yükleme (embedding pipeline)
├── retrieval/         # Qdrant üzerinde vektör arama (RAG kolu)
├── sql_engine/        # Text-to-SQL motoru (SQL kolu)
├── router/            # Intent Classifier / sorgu yönlendirme
├── init-db/
│   └── init.sql       # PostgreSQL şema kurulum scripti (otomatik çalışır)
├── app.py             # Streamlit arayüzü
├── config.py          # Merkezi konfigürasyon (pydantic-settings, .env okur)
├── logging_config.py  # Merkezi loglama altyapısı
├── docker-compose.yml # PostgreSQL + Qdrant servisleri
├── requirements.txt
├── .env.example
└── test_baglanti.py   # PostgreSQL/Qdrant bağlantı doğrulama scripti
```

## Kurulum

```bash
# 1. Ortam değişkenlerini ayarla
cp .env.example .env

# 2. Docker servislerini ayağa kaldır (PostgreSQL + Qdrant)
docker compose up -d

# 3. Python sanal ortamını kur
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 4. Bağımlılıkları kur
pip install -r requirements.txt

# 5. Bağlantıları doğrula
python test_baglanti.py

# 6. Uygulamayı çalıştır
streamlit run app.py
```

## Teknolojiler

- **LLM & Embedding:** OpenAI API (ileride Google Gemini'ye geçiş değerlendiriliyor)
- **Vektör veritabanı:** Qdrant
- **Yapılandırılmış veri:** PostgreSQL
- **Orkestrasyon:** Python / LangChain
- **Arayüz:** Streamlit
- **Config:** pydantic-settings
- **Sentetik veri:** Faker

## Veri Şeması

`musteriler`, `adresler`, `su_tuketimi`, `faturalar` tablolarından oluşan normalize edilmiş bir
şema kullanılıyor. Detaylar için Gün 1 raporuna bakınız.

## Notlar

- Tüm veri sentetiktir; gerçek İZSU verisi kullanılmamaktadır.
- `.env` dosyası git'e commit edilmez (bkz. `.gitignore`).

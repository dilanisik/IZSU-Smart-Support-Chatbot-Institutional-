"""
Merkezi konfigurasyon modulu.
Tum .env degiskenleri buradan tek bir yerden okunur; projenin geri
kalani (db, retrieval, router, sql_engine vb.) bu modulu import eder.

Kullanim:
    from config import settings
    print(settings.postgres_url)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_pass"
    postgres_db: str = "izsu_db"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # --- LLM ---
    google_api_key: str | None = None  # NOT: Gün 9'da Gemini kaldırıldı, embedding tamamen yerel BGE-M3.
                                        # Bu alan şu an kullanılmıyor, ileride tamamen silinebilir.

    # --- RAG / Ollama (Gün 10) ---
    ollama_url: str = "http://10.100.17.144:11434/api/chat"
    ollama_model: str = "qwen3.5:27b"
    ollama_timeout_sn: int = 300       # soğuk model yüklemesi ~150-200sn sürebiliyor, pay bırakıldı
    ollama_keep_alive: str = "30m"     # model yüklendikten sonra bellekte kalma süresi
    rag_score_threshold: float = 0.55  # Gün 9-10 testlerinde gözlemlenen skor dağılımına göre başlangıç değeri

    # --- Uygulama ---
    log_level: str = "INFO"
    app_env: str = "development"  # development | production

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Tum proje boyunca tek bir ortak settings nesnesi kullanilir.
settings = Settings()

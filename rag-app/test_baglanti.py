"""
Baglanti dogrulama scripti.
PostgreSQL ve Qdrant baglantilarinin gercekten calistigini kontrol eder.
Artik ortak config.py ve logging_config.py altyapisini kullanir.

Calistirma: python test_baglanti.py  (venv aktifken, proje kok klasorunden)
"""

from config import settings
from logging_config import get_logger

logger = get_logger(__name__)


def test_postgres() -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(settings.postgres_url)

    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"
        ))
        tables = [row[0] for row in result]

    logger.info("PostgreSQL baglantisi basarili. Bulunan tablolar: %s", tables)


def test_qdrant() -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collections = client.get_collections()

    logger.info("Qdrant baglantisi basarili. Mevcut collection sayisi: %d", len(collections.collections))


if __name__ == "__main__":
    logger.info("--- Baglanti testleri basliyor ---")

    try:
        test_postgres()
    except Exception as e:
        logger.error("PostgreSQL baglantisi basarisiz: %s", e)

    try:
        test_qdrant()
    except Exception as e:
        logger.error("Qdrant baglantisi basarisiz: %s", e)

    logger.info("--- Test tamamlandi ---")

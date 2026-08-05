"""
Merkezi loglama altyapisi.
Projenin her modulu bu fonksiyonu cagirarak kendi logger'ini alir;
boylece tum loglar tutarli bir formatta ve tek bir yerden
yapilandirilmis olur.

Kullanim:
    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Baglanti kuruldu")
"""

import logging
import sys

from config import settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Verilen isimle (genelde __name__) yapilandirilmis bir logger dondurur."""
    _configure_root_logger()
    return logging.getLogger(name)

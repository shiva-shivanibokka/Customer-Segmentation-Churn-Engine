"""
Central logging configuration for the Customer Segmentation & Churn Engine.

Call configure_logging() once at pipeline/app entry points.
All modules obtain their logger via logging.getLogger(__name__).
"""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured, readable format."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        stream=sys.stdout,
    )

    # Silence noisy third-party loggers
    for noisy in ("mlflow", "numba", "umap", "urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

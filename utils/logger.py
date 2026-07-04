"""Logging setup for experiments."""

import logging
import os


def build_logger(log_path: str):
    """Create a console/file logger."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("LMCF-Net")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

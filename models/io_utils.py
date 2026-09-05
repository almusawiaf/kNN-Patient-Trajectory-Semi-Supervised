"""Pickle/JSON helpers and logging configuration."""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Any

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"


def setup_logging(logs_dir: str, name: str, level: str = "INFO") -> logging.Logger:
    """Log to stdout and to a timestamped file under ``logs_dir``."""
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(logs_dir, f"{name}_{stamp}.log")

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logger = logging.getLogger(name)
    logger.info("Logging to %s", logfile)
    return logger


def save_pickle(obj: Any, filename: str) -> None:
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filename, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logging.getLogger("io").info("Wrote %s", filename)


def load_pickle(filename: str) -> Any:
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def save_json(obj: Any, filename: str) -> None:
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filename, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    logging.getLogger("io").info("Wrote %s", filename)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)

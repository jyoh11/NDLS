"""Consistent console and rotating file logs for CLI and API execution."""
from __future__ import annotations
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Logging still works for CLI help/inspection before install.
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

def configure_logging(level: str | None = None, log_dir: str | Path = "logs") -> None:
    """Configure readable console logs and a rotating local log file."""
    load_dotenv(override=False)
    log_level = getattr(logging, (level or os.getenv("NIDS_LOG_LEVEL", "INFO")).upper(), logging.INFO)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(log_level)
    if root.handlers:
        return
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(Path(log_dir) / "nids.log", maxBytes=2_000_000,
                                       backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(console)
    root.addHandler(file_handler)

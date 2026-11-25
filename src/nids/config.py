"""Configuration loading and defaults used by CLI and API."""
from pathlib import Path
import os
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # Environment variables can still configure every supported override.
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

DEFAULT_CONFIG = Path("config.yaml")

def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load YAML settings and apply supported environment variable overrides."""
    load_dotenv(override=False)
    configured_path = os.getenv("NIDS_CONFIG") if str(path) == str(DEFAULT_CONFIG) else None
    source = Path(configured_path or path)
    with source.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    if os.getenv("NIDS_DATA_PATH"):
        config.setdefault("data", {})["path"] = os.environ["NIDS_DATA_PATH"]
    if os.getenv("NIDS_LABEL_COLUMN"):
        config.setdefault("data", {})["label_column"] = os.environ["NIDS_LABEL_COLUMN"]
    if os.getenv("NIDS_ARTIFACTS_DIR"):
        config.setdefault("training", {})["artifacts_dir"] = os.environ["NIDS_ARTIFACTS_DIR"]
    return config

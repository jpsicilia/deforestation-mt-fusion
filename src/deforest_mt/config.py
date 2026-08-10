"""Configuration loader.

Loads config/config.yaml once and exposes it plus resolved repo paths, so no
script hardcodes a study-area bound, a class code, or a file path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"

DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM: Path = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config into a nested dict.

    Args:
        path: Path to config.yaml (defaults to repo's config/config.yaml).

    Returns:
        Nested dict mirroring the YAML.

    Example:
        >>> cfg = load_config()
        >>> cfg["study_area"]["utm_crs"]
        'EPSG:31978'
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

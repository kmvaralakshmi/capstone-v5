"""Shared paths and small helpers for the agent files."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import (COMPANIES, ESG_METRICS, OUTPUT_FILES,  # noqa: E402
                          PROCESSED_DATA_DIR, BRSR_PDF_DIR, DATA_PATHS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import os  # noqa: E402

# BASE_PROCESSED holds shared caches; PROCESSED is where THIS run writes.
# On-demand runs set ESG_OUT_DIR so one company never overwrites batch results.
BASE_PROCESSED = Path(PROCESSED_DATA_DIR)
PROCESSED = Path(os.getenv("ESG_OUT_DIR")) if os.getenv("ESG_OUT_DIR") else BASE_PROCESSED
PROCESSED.mkdir(parents=True, exist_ok=True)
XBRL_RAW_DIR = ROOT / "raw-datasets" / "xbrl"


def out_path(key: str) -> Path:
    return PROCESSED / OUTPUT_FILES[key]


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def is_num(x) -> bool:
    try:
        return x is not None and not (isinstance(x, float) and np.isnan(x)) \
            and not pd.isna(x)
    except Exception:
        return False


def clean(obj):
    """Make a structure JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (obj != obj) else float(obj)
    return obj

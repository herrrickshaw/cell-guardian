"""
Loaders for real Kaggle battery datasets.

Download instructions
---------------------
Install Kaggle CLI and run from the repo root:

    pip install kaggle
    # Place kaggle.json (from kaggle.com → Account → API Token) in ~/.kaggle/

    mkdir -p data/kaggle
    kaggle datasets download patrickfleith/nasa-battery-dataset         -p data/kaggle/nasa         --unzip
    kaggle datasets download programmer3/lithium-ion-battery-degradation-dataset -p data/kaggle/degradation --unzip
    kaggle datasets download ziya07/ev-battery-charging-data            -p data/kaggle/ev_charging  --unzip
    kaggle datasets download ignaciovinuales/battery-remaining-useful-life-rul   -p data/kaggle/rul         --unzip
    kaggle datasets download akhileshdkapse/version21-bms-dataset       -p data/kaggle/bms_v21      --unzip
    kaggle datasets download micamadi/synthetic-distributed-battery-management-system -p data/kaggle/dist_bms --unzip

Each loader below finds the CSV automatically under its directory.
Falls back to the synthetic generator if the directory does not exist.
"""

from __future__ import annotations

import glob
import os
from typing import Callable, Optional

import pandas as pd

from .synthetic import (
    generate_bms_telemetry,
    generate_degradation,
    generate_distributed_bms,
    generate_ev_charging,
    generate_nasa_battery,
    generate_rul_features,
)

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kaggle")


def _first_csv(directory: str) -> Optional[str]:
    """Return the first CSV file found recursively in `directory`."""
    hits = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    return hits[0] if hits else None


def _load_or_synthesize(
    tag: str,
    directory: str,
    generator: Callable[[], pd.DataFrame],
    missing_hint: str = "",
) -> pd.DataFrame:
    """Read the first CSV under `directory`, else return `generator()` (synthetic data).

    `tag` labels the console message; `missing_hint` is appended to the fallback message.
    """
    csv = _first_csv(directory)
    if csv:
        df = pd.read_csv(csv)
        print(f"  [{tag}] loaded real data: {csv}  ({len(df):,} rows)")
        return df
    print(f"  [{tag}] real data not found — using synthetic{missing_hint}")
    return generator()


def load_nasa(directory: str = os.path.join(_BASE, "nasa")) -> pd.DataFrame:
    """
    Load the NASA battery dataset.
    Falls back to synthetic if the directory does not exist.
    """
    return _load_or_synthesize(
        "nasa", directory, generate_nasa_battery,
        " (run loaders.py download block)",
    )


def load_degradation(directory: str = os.path.join(_BASE, "degradation")) -> pd.DataFrame:
    """Load the Li-Ion Degradation dataset (cycle-level summaries)."""
    return _load_or_synthesize("degradation", directory, generate_degradation)


def load_ev_charging(directory: str = os.path.join(_BASE, "ev_charging")) -> pd.DataFrame:
    """Load the EV Battery Charging dataset."""
    return _load_or_synthesize("ev_charging", directory, generate_ev_charging)


def load_rul(directory: str = os.path.join(_BASE, "rul")) -> pd.DataFrame:
    """Load the Battery RUL feature dataset."""
    return _load_or_synthesize("rul", directory, generate_rul_features)


def load_bms_telemetry(directory: str = os.path.join(_BASE, "bms_v21")) -> pd.DataFrame:
    """Load the BMS v2.1 telemetry dataset."""
    return _load_or_synthesize("bms_v21", directory, generate_bms_telemetry)


def load_distributed_bms(directory: str = os.path.join(_BASE, "dist_bms")) -> pd.DataFrame:
    """Load the Synthetic Distributed BMS dataset."""
    return _load_or_synthesize("dist_bms", directory, generate_distributed_bms)


DOWNLOAD_COMMANDS = """
# ── Kaggle dataset download commands ─────────────────────────────────────────
# Run these from the repo root after placing kaggle.json in ~/.kaggle/

mkdir -p data/kaggle

kaggle datasets download patrickfleith/nasa-battery-dataset \\
    -p data/kaggle/nasa --unzip

kaggle datasets download programmer3/lithium-ion-battery-degradation-dataset \\
    -p data/kaggle/degradation --unzip

kaggle datasets download ziya07/ev-battery-charging-data \\
    -p data/kaggle/ev_charging --unzip

kaggle datasets download ignaciovinuales/battery-remaining-useful-life-rul \\
    -p data/kaggle/rul --unzip

kaggle datasets download akhileshdkapse/version21-bms-dataset \\
    -p data/kaggle/bms_v21 --unzip

kaggle datasets download micamadi/synthetic-distributed-battery-management-system \\
    -p data/kaggle/dist_bms --unzip
# ─────────────────────────────────────────────────────────────────────────────
"""

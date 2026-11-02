"""
bms.datasets — Kaggle battery dataset integration layer.

Datasets covered
----------------
1. NASA Battery (patrickfleith/nasa-battery-dataset)
   18650 Li-Co cells cycled to EOL; step-level V/I/T + capacity per cycle.

2. Li-Ion Degradation (programmer3/lithium-ion-battery-degradation-dataset)
   Cycle-level summary: SOH, capacity, resistance, RUL.

3. EV Battery Charging (ziya07/ev-battery-charging-data)
   Charging-session time-series: V, I, T, SOC.

4. Battery RUL (ignaciovinuales/battery-remaining-useful-life-rul)
   ML feature set (voltage stats + temperature) with RUL target.

5. BMS v2.1 (akhileshdkapse/version21-bms-dataset)
   Per-cell BMS telemetry: voltage, current, temperature, SOC, status.

6. Synthetic Distributed BMS (micamadi/synthetic-distributed-battery-management-system)
   Multi-module BMS sensor streams with health indicators.

Usage
-----
    from bms.datasets import load_all_synthetic, validate_bms_model

    datasets = load_all_synthetic()          # generate all six datasets
    report   = validate_bms_model(datasets)  # compare BMS predictions vs data

To use real Kaggle data (download CSVs first):
    from bms.datasets.loaders import load_nasa, load_degradation, load_ev_charging
    nasa = load_nasa("data/kaggle/nasa_battery/")
"""

from .loaders import (
    load_bms_telemetry,
    load_degradation,
    load_ev_charging,
    load_nasa,
    load_rul,
)
from .synthetic import (
    DATASET_REGISTRY,
    generate_bms_telemetry,
    generate_degradation,
    generate_distributed_bms,
    generate_ev_charging,
    generate_nasa_battery,
    generate_rul_features,
)
from .validation import print_validation_report, validate_bms_model

__all__ = [
    # Synthetic generators
    "generate_nasa_battery",
    "generate_degradation",
    "generate_ev_charging",
    "generate_rul_features",
    "generate_bms_telemetry",
    "generate_distributed_bms",
    "DATASET_REGISTRY",
    # Real-data loaders
    "load_nasa",
    "load_degradation",
    "load_ev_charging",
    "load_rul",
    "load_bms_telemetry",
    # Validation
    "validate_bms_model",
    "print_validation_report",
]


def load_all_synthetic() -> dict:
    """Generate all six datasets and return as a dict keyed by dataset name."""
    return {name: fn() for name, fn in DATASET_REGISTRY.items()}

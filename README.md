# cell-guardian

A Python battery-management-system (BMS) simulation toolkit: a 2-RC Thevenin cell model, an Extended
Kalman Filter (EKF) for state of charge (SOC), a state-of-health (SOH) tracker, a supervisory state
machine, and a pack controller that ties them together for eight cell chemistries and several vehicle
classes. It is for people studying or prototyping BMS algorithms for electric two-wheelers, cars and
buses. It produces console tables (charge/discharge traces, degradation and cold-temperature sweeps,
driving-style SOH projections, a validation report) and, optionally, a comparison plot.

> Status: research-only. Model parameters are literature-style presets, not manufacturer data; the
> validation runs on synthetic data shaped like the Kaggle datasets unless you download the real ones.
> Educational use only; not safety-certified and not for use in a real battery pack.

## What it does
- Simulates a cell with a 2-RC Thevenin equivalent circuit and a thermal model (`bms/cell_model.py`).
- Estimates SOC with an EKF over the state `[SOC, Vrc1, Vrc2]` and tracks capacity and resistance SOH.
- Runs a supervisory state machine (INIT, PRE_CHARGE, CC/CV/float charging, DISCHARGING, BALANCING,
  FAULT, SHUTDOWN) with passive cell balancing, power limits and a PWM fan/heater controller.
- Ships 8 chemistry profiles (LFP, PbA, NMC, Na-Ion, LMFP, LTO, Li-S, SS-NMC), 4 vehicle presets
  (2W, 4W_RETRO, 4W_MODERN, BUS), 4 manufacturer-style presets (Ather 450X, Ola S1 Pro, Tesla Model 3 SR,
  BYD Han) and 6 driving profiles (gentle, city, highway, aggressive, cold, regen).
- Projects long-term SOH loss from driving style (C-rate, temperature, depth-of-discharge stress, calendar ageing).
- Validates the EKF, the capacity-fade model, the voltage model, RUL and multi-module health detection
  against six Kaggle battery dataset schemas (synthetic stand-ins generated offline by default).

## Quick start
```bash
git clone https://github.com/herrrickshaw/cell-guardian.git
cd cell-guardian
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # numpy + pandas; add ".[plots]" for matplotlib
python -m bms.simulation --vehicle 2W
```

Other entry points (all under `python -m bms.simulation`):

| Flag | Output |
|---|---|
| *(none)* | comparison table, then 2W, 4W retrofit and bus charge/discharge runs |
| `--vehicle 2W\|4W_RETRO\|BUS` | a single vehicle run |
| `--plot` | also saves `reports/bms_multi_vehicle.png` (needs matplotlib) |
| `--degradation` | 300-cycle capacity/resistance fade tables |
| `--cold` | discharge sweep from +25 C to -30 C |
| `--driving` | 10-year SOH projection per driving profile |
| `--manufacturers` | Ather / Ola / Tesla / BYD pack and degradation comparison |
| `--datasets` | validation report against the Kaggle dataset schemas |

## How it works
```
config / chemistries -> cell_model (2-RC + thermal) -> soc_estimator (EKF) + soh_estimator
        -> cell_balancer, power_limits, thermal, supervisor -> bms_controller.step()
        -> simulation (scenarios, tables, plot)      datasets (generate / load / validate)
```
`BMSController.step(current, dt, ...)` advances every cell, runs one EKF per cell and a pack-level SOH estimator,
balances, derates power limits, drives the fan/heater and updates the supervisor, returning one
`BMSPackState` snapshot. `bms/simulation.py` builds scenarios on top of it: the per-second charge/discharge
runs use the controller; the degradation and driving projections are analytical models (no per-second steps).

EKF equations (`bms/soc_estimator.py`): SOC[k+1] = SOC[k] - I*dt/(Q*3600); each RC branch relaxes with
alpha = exp(-dt/(R*C)); measurement h(x) = OCV(SOC) - R0*I - Vrc1 - Vrc2. Noise and initial covariance
live in `EKFConfig` (`bms/config.py`); chemistry parameter tables live in `bms/chemistries.py`.

## Repository layout
| Path | Purpose |
|---|---|
| `bms/config.py`, `bms/chemistries.py` | cell/pack/EKF parameters; 8 chemistry profiles and registry |
| `bms/cell_model.py`, `bms/thermal.py` | 2-RC cell model with OCV tables; fan/heater controller |
| `bms/soc_estimator.py`, `bms/soh_estimator.py` | EKF SOC; capacity and resistance SOH |
| `bms/cell_balancer.py`, `bms/power_limits.py`, `bms/supervisor.py` | balancing, derating, state machine |
| `bms/bms_controller.py` | pack controller (`BMSController`, `BMSPackState`) |
| `bms/vehicles.py`, `bms/manufacturer_profiles.py`, `bms/driving_profiles.py` | presets and degradation stress model |
| `bms/simulation.py` | CLI scenarios and console tables |
| `bms/datasets/` | `synthetic.py` (seeded generators), `loaders.py` (real CSV or synthetic fallback), `validation.py` |
| `tests/` | unit tests, plus golden-digest regression tests (`tests/golden/`) |
| `download_kaggle_datasets.sh` | fetches the six Kaggle datasets into `data/kaggle/` (needs your Kaggle credentials) |
| `fix_and_push_kaggle.sh` | legacy helper that copies already-downloaded datasets into `data/kaggle/` and pushes a hard-coded branch |
| `nasa-battery-life-prediction-dataset-cleaning.ipynb` | NASA battery cleaning notebook |
| `HOW_TO_USE_WITH_AI.md` | how to explore the repo with an AI assistant |

## Data sources
| Source | What is taken | Access route | Licence / caveat |
|---|---|---|---|
| Kaggle `patrickfleith/nasa-battery-dataset` | NASA 18650 cycling data | `download_kaggle_datasets.sh` | Kaggle dataset terms; not redistributed here |
| Kaggle `programmer3/lithium-ion-battery-degradation-dataset` | cycle-level SOH/capacity | same | same |
| Kaggle `ziya07/ev-battery-charging-data` | charging sessions | same | same |
| Kaggle `ignaciovinuales/battery-remaining-useful-life-rul` | RUL features | same | same |
| Kaggle `akhileshdkapse/version21-bms-dataset` | per-cell BMS telemetry | same | same |
| Kaggle `micamadi/synthetic-distributed-battery-management-system` | multi-module telemetry | same | same |

Nothing is downloaded automatically; the code and tests never touch the network. Without the CSVs under
`data/kaggle/<name>/`, each loader falls back to a seeded synthetic generator with the same schema.

## Configuration
No environment variables. The Kaggle CLI reads `~/.kaggle/kaggle.json` (never committed). Pack, cell and
EKF parameters are frozen dataclasses in `bms/config.py`; chemistry and vehicle presets override them.

## Testing
```bash
pip install -e ".[dev]"
pytest -q                 # fast suite (unit + golden regression)
pytest -q -m slow         # adds the full bus and 4W retrofit runs (about a minute)
ruff check .
```
The golden digests in `tests/golden/characterisation.json` pin seeded simulation traces, dataset
generators and printed reports bit-for-bit.

## Limitations and known issues
- Validation numbers come from synthetic data unless real Kaggle CSVs are present, so they test internal
  consistency, not accuracy on real cells.
- The degradation and driving projections are analytical models with accelerated demo defaults (0.1 %/cycle).
- `fix_and_push_kaggle.sh` checks out a hard-coded branch (`claude/bms-battery-management-y26x71`) and is
  specific to the original author's setup.
- No licence file yet; all rights reserved. Kaggle datasets carry their own licences.

## Related repositories
This code overlaps with two sibling repos, which are kept separate:
- [bms-battery-management](https://github.com/herrrickshaw/bms-battery-management) - EV BMS cell models, SOC/SOH estimators, thermal and power-limit modules.
- [ev-battery-management-sim](https://github.com/herrrickshaw/ev-battery-management-sim) - EV battery-management simulation, NASA battery-life notebook, Kaggle validation scripts.

## Licence
No licence file yet — all rights reserved.

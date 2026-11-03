"""
Synthetic dataset generators that replicate the schema and published statistics
of six Kaggle battery datasets.

Each generator produces a pandas DataFrame whose columns match the real dataset
exactly — so analysis code written against the synthetic data works unchanged
when the real CSVs are substituted via bms/datasets/loaders.py.

Data is generated using the BMS package's own CellModel (2-RC Thevenin circuit)
so the physics are internally consistent and can be directly compared against
BMS predictions in bms/datasets/validation.py.

Published sources used for calibration
---------------------------------------
NASA PCoE:  B0005/B0006/B0018 18650 Li-Co; 1.5 A charge / 2 A discharge;
            168 cycles to 70 % SOH; ambient 24 °C.
MIT:        LFP/graphite fast-charge study; 124 cells; 150–2300 cycle life.
Lyten/NMC:  degradation dataset statistics from programmer3 Kaggle page.
"""

from dataclasses import replace as _dc_replace

import numpy as np
import pandas as pd

from ..cell_model import CellModel
from ..chemistries import LFP, NMC

# ---------------------------------------------------------------------------
# Shared RNG — pass seed= to each generator for reproducibility
# ---------------------------------------------------------------------------
_DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# 1. NASA Battery Dataset
#    Schema: cycle | battery_id | step_type | step_time_s | voltage_v |
#            current_a | temperature_c | capacity_ah | soh_pct | rul
#
#    Real stats: B0005 starts at 2.0 Ah, reaches ~1.4 Ah at cycle 168.
#    Chemistry: 18650 Li-Co (modelled here as NMC for closest match).
#    Charge  : 1.5 A CC → 4.2 V CV until current drops to 20 mA
#    Discharge: 2.0 A CC → 2.7 V cut-off
# ---------------------------------------------------------------------------
_NASA_CELL_CONFIG = _dc_replace(
    NMC.cell_config,
    nominal_capacity_ah=2.0,
    r0=0.080, r1=0.030, c1=1500.0, r2=0.045, c2=12000.0,
    i_max_charge=-1.5,
    i_max_discharge=2.0,
    i_cc_charge=-1.5,
    i_cv_term=-0.020,
    v_max=4.20, v_min=2.70,
    t_min_charge=0.0, t_max_charge=45.0,
    t_min_discharge=-10.0, t_max_discharge=60.0,
    t_nominal=25.0,
    thermal_capacity_j_k=30.0,
    thermal_resistance_k_w=15.0,
)


def generate_nasa_battery(
    n_cycles: int = 168,
    battery_ids: tuple = ("B0005", "B0006", "B0018"),
    seed: int = _DEFAULT_SEED,
    steps_per_cycle: int = 200,
) -> pd.DataFrame:
    """
    Synthetic NASA PCoE 18650 battery dataset.

    Parameters
    ----------
    n_cycles      : total charge/discharge cycles per battery
    battery_ids   : tuple of battery labels
    seed          : random seed for noise
    steps_per_cycle: time steps to record per discharge phase

    Returns
    -------
    DataFrame with one row per time step:
      cycle, battery_id, step_type, step_time_s, voltage_v, current_a,
      temperature_c, capacity_ah, soh_pct, rul
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Capacity fade: NASA B0005 goes from 2.0 Ah → 1.4 Ah over 168 cycles
    # Fitting a linear model: cap(n) = 2.0 - 0.00357 * n
    cap_0 = 2.0
    fade_per_cycle = (cap_0 - 1.4) / n_cycles   # ≈ 0.00357 Ah/cycle

    for bid in battery_ids:
        # Slight battery-to-battery variation (±2 %)
        b_scale = rng.uniform(0.98, 1.02)
        cap_b0 = cap_0 * b_scale
        fade_b = fade_per_cycle * b_scale

        for cyc in range(1, n_cycles + 1):
            cap_this = max(cap_b0 - fade_b * cyc, 0.5)
            soh = 100.0 * cap_this / cap_b0
            rul = n_cycles - cyc

            # Simulate one discharge using the scaled CellModel
            cfg_cyc = _dc_replace(_NASA_CELL_CONFIG, nominal_capacity_ah=cap_this)
            cell = CellModel(config=cfg_cyc, ocv_table=NMC.ocv_table, soc_init=1.0,
                             temp_init=24.0 + rng.uniform(-1.0, 1.0))
            dt = 1.0
            t = 0.0
            discharge_i = 2.0

            for step in range(steps_per_cycle):
                state = cell.step(current=discharge_i, dt=dt, ambient_temp=24.0)
                t += dt
                rows.append({
                    "cycle":         cyc,
                    "battery_id":    bid,
                    "step_type":     "discharge",
                    "step_time_s":   t,
                    "voltage_v":     round(state.v_terminal + rng.normal(0, 0.002), 4),
                    "current_a":     round(discharge_i + rng.normal(0, 0.005), 4),
                    "temperature_c": round(state.temperature + rng.normal(0, 0.1), 2),
                    "capacity_ah":   round(cap_this, 4),
                    "soh_pct":       round(soh, 2),
                    "rul":           rul,
                })
                if state.v_terminal <= 2.70 or state.soc <= 0.02:
                    break

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Li-Ion Degradation Dataset (programmer3)
#    Schema: cycle | voltage_avg_v | current_avg_a | temp_avg_c |
#            capacity_ah | soh_pct | r0_mohm | rul
#
#    Cycle-level summaries; NMC 18650; starts at 3.0 Ah → EOL ~2.1 Ah (300 cyc)
# ---------------------------------------------------------------------------

def generate_degradation(
    n_cycles: int = 300,
    nominal_capacity_ah: float = 3.0,
    eol_capacity_ah: float = 2.1,
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Synthetic Li-Ion degradation dataset (cycle-level summaries).

    Returns
    -------
    DataFrame: one row per cycle with averaged electrical + SOH metrics.
    """
    rng = np.random.default_rng(seed)
    rows = []

    cap_fade = (nominal_capacity_ah - eol_capacity_ah) / n_cycles
    r0_fresh = 0.045   # Ω for 18650 NMC 3 Ah cell
    # R0 grows roughly 0.5 mΩ/cycle (doubles from 45 → 90 mΩ by cycle 300)
    r0_growth = (0.090 - r0_fresh) / n_cycles

    for cyc in range(1, n_cycles + 1):
        soh = 100.0 * max(nominal_capacity_ah - cap_fade * cyc, 0.1) / nominal_capacity_ah
        cap = nominal_capacity_ah * soh / 100.0
        r0 = (r0_fresh + r0_growth * cyc) * 1000  # convert to mΩ
        rul = max(n_cycles - cyc, 0)

        # Add realistic cycle-to-cycle noise
        rows.append({
            "cycle":          cyc,
            "voltage_avg_v":  round(3.70 - 0.001 * cyc + rng.normal(0, 0.005), 4),
            "current_avg_a":  round(1.5 + rng.normal(0, 0.02), 4),
            "temp_avg_c":     round(26.0 + 0.005 * cyc + rng.normal(0, 0.3), 2),
            "capacity_ah":    round(cap + rng.normal(0, 0.005), 4),
            "soh_pct":        round(soh + rng.normal(0, 0.05), 2),
            "r0_mohm":        round(r0 + rng.normal(0, 0.3), 3),
            "rul":            rul,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. EV Battery Charging Dataset (ziya07)
#    Schema: timestamp_s | session_id | voltage_v | current_a | temperature_c |
#            soc_pct | power_kw | energy_kwh | charge_rate | phase
#
#    Charging session: CC phase (0.75C) then CV phase at 3.65V (LFP pack)
#    Pack: 22S LFP (Ola S1 Pro style), 70.4V, 56.8 Ah
# ---------------------------------------------------------------------------

def generate_ev_charging(
    n_sessions: int = 50,
    pack_voltage_v: float = 70.4,
    pack_capacity_ah: float = 56.8,
    soc_init_range: tuple = (0.15, 0.35),
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Synthetic EV battery charging session time-series (LFP pack, 22S1P).

    Returns
    -------
    DataFrame: one row per second across all sessions.
    """
    rng = np.random.default_rng(seed)
    rows = []

    cc_current = pack_capacity_ah * 0.75   # 0.75C ≈ 42.6 A

    for sid in range(1, n_sessions + 1):
        soc = rng.uniform(*soc_init_range)
        temp = rng.uniform(22.0, 38.0)     # varying ambient temperature
        t = 0
        energy_kwh = 0.0

        # Simple CC-CV model (not full BMS sim — matches dataset schema)
        v_cell = 3.20 + 0.45 * soc         # linear LFP OCV approximation
        v_pack = v_cell * 22

        while soc < 0.98:
            # CC phase: constant current until voltage limit
            if v_pack < pack_voltage_v * 0.995:
                phase = "CC"
                current = cc_current
            else:
                # CV phase: current decays exponentially
                phase = "CV"
                current = cc_current * np.exp(-0.002 * (t - 2000))
                if current < pack_capacity_ah * 0.05:
                    break

            soc += (current / pack_capacity_ah) / 3600.0
            soc = min(soc, 1.0)
            v_cell = 3.20 + 0.45 * soc
            v_pack = v_cell * 22
            power_kw = v_pack * current / 1000.0
            energy_kwh += power_kw / 3600.0
            temp += rng.normal(0.002, 0.001)   # mild self-heating

            rows.append({
                "timestamp_s": t,
                "session_id":  sid,
                "voltage_v":   round(v_pack + rng.normal(0, 0.05), 3),
                "current_a":   round(current + rng.normal(0, 0.1), 3),
                "temperature_c": round(temp, 2),
                "soc_pct":     round(soc * 100, 2),
                "power_kw":    round(power_kw + rng.normal(0, 0.01), 3),
                "energy_kwh":  round(energy_kwh, 4),
                "charge_rate": "DC_FAST" if current > 30 else "AC_SLOW",
                "phase":       phase,
            })
            t += 1
            if t > 14400:  # 4-hour cap
                break

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Battery RUL Feature Dataset (ignaciovinuales)
#    Schema: cycle | min_voltage_v | max_voltage_v | avg_voltage_v |
#            avg_current_a | min_temp_c | avg_temp_c | max_temp_c |
#            capacity_ah | rul
#
#    Feature-engineered from cycle-level statistics; ready for ML models.
# ---------------------------------------------------------------------------

def generate_rul_features(
    n_cycles: int = 500,
    nominal_capacity_ah: float = 2.0,
    eol_soh: float = 0.80,   # 80 % SOH = EOL threshold (industry standard)
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Synthetic RUL feature dataset (per-cycle statistics).

    Matches the schema used by ignaciovinuales/battery-remaining-useful-life-rul.
    Features are derived from the underlying degradation model with noise.

    Returns
    -------
    DataFrame: one row per cycle with voltage/current/temperature features + RUL.
    """
    rng = np.random.default_rng(seed)
    rows = []

    cap_0 = nominal_capacity_ah
    cap_eol = cap_0 * eol_soh                      # capacity at EOL (e.g. 1.6 Ah)
    fade_per_cycle = (cap_0 - cap_eol) / n_cycles  # linear fade rate
    # EOL is when capacity first crosses cap_eol — not necessarily n_cycles
    eol_cycle = int((cap_0 - cap_eol) / fade_per_cycle)  # = n_cycles by construction

    for cyc in range(1, n_cycles + 1):
        cap = max(cap_0 - fade_per_cycle * cyc, cap_eol * 0.5)
        # Voltage stats during discharge: min drops, avg drops, max relatively stable
        v_min = 2.80 - 0.001 * cyc + rng.normal(0, 0.01)
        v_max = 4.18 - 0.0002 * cyc + rng.normal(0, 0.003)
        v_avg = (v_min + v_max) / 2 + rng.normal(0, 0.005)
        i_avg = 2.0 + rng.normal(0, 0.03)
        t_min = 23.5 + rng.normal(0, 0.5)
        t_avg = 26.0 + 0.004 * cyc + rng.normal(0, 0.3)
        t_max = t_avg + 2.0 + rng.normal(0, 0.2)
        rul = max(eol_cycle - cyc, 0)  # cycles remaining until SOH = eol_soh

        rows.append({
            "cycle":         cyc,
            "min_voltage_v": round(v_min, 4),
            "max_voltage_v": round(v_max, 4),
            "avg_voltage_v": round(v_avg, 4),
            "avg_current_a": round(i_avg, 4),
            "min_temp_c":    round(t_min, 2),
            "avg_temp_c":    round(t_avg, 2),
            "max_temp_c":    round(t_max, 2),
            "capacity_ah":   round(cap + rng.normal(0, 0.003), 4),
            "rul":           rul,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. BMS v2.1 Telemetry Dataset (akhileshdkapse)
#    Schema: timestamp_s | cell_id | voltage_v | current_a | temperature_c |
#            soc_pct | status | fault_code | balancing_active
#
#    Per-cell BMS stream: 15 cells in series (48V LFP scooter pack)
# ---------------------------------------------------------------------------

def generate_bms_telemetry(
    duration_s: int = 3600,
    n_cells: int = 15,
    sample_hz: int = 1,
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Synthetic BMS v2.1 per-cell telemetry dataset.

    Simulates a 15S LFP scooter pack over a 1-hour discharge/charge session.
    Each cell gets a slightly different initial SOC to trigger passive balancing.

    Returns
    -------
    DataFrame: one row per (timestamp, cell_id) sample.
    """
    rng = np.random.default_rng(seed)
    rows = []
    dt = 1.0 / sample_hz

    # Build 15 cells with slight SOC spread to simulate real-world imbalance
    soc_spread = np.linspace(0.78, 0.82, n_cells)
    cells = [
        CellModel(config=LFP.cell_config, ocv_table=LFP.ocv_table,
                  soc_init=float(soc_spread[i]), temp_init=25.0 + rng.uniform(-1, 1))
        for i in range(n_cells)
    ]

    discharge_current = 30.0   # 0.3C for 100 Ah LFP cell
    # First 1800s discharge, then rest, then charge
    for tick in range(0, duration_s, int(1 / sample_hz)):
        t_s = tick * dt
        if t_s < 1800:
            current = discharge_current
            status = "DISCHARGING"
        elif t_s < 2100:
            current = 0.0
            status = "IDLE"
        else:
            current = -discharge_current * 0.6   # 0.18C slow charge
            status = "CHARGING"

        min_v = min(c.v_terminal for c in cells)
        max_v = max(c.v_terminal for c in cells)

        for cid, cell in enumerate(cells):
            state = cell.step(current=current, dt=dt, ambient_temp=25.0)
            v_spread = max_v - min_v
            bal = v_spread > 0.010 and state.v_terminal >= max_v - 0.003

            fault = "NONE"
            if state.v_terminal < LFP.cell_config.v_min + 0.05:
                fault = "UNDERVOLTAGE"
            elif state.temperature > 50.0:
                fault = "OVERTEMP"

            rows.append({
                "timestamp_s":      tick,
                "cell_id":          cid + 1,
                "voltage_v":        round(state.v_terminal + rng.normal(0, 0.001), 4),
                "current_a":        round(current + rng.normal(0, 0.05), 3),
                "temperature_c":    round(state.temperature + rng.normal(0, 0.1), 2),
                "soc_pct":          round(state.soc * 100, 2),
                "status":           status,
                "fault_code":       fault,
                "balancing_active": bal,
            })

        if any(c.v_terminal <= LFP.cell_config.v_min + 0.02 for c in cells):
            break

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Synthetic Distributed BMS (micamadi)
#    Schema: timestamp_s | module_id | cell_id | voltage_v | current_a |
#            temp_c | soc_pct | soh_pct | power_kw | alert
#
#    Multi-module pack: 4 modules × 8 cells = 32S (bus-scale ~102V LFP)
# ---------------------------------------------------------------------------

def generate_distributed_bms(
    duration_s: int = 1800,
    n_modules: int = 4,
    cells_per_module: int = 8,
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Synthetic multi-module distributed BMS dataset.

    Each module has its own health profile — module 3 is partially degraded
    (SOH 85 %) to demonstrate the health-aware balancing that the micamadi
    dataset demonstrates.

    Returns
    -------
    DataFrame: one row per (timestamp, module_id, cell_id).
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Module-level SOH profile (module 3 degraded to 85 %)
    module_soh = {1: 1.00, 2: 0.97, 3: 0.85, 4: 0.95}
    # Higher current makes the SOH difference visible within 1800 s
    discharge_i = 45.0   # 0.45C on nominal 100 Ah → ~45% DoD in 1800 s

    for mod_id in range(1, n_modules + 1):
        soh_m = module_soh[mod_id]
        # Each cell in the module has slight capacity variation
        mod_cells = [
            CellModel(
                config=_dc_replace(LFP.cell_config,
                                   nominal_capacity_ah=100.0 * soh_m * rng.uniform(0.98, 1.02)),
                ocv_table=LFP.ocv_table,
                soc_init=rng.uniform(0.60, 0.70),
                temp_init=25.0 + rng.uniform(0, 3),
            )
            for _ in range(cells_per_module)
        ]

        for tick in range(duration_s):
            for cid, cell in enumerate(mod_cells):
                state = cell.step(current=discharge_i, dt=1.0, ambient_temp=26.0)
                v_w = state.v_terminal * discharge_i / 1000.0

                alert = "OK"
                if state.soc < 0.15:
                    alert = "LOW_SOC"
                elif state.temperature > 45.0:
                    alert = "HIGH_TEMP"
                elif state.v_terminal < LFP.cell_config.v_min + 0.10:
                    alert = "UNDERVOLTAGE"

                rows.append({
                    "timestamp_s": tick,
                    "module_id":   mod_id,
                    "cell_id":     cid + 1,
                    "voltage_v":   round(state.v_terminal + rng.normal(0, 0.001), 4),
                    "current_a":   round(discharge_i + rng.normal(0, 0.02), 3),
                    "temp_c":      round(state.temperature + rng.normal(0, 0.05), 2),
                    "soc_pct":     round(state.soc * 100, 2),
                    "soh_pct":     round(soh_m * 100, 1),
                    "power_kw":    round(v_w, 4),
                    "alert":       alert,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "nasa_battery":       generate_nasa_battery,
    "degradation":        generate_degradation,
    "ev_charging":        generate_ev_charging,
    "rul_features":       generate_rul_features,
    "bms_telemetry":      generate_bms_telemetry,
    "distributed_bms":    generate_distributed_bms,
}

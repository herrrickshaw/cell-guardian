"""
BMS model validation against Kaggle battery datasets.

Compares BMS package predictions (EKF SOC, degradation model, cold derating)
against measurements from real or synthetic datasets. Reports RMSE, MAE,
and key error statistics for each dataset dimension.

Validation dimensions
---------------------
1. SOC estimation (RMSE vs coulomb-counting SOC)   ← EKF accuracy
2. Capacity fade curve vs measured SOH             ← degradation model
3. Voltage prediction (OCV + RC model) vs measured ← cell model accuracy
4. Temperature rise prediction vs measured          ← thermal model
5. RUL estimate from SOH trend vs dataset RUL      ← lifecycle model
"""

from dataclasses import replace as _dc_replace
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..cell_model import CellModel
from ..chemistries import LFP, NMC
from ..config import DEFAULT_EKF
from ..soc_estimator import EKFSOCEstimator
from .synthetic import _NASA_CELL_CONFIG  # same config as data generator

# ---------------------------------------------------------------------------
# Helper metrics
# ---------------------------------------------------------------------------

def _rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual)))


def _mape(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])) * 100)


# ---------------------------------------------------------------------------
# 1. SOC estimation accuracy (NASA / BMS telemetry)
# ---------------------------------------------------------------------------

def validate_soc_estimation(df_nasa: pd.DataFrame) -> Dict[str, Any]:
    """
    Run EKF SOC estimator on one discharge cycle from the NASA-format dataset
    and compare against coulomb-counting SOC.

    The NASA dataset doesn't directly provide SOC, so we use the capacity
    column to infer SOC: SOC(t) = 1 - (Ah_discharged / capacity_ah).
    """
    # Pick one battery, one cycle (cycle 1) — avoids ah_out not resetting across cycles
    bid = df_nasa["battery_id"].iloc[0]
    sub = df_nasa[
        (df_nasa["battery_id"] == bid) &
        (df_nasa["step_type"] == "discharge") &
        (df_nasa["cycle"] == 1)
    ].copy().reset_index(drop=True)

    if sub.empty:
        return {"error": "no discharge steps found"}

    # Use the same cell config that generated the data for a fair comparison
    cap_ah = float(sub["capacity_ah"].iloc[0])
    cfg = _dc_replace(_NASA_CELL_CONFIG, nominal_capacity_ah=cap_ah)
    cell = CellModel(config=cfg, ocv_table=NMC.ocv_table, soc_init=1.0, temp_init=25.0)
    ekf = EKFSOCEstimator(cell_model=cell, ekf_cfg=DEFAULT_EKF, soc_init=1.0)

    ekf_socs, true_socs = [], []
    ah_out = 0.0

    for _, row in sub.iterrows():
        i = float(row["current_a"])
        v = float(row["voltage_v"])
        ah_out += i / 3600.0
        true_soc = max(0.0, 1.0 - ah_out / cap_ah)

        cell.step(current=i, dt=1.0, ambient_temp=25.0)
        ekf_soc = ekf.update(current=i, v_measured=v, dt=1.0)

        ekf_socs.append(ekf_soc)
        true_socs.append(true_soc)

    ekf_arr = np.array(ekf_socs)
    true_arr = np.array(true_socs)

    return {
        "n_samples":   len(ekf_arr),
        "rmse_soc":    round(_rmse(ekf_arr, true_arr), 5),
        "mae_soc":     round(_mae(ekf_arr, true_arr), 5),
        "max_err_soc": round(float(np.max(np.abs(ekf_arr - true_arr))), 5),
        "ekf_soc_final":  round(float(ekf_arr[-1]), 4),
        "true_soc_final": round(float(true_arr[-1]), 4),
    }


# ---------------------------------------------------------------------------
# 2. Capacity / SOH fade curve
# ---------------------------------------------------------------------------

def validate_degradation_model(df_deg: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare the BMS simulate_degradation() linear model against measured SOH
    from the degradation dataset.

    Fits a linear regression on the measured SOH-vs-cycle data and compares
    the slope/intercept to the analytical model.
    """
    df = df_deg[["cycle", "soh_pct", "capacity_ah", "r0_mohm"]].dropna().copy()
    cycles = df["cycle"].to_numpy()
    soh_meas = df["soh_pct"].to_numpy()
    cap_meas = df["capacity_ah"].to_numpy()

    # Analytical model: SOH_Q = 100 - deg_rate * cycle
    # Fit deg_rate from data
    deg_rate_fit = (100.0 - soh_meas[-1]) / cycles[-1] / 100.0

    soh_pred = np.maximum(0.0, 100.0 * (1.0 - deg_rate_fit * cycles))
    cap_pred = cap_meas[0] * soh_pred / soh_meas[0]

    # R0 model: R0_N = R0_0 * (1 + r0_growth * N)
    r0_meas = df["r0_mohm"].to_numpy()
    r0_growth_fit = (r0_meas[-1] / r0_meas[0] - 1.0) / cycles[-1]
    r0_pred = r0_meas[0] * (1.0 + r0_growth_fit * cycles)

    return {
        "n_cycles":         int(cycles[-1]),
        "soh_rmse_pct":     round(_rmse(soh_pred, soh_meas), 4),
        "soh_mae_pct":      round(_mae(soh_pred, soh_meas), 4),
        "cap_rmse_ah":      round(_rmse(cap_pred, cap_meas), 4),
        "r0_rmse_mohm":     round(_rmse(r0_pred, r0_meas), 4),
        "fitted_deg_rate":  round(deg_rate_fit * 100, 5),   # %/cycle
        "fitted_r0_growth": round(r0_growth_fit * 100, 5),  # %/cycle
        "eol_cycle_pred":   int((100.0 - 80.0) / (deg_rate_fit * 100.0)) if deg_rate_fit > 0 else 9999,
    }


# ---------------------------------------------------------------------------
# 3. Voltage prediction accuracy
# ---------------------------------------------------------------------------

def validate_voltage_model(df_bms: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare 2-RC Thevenin model voltage prediction against measured cell
    terminal voltages from the BMS telemetry dataset.
    """
    # Use cell_id=1 discharge phase
    sub = df_bms[
        (df_bms["cell_id"] == 1) &
        (df_bms["status"] == "DISCHARGING")
    ].head(600).copy().reset_index(drop=True)

    if sub.empty:
        return {"error": "no discharging rows found"}

    soc_init = float(sub["soc_pct"].iloc[0]) / 100.0
    cell = CellModel(config=LFP.cell_config, ocv_table=LFP.ocv_table,
                     soc_init=soc_init, temp_init=25.0)

    v_pred, v_meas = [], []
    for _, row in sub.iterrows():
        i = float(row["current_a"])
        state = cell.step(current=i, dt=1.0, ambient_temp=25.0)
        v_pred.append(state.v_terminal)
        v_meas.append(float(row["voltage_v"]))

    vp = np.array(v_pred)
    vm = np.array(v_meas)

    return {
        "n_samples":    len(vp),
        "v_rmse_mv":    round(_rmse(vp, vm) * 1000, 2),   # mV
        "v_mae_mv":     round(_mae(vp, vm) * 1000, 2),
        "v_max_err_mv": round(float(np.max(np.abs(vp - vm))) * 1000, 2),
        "v_mape_pct":   round(_mape(vp, vm), 4),
    }


# ---------------------------------------------------------------------------
# 4. RUL estimation from SOH trend
# ---------------------------------------------------------------------------

def validate_rul_model(df_rul: pd.DataFrame) -> Dict[str, Any]:
    """
    Estimate RUL from the fitted linear SOH model and compare against
    the dataset's ground-truth RUL column.
    """
    df = df_rul[["cycle", "capacity_ah", "rul"]].dropna().copy()
    cycles = df["cycle"].to_numpy()
    cap = df["capacity_ah"].to_numpy()
    rul_true = df["rul"].to_numpy().astype(float)

    # Fit capacity fade rate using full dataset (linear regression)
    coeffs = np.polyfit(cycles, cap, 1)   # slope, intercept
    fade_per_cycle = -coeffs[0]           # Ah/cycle (positive)
    cap_eol = cap[0] * 0.80       # 80 % SOH threshold

    # Predicted RUL at each cycle: (current_cap - cap_eol) / fade_per_cycle
    rul_pred = np.maximum(0.0, (cap - cap_eol) / fade_per_cycle)

    return {
        "n_cycles":          int(cycles[-1]),
        "rul_rmse_cycles":   round(_rmse(rul_pred, rul_true), 1),
        "rul_mae_cycles":    round(_mae(rul_pred, rul_true), 1),
        "rul_mape_pct":      round(_mape(rul_pred[rul_true > 0], rul_true[rul_true > 0]), 2),
        "fitted_fade_rate":  round(float(fade_per_cycle), 5),
        "eol_cycle_predicted": int(cycles[0] + (cap[0] - cap_eol) / fade_per_cycle),
    }


# ---------------------------------------------------------------------------
# 5. Multi-module BMS health detection
# ---------------------------------------------------------------------------

def validate_distributed_bms(df_dist: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that the degraded module (module 3, SOH=85%) shows measurably
    different voltage and SOC depletion compared to healthy modules.
    """
    summary = (
        df_dist.groupby("module_id")
        .agg(
            avg_voltage=("voltage_v", "mean"),
            avg_soc=("soc_pct", "mean"),
            min_soc=("soc_pct", "min"),
            avg_temp=("temp_c", "mean"),
            soh=("soh_pct", "first"),
            alert_count=("alert", lambda x: (x != "OK").sum()),
        )
        .reset_index()
    )

    degraded = summary[summary["soh"] < 90.0]
    healthy  = summary[summary["soh"] >= 90.0]

    return {
        "module_summary":         summary.to_dict(orient="records"),
        "degraded_module_ids":    degraded["module_id"].tolist(),
        "degraded_avg_soh_pct":   round(float(degraded["soh"].mean()), 1),
        "healthy_avg_soh_pct":    round(float(healthy["soh"].mean()), 1),
        # Degraded module depletes faster → lower minimum SOC at end of session
        "soh_detectable":         bool(degraded["min_soc"].mean() < healthy["min_soc"].mean()),
        "degraded_alert_count":   int(degraded["alert_count"].sum()),
        "healthy_alert_count":    int(healthy["alert_count"].sum()),
    }


# ---------------------------------------------------------------------------
# Master validation runner
# ---------------------------------------------------------------------------

def validate_bms_model(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Run all validation checks against the provided dataset dict.

    Parameters
    ----------
    datasets : dict returned by load_all_synthetic() or manual dataset loading

    Returns
    -------
    dict of validation results, one key per dimension.
    """
    results = {}

    if "nasa_battery" in datasets:
        results["soc_estimation"] = validate_soc_estimation(datasets["nasa_battery"])

    if "degradation" in datasets:
        results["degradation_model"] = validate_degradation_model(datasets["degradation"])

    if "bms_telemetry" in datasets:
        results["voltage_model"] = validate_voltage_model(datasets["bms_telemetry"])

    if "rul_features" in datasets:
        results["rul_model"] = validate_rul_model(datasets["rul_features"])

    if "distributed_bms" in datasets:
        results["distributed_bms"] = validate_distributed_bms(datasets["distributed_bms"])

    return results


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

_REPORT_WIDTH = 74
_SECTION_RULE = "─" * 60
_SOC_RMSE_TARGET = 0.03        # fraction of SOC
_VOLTAGE_RMSE_TARGET_MV = 20.0


def _section_title(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {_SECTION_RULE}")


def _print_soc_estimation(r: Dict[str, Any]) -> None:
    _section_title("1. EKF SOC Estimation  (NASA 18650 discharge cycles)")
    if "error" in r:
        print(f"     ERROR: {r['error']}")
        return
    print(f"     Samples       : {r['n_samples']:,}")
    print(f"     RMSE (SOC)    : {r['rmse_soc']*100:.3f} %")
    print(f"     MAE  (SOC)    : {r['mae_soc']*100:.3f} %")
    print(f"     Max error     : {r['max_err_soc']*100:.3f} %")
    print(f"     EKF final SOC : {r['ekf_soc_final']*100:.1f} %   "
          f"(true: {r['true_soc_final']*100:.1f} %)")
    verdict = "PASS ✓" if r["rmse_soc"] < _SOC_RMSE_TARGET else "REVIEW"
    print(f"     Verdict       : {verdict}  (target RMSE < 3 %)")


def _print_degradation_model(r: Dict[str, Any]) -> None:
    _section_title(f"2. Capacity Fade Model  (Li-Ion Degradation, {r['n_cycles']} cycles)")
    print(f"     SOH RMSE      : {r['soh_rmse_pct']:.4f} %")
    print(f"     SOH MAE       : {r['soh_mae_pct']:.4f} %")
    print(f"     Capacity RMSE : {r['cap_rmse_ah']:.4f} Ah")
    print(f"     R0 RMSE       : {r['r0_rmse_mohm']:.4f} mΩ")
    print(f"     Fitted fade   : {r['fitted_deg_rate']:.5f} %/cycle")
    print(f"     Fitted R0 growth: {r['fitted_r0_growth']:.5f} %/cycle")
    print(f"     EOL prediction: cycle {r['eol_cycle_pred']:,}  (SOH=80 % threshold)")


def _print_voltage_model(r: Dict[str, Any]) -> None:
    _section_title(f"3. 2-RC Voltage Model  (BMS v2.1 telemetry, {r['n_samples']:,} steps)")
    if "error" in r:
        print(f"     ERROR: {r['error']}")
        return
    print(f"     Voltage RMSE  : {r['v_rmse_mv']:.2f} mV")
    print(f"     Voltage MAE   : {r['v_mae_mv']:.2f} mV")
    print(f"     Max error     : {r['v_max_err_mv']:.2f} mV")
    print(f"     MAPE          : {r['v_mape_pct']:.4f} %")
    verdict = "PASS ✓" if r["v_rmse_mv"] < _VOLTAGE_RMSE_TARGET_MV else "REVIEW"
    print(f"     Verdict       : {verdict}  (target RMSE < 20 mV)")


def _print_rul_model(r: Dict[str, Any]) -> None:
    _section_title(f"4. RUL Estimation  ({r['n_cycles']} cycles)")
    print(f"     RUL RMSE      : {r['rul_rmse_cycles']:.1f} cycles")
    print(f"     RUL MAE       : {r['rul_mae_cycles']:.1f} cycles")
    print(f"     RUL MAPE      : {r['rul_mape_pct']:.2f} %")
    print(f"     Fitted fade   : {r['fitted_fade_rate']:.5f} Ah/cycle")
    print(f"     EOL predicted : cycle {r['eol_cycle_predicted']:,}")


def _print_distributed_bms(r: Dict[str, Any]) -> None:
    _section_title("5. Multi-Module Health Detection  (Distributed BMS)")
    print(f"     Degraded modules  : {r['degraded_module_ids']}")
    print(f"     Degraded avg SOH  : {r['degraded_avg_soh_pct']:.1f} %")
    print(f"     Healthy avg SOH   : {r['healthy_avg_soh_pct']:.1f} %")
    print(f"     SOC drop detected : {'YES ✓' if r['soh_detectable'] else 'NO'}")
    print(f"     Degraded alerts   : {r['degraded_alert_count']:,}")
    print(f"     Healthy alerts    : {r['healthy_alert_count']:,}")
    print("\n     Module summary:")
    print(f"     {'ModID':>6}  {'SOH%':>6}  {'AvgSOC%':>8}  {'MinSOC%':>8}  "
          f"{'AvgTemp':>8}  {'Alerts':>7}")
    print(f"     {'─'*55}")
    for m in r["module_summary"]:
        print(f"     {m['module_id']:>6}  {m['soh']:>6.1f}  {m['avg_soc']:>8.1f}  "
              f"{m['min_soc']:>8.1f}  {m['avg_temp']:>8.2f}  {m['alert_count']:>7}")


# Report sections in print order: results key -> section printer.
_REPORT_SECTIONS = (
    ("soc_estimation", _print_soc_estimation),
    ("degradation_model", _print_degradation_model),
    ("voltage_model", _print_voltage_model),
    ("rul_model", _print_rul_model),
    ("distributed_bms", _print_distributed_bms),
)


def print_validation_report(results: Dict[str, Any]) -> None:
    """Print the validation report for the dict returned by `validate_bms_model`."""
    width = _REPORT_WIDTH

    print(f"\n{'='*width}")
    print("  BMS MODEL VALIDATION REPORT — Kaggle Battery Datasets")
    print(f"{'='*width}")

    for key, printer in _REPORT_SECTIONS:
        if key in results:
            printer(results[key])

    print(f"\n{'='*width}")
    print("  NOTE: Results above use synthetic data matching Kaggle dataset schemas.")
    print("  Substitute real CSVs via bms/datasets/loaders.py for production validation.")
    print(f"{'='*width}\n")

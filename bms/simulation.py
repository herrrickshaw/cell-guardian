"""
BMS multi-vehicle desktop simulation.

Demonstrates the BMS running across three EV classes:
  1. Electric 2-Wheeler  (48V LFP scooter)
  2. Retrofit 4-Wheeler  (72V Lead-Acid, IJAREEIE 2019 paper)
  3. Electric Bus        (409.6V LFP, 150 kW)

Also includes:
  4. Degradation simulation — SOH capacity/resistance fade over N cycles.
  5. Cold-temperature simulation — performance and fault behavior at low temperatures.
  6. Driving conditions simulation — how driving style affects long-term SOH.
  7. Dataset validation — BMS model accuracy against 6 Kaggle battery datasets.

Run:
    python -m bms.simulation                  # all three vehicles
    python -m bms.simulation --vehicle 2W     # single vehicle
    python -m bms.simulation --plot           # save plots to reports/
    python -m bms.simulation --degradation    # capacity fade over 300 cycles
    python -m bms.simulation --cold           # cold temperature sweep
    python -m bms.simulation --driving        # driving conditions degradation
    python -m bms.simulation --manufacturers  # Ather/Ola/Tesla/BYD comparison
    python -m bms.simulation --datasets       # validate against Kaggle datasets
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np

from .bms_controller import BMSPackState
from .projections import (  # noqa: F401  (re-exported: old import paths keep working)
    _left_aligned_format,
    _soh_marker,
    print_comparison_table,
    simulate_driving_degradation,
    simulate_manufacturer_comparison,
)
from .vehicles import (
    ELECTRIC_BUS,
    FOUR_WHEELER_RETRO,
    TWO_WHEELER,
    VehicleProfile,
    make_bms_controller,
)


def _header(profile: VehicleProfile):
    print(f"\n{profile.summary()}")
    cols = ["Time(s)", "State", "SOC%", "V_pack(V)", "I(A)", "T_max(C)", "Fan%", "Alerts"]
    print(f"  {'|'.join(f'{c:>10}' for c in cols)}")
    print("  " + "-" * (11 * len(cols)))


def _row(state: BMSPackState, t: float, interval: int):
    if int(t) % interval != 0:
        return
    alert_str = ",".join(a.name for a in state.alerts) if state.alerts else "-"
    cols = [
        f"{t:>10.0f}",
        f"{state.supervisor.state.name:>10}",
        f"{state.pack_soc_mean*100:>10.1f}",
        f"{state.pack_voltage_v:>10.2f}",
        f"{state.pack_current_a:>10.1f}",
        f"{state.pack_temp_max_c:>10.1f}",
        f"{state.thermal.fan_pwm_pct:>10.1f}",
        f"{alert_str:>10}",
    ]
    print("  " + "|".join(cols))


def _simulate_charge_discharge(
    profile: VehicleProfile,
    soc_init: float = 0.20,
    dt: float = 1.0,
    charge_s: int = 5400,
    discharge_s: int = 3600,
    discharge_current: Optional[float] = None,
    print_interval: int = 600,
) -> List[BMSPackState]:
    """Run a CC-CV charge followed by constant-current discharge."""

    # Slight SOC spread across cells to show balancing
    n = profile.n_cells_series
    spread = 0.02
    socs = [float(np.clip(soc_init + spread * (i / max(n - 1, 1) - 0.5), 0.0, 1.0))
            for i in range(n)]

    bms = make_bms_controller(profile, soc_init=socs, ambient_temp_c=25.0)
    history: List[BMSPackState] = []

    _header(profile)

    # ---- Phase 1: CC-CV Charge ----
    cc_i = profile.cell_config.i_max_charge * profile.n_cells_parallel
    cv_term = abs(profile.cell_config.i_cv_term) * profile.n_cells_parallel
    in_cv = False
    cv_start = 0.0
    t = 0.0

    for tick in range(charge_s):
        t = tick * dt
        if not in_cv:
            charge_current = cc_i
        else:
            decay = np.exp(-(t - cv_start) / 1200.0)
            charge_current = cc_i * decay
            if abs(charge_current) < cv_term:
                break

        state = bms.step(current=charge_current, dt=dt, charge_requested=True)
        history.append(state)
        _row(state, t, print_interval)

        if not in_cv and max(state.cell_voltages_v) >= profile.cell_config.v_max:
            in_cv = True
            cv_start = t
        if state.pack_soc_mean >= 0.95:
            break

    print(f"\n  >> Charge done  SOC={history[-1].pack_soc_mean*100:.1f}%  "
          f"V={history[-1].pack_voltage_v:.1f}V  T={history[-1].pack_temp_max_c:.1f}°C")

    # ---- Phase 2: Balancing at rest ----
    for _ in range(min(300, charge_s)):
        t += dt
        state = bms.step(current=0.0, dt=dt)
        history.append(state)
        if state.balancing.v_spread < 0.003:
            break

    # ---- Phase 3: CC Discharge ----
    if discharge_current is None:
        discharge_current = profile.cell_config.i_max_discharge * profile.n_cells_parallel * 0.5

    for tick in range(discharge_s):
        t += dt
        state = bms.step(current=discharge_current, dt=dt, discharge_requested=True)
        history.append(state)
        _row(state, t, print_interval)
        if state.pack_soc_mean <= 0.05 or state.fault_active:
            break

    last = history[-1]
    print(f"\n  >> Discharge done  SOC={last.pack_soc_mean*100:.1f}%  "
          f"Fault={last.fault_description}  "
          f"SOH={last.soh.soh_capacity_pct:.2f}%  "
          f"Fan={last.thermal.fan_pwm_pct:.0f}%\n")

    return history


# ---------------------------------------------------------------------------
# Per-vehicle simulation functions
# ---------------------------------------------------------------------------

def simulate_two_wheeler(**kwargs) -> List[BMSPackState]:
    """Electric scooter / e-bike: 15S LFP, 48V, 1.5 kW."""
    return _simulate_charge_discharge(
        TWO_WHEELER,
        soc_init=0.15,
        charge_s=3600,
        discharge_s=2400,
        discharge_current=20.0,     # 0.67C — typical scooter load
        print_interval=600,
        **kwargs,
    )


def simulate_four_wheeler_retrofit(**kwargs) -> List[BMSPackState]:
    """
    Retrofit IC→EV car: 36S Lead-Acid, 72V/130Ah, 10 kW BLDC.
    Source: IJAREEIE Vol. 8, Issue 4, April 2019.
    3-stage CC → CV(Topping) → Float charging.
    """
    return _simulate_charge_discharge(
        FOUR_WHEELER_RETRO,
        soc_init=0.30,
        charge_s=28800,             # paper: 8 hours full charge
        discharge_s=10800,          # paper: ~3 hours at normal load
        discharge_current=43.0,     # ~10kW / 72V ≈ 138A → 43A avg
        print_interval=3600,
        **kwargs,
    )


def simulate_electric_bus(**kwargs) -> List[BMSPackState]:
    """Electric bus: 128S2P LFP, 409.6V, 150 kW, 163.8 kWh."""
    return _simulate_charge_discharge(
        ELECTRIC_BUS,
        soc_init=0.20,
        charge_s=7200,
        discharge_s=3600,
        discharge_current=100.0,    # ~41 kW city cruising load (peak is 366 A)
        print_interval=600,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Degradation simulation
# ---------------------------------------------------------------------------

def _degradation_status(soh_q: float, warn_pct: float, crit_pct: float) -> str:
    """Label a capacity SOH [%] as OK / WARNING / CRITICAL against the pack thresholds."""
    if soh_q < crit_pct:
        return "CRITICAL"
    if soh_q < warn_pct:
        return "WARNING"
    return "OK"


def _degradation_point(cycle: int, q_nominal: float, r0_fresh_mohm: float,
                       deg_rate: float, r0_growth_rate: float):
    """Analytical fade at `cycle`: (soh_q %, usable Ah, soh_r %, aged R0 mΩ, dR0 %)."""
    # Capacity fade: linear (Wöhler-equivalent simplified model)
    soh_q = max(0.0, 100.0 * (1.0 - deg_rate * cycle))
    q_usable = q_nominal * soh_q / 100.0

    # Resistance growth: empirical power-law approximation
    r0_aged = r0_fresh_mohm * (1.0 + r0_growth_rate * cycle)
    soh_r = min(100.0, r0_fresh_mohm / r0_aged * 100.0)
    dr0_pct = (r0_aged - r0_fresh_mohm) / r0_fresh_mohm * 100.0
    return soh_q, q_usable, soh_r, r0_aged, dr0_pct


def simulate_degradation(
    profile: VehicleProfile,
    n_cycles: int = 300,
    deg_rate: float = 0.001,
    r0_growth_rate: float = 0.001,
    print_every: int = 25,
) -> List[dict]:
    """
    Model battery capacity fade and resistance growth over N charge/discharge cycles.

    Uses an analytical degradation model (not a per-second simulation) to quickly
    project SOH evolution across hundreds of cycles.

    Parameters
    ----------
    profile : VehicleProfile
        Vehicle/pack profile (determines nominal capacity and SOH thresholds).
    n_cycles : int
        Number of charge/discharge cycles to simulate.
    deg_rate : float
        Fraction of capacity lost per equivalent full cycle.
        Default 0.001 (0.1 %/cycle ≈ 10× accelerated vs real LFP for demo visibility).
    r0_growth_rate : float
        Fraction of R0 increase per cycle.
        Default 0.001 — R0 doubles by cycle ~1000.
    print_every : int
        Print a table row every N cycles.

    Returns
    -------
    List of dicts with cycle, soh_capacity_pct, capacity_ah, soh_resistance_pct,
    r0_mohm, status.
    """
    cell = profile.cell_config
    q_nominal = cell.nominal_capacity_ah * profile.n_cells_parallel
    r0_fresh_mohm = cell.r0 * 1000.0
    eol_pct = profile.pack_cfg.soh_warning_pct

    print(f"\n{'='*72}")
    print(f"  DEGRADATION SIMULATION — {profile.name}")
    print(f"  {n_cycles} cycles  |  deg_rate={deg_rate*100:.2f}%/cycle  "
          f"|  Q₀={q_nominal:.0f} Ah  |  R₀={r0_fresh_mohm:.2f} mΩ")
    print(f"  (Accelerated model: ~{round(0.02/deg_rate/100):,} real cycles = 1 simulated cycle "
          f"for LFP if actual loss is 0.002%/cycle)")
    print(f"{'='*72}")

    headers = ["Cycle", "SOH-Q%", "Cap(Ah)", "SOH-R%", "R0(mΩ)", "dR0%", "Status"]
    print(f"  {'|'.join(f'{h:>10}' for h in headers)}")
    print("  " + "-" * (11 * len(headers)))

    history: List[dict] = []

    warn_pct = profile.pack_cfg.soh_warning_pct
    crit_pct = profile.pack_cfg.soh_critical_pct

    for cycle in range(1, n_cycles + 1):
        soh_q, q_usable, soh_r, r0_aged, dr0_pct = _degradation_point(
            cycle, q_nominal, r0_fresh_mohm, deg_rate, r0_growth_rate)

        if cycle == 1 or cycle % print_every == 0 or cycle == n_cycles:
            status = _degradation_status(soh_q, warn_pct, crit_pct)

            cols = [
                f"{cycle:>10d}",
                f"{soh_q:>10.1f}",
                f"{q_usable:>10.1f}",
                f"{soh_r:>10.1f}",
                f"{r0_aged:>10.2f}",
                f"{dr0_pct:>10.1f}",
                f"{status:>10}",
            ]
            print("  " + "|".join(cols))
            history.append({
                "cycle": cycle,
                "soh_capacity_pct": round(soh_q, 2),
                "capacity_ah": round(q_usable, 2),
                "soh_resistance_pct": round(soh_r, 2),
                "r0_mohm": round(r0_aged, 3),
                "status": status,
            })

    # End-of-life projections
    eol_cycle = int((100.0 - eol_pct) / (deg_rate * 100.0)) if deg_rate > 0 else 99999
    print(f"\n  Capacity model  : SOH_Q = 100% − {deg_rate*100:.3f}% × N")
    print(f"  Resistance model: R0_N  = R0_fresh × (1 + {r0_growth_rate*100:.3f}% × N)")
    print(f"  EOL at SOH={eol_pct}% : ~{eol_cycle:,} cycles")
    print(f"  Final state (N={n_cycles}): SOH={max(0.0, 100-deg_rate*n_cycles*100):.1f}%  "
          f"R0={r0_fresh_mohm*(1+r0_growth_rate*n_cycles):.2f} mΩ\n")

    return history


# ---------------------------------------------------------------------------
# Cold-temperature simulation
# ---------------------------------------------------------------------------

def _run_cold_discharge(profile: VehicleProfile, t_amb: float, discharge_current: float,
                        max_discharge_s: int, soc_init: float) -> dict:
    """Discharge one pack at ambient `t_amb` until fault / SOC<=5% / timeout; return raw results."""
    bms = make_bms_controller(profile, soc_init=soc_init, ambient_temp_c=t_amb)
    t_peak = t_amb
    fault_str = "-"
    ah_out = 0.0
    soc_final = soc_init
    dis_blocked = False
    chg_blocked = False

    for tick in range(max_discharge_s):
        state = bms.step(
            current=discharge_current, dt=1.0,
            discharge_requested=True,
        )
        t_peak = max(t_peak, state.pack_temp_max_c)

        # Check power limits at first discharging step
        if tick == 7:  # after INIT+PRECHARGE settle
            if state.power_limits.i_max_discharge < 1.0:
                dis_blocked = True
            if state.power_limits.i_max_charge > -1.0:
                chg_blocked = True

        if state.fault_active:
            fault_str = state.fault_description
            soc_final = state.pack_soc_mean
            ah_out += discharge_current / 3600.0
            break

        if state.pack_soc_mean <= 0.05:
            soc_final = state.pack_soc_mean
            ah_out += discharge_current / 3600.0
            break

        soc_final = state.pack_soc_mean
        ah_out += discharge_current / 3600.0

    return {"t_peak": t_peak, "fault_str": fault_str, "ah_out": ah_out,
            "soc_final": soc_final, "dis_blocked": dis_blocked, "chg_blocked": chg_blocked}


def simulate_cold_temperature(
    profile: VehicleProfile,
    ambient_temps_c: Optional[List[float]] = None,
    discharge_current: Optional[float] = None,
    max_discharge_s: int = 1800,
    soc_init: float = 0.80,
) -> dict:
    """
    Compare discharge performance and fault behavior at different ambient temperatures.

    Shows three key cold-temperature effects:
    1. Reduced power limits (PowerLimitsCalculator blocks charge/discharge below t_min).
    2. Increased cell resistance at low temperatures (cell model thermal derating).
    3. BMS fault when temperature is below t_min_discharge (UNDER_TEMPERATURE).

    Parameters
    ----------
    profile : VehicleProfile
        Pack to test.
    ambient_temps_c : list[float]
        Ambient temperatures to sweep (°C).  Default: [25, 5, -10, -20, -30].
    discharge_current : float
        Pack discharge current [A].  Default: 30% of max discharge current.
    max_discharge_s : int
        Maximum simulation duration per temperature [s].
    soc_init : float
        Initial SOC for all runs.

    Returns
    -------
    dict  keyed by ambient temperature.
    """
    if ambient_temps_c is None:
        ambient_temps_c = [25.0, 5.0, -10.0, -20.0, -30.0]
    if discharge_current is None:
        discharge_current = profile.cell_config.i_max_discharge * profile.n_cells_parallel * 0.30

    cell = profile.cell_config
    t_min_dis = cell.t_min_discharge
    t_min_chg = cell.t_min_charge

    print(f"\n{'='*80}")
    print(f"  COLD TEMPERATURE SIMULATION — {profile.name}")
    print(f"  Chemistry: {profile.chemistry.symbol}  |  I_discharge={discharge_current:.0f} A  |  SOC_init={soc_init*100:.0f}%")
    print(f"  t_min_discharge={t_min_dis}°C  |  t_min_charge={t_min_chg}°C")
    print(f"{'='*80}")

    headers = ["T_amb(°C)", "T_start(°C)", "T_peak(°C)", "SOC_f%", "Ah_out",
               "DchgBlocked", "ChgBlocked", "Fault"]
    print(f"  {'|'.join(f'{h:>11}' for h in headers)}")
    print("  " + "-" * (12 * len(headers)))

    results = {}

    for t_amb in ambient_temps_c:
        run = _run_cold_discharge(profile, t_amb, discharge_current, max_discharge_s, soc_init)
        t_start = t_amb
        t_peak = run["t_peak"]
        fault_str = run["fault_str"]
        ah_out = run["ah_out"]
        soc_final = run["soc_final"]
        dis_blocked = run["dis_blocked"]
        chg_blocked = run["chg_blocked"]

        cols = [
            f"{t_amb:>11.1f}",
            f"{t_start:>11.1f}",
            f"{t_peak:>11.1f}",
            f"{soc_final*100:>11.1f}",
            f"{ah_out:>11.1f}",
            f"{'YES' if dis_blocked else 'no':>11}",
            f"{'YES' if chg_blocked else 'no':>11}",
            f"{fault_str:>11}",
        ]
        print("  " + "|".join(cols))

        results[t_amb] = {
            "t_amb_c": t_amb,
            "t_peak_c": round(t_peak, 1),
            "soc_final": round(soc_final, 3),
            "ah_out": round(ah_out, 1),
            "discharge_blocked": dis_blocked,
            "charge_blocked": chg_blocked,
            "fault": fault_str,
        }

    print(f"\n  Note: BMS faults at cell T < {t_min_dis}°C (discharge) / T < {t_min_chg}°C (charge).")
    print("  'DchgBlocked': PowerLimits sets i_max_discharge=0 at t_min_discharge threshold.")
    print("  'ChgBlocked' : PowerLimits sets i_max_charge=0 at t_min_charge threshold.")
    print(f"  Resistance increases ~0.5%/°C below t_nominal={cell.t_nominal}°C → more voltage sag.\n")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(histories: dict, output_dir: str = "reports"):
    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    colors = {"2W": "#2ecc71", "4W_RETRO": "#e74c3c", "BUS": "#3498db"}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("BMS Multi-Vehicle Simulation", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    plots = [
        (gs[0, 0], "pack_soc_mean", 100, "SOC (%)", "State of Charge"),
        (gs[0, 1], "pack_voltage_v", 1,  "Voltage (V)", "Pack Voltage"),
        (gs[0, 2], "pack_temp_max_c", 1, "Temp (°C)", "Max Cell Temperature"),
        (gs[1, 0], "pack_current_a", 1,  "Current (A)", "Pack Current"),
        (gs[1, 1], None, 1, "Fan (%)", "Cooling Fan Duty"),
        (gs[1, 2], None, 1, "SOH (%)", "Capacity SOH"),
    ]

    for spec, attr, scale, ylabel, title in plots:
        ax = fig.add_subplot(spec)
        for label, hist in histories.items():
            t = [s.timestamp_s for s in hist]
            if attr:
                y = [getattr(s, attr) * scale for s in hist]
            elif ylabel == "Fan (%)":
                y = [s.thermal.fan_pwm_pct for s in hist]
            else:
                y = [s.soh.soh_capacity_pct for s in hist]
            ax.plot(t, y, label=label, color=colors.get(label, "gray"), linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "bms_multi_vehicle.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

VEHICLE_MAP = {
    "2W": simulate_two_wheeler,
    "4W_RETRO": simulate_four_wheeler_retrofit,
    "BUS": simulate_electric_bus,
}


def main():
    parser = argparse.ArgumentParser(description="BMS multi-vehicle simulation")
    parser.add_argument("--vehicle", choices=list(VEHICLE_MAP.keys()),
                        help="Run only this vehicle class (default: all)")
    parser.add_argument("--plot", action="store_true", help="Save comparison plot")
    parser.add_argument("--degradation", action="store_true",
                        help="Run degradation simulation (capacity/resistance fade over 300 cycles)")
    parser.add_argument("--cold", action="store_true",
                        help="Run cold-temperature discharge sweep (+25 to −30°C)")
    parser.add_argument("--driving", action="store_true",
                        help="Run driving-conditions degradation projection (10-year SOH table)")
    parser.add_argument("--manufacturers", action="store_true",
                        help="Ather / Ola / Tesla / BYD pack specs and degradation comparison")
    parser.add_argument("--datasets", action="store_true",
                        help="Validate BMS model against 6 Kaggle battery datasets")
    args = parser.parse_args()

    print_comparison_table()

    if args.degradation:
        # Show degradation for 2-wheeler (fast, readable output) and bus (large pack)
        simulate_degradation(TWO_WHEELER, n_cycles=300, deg_rate=0.001)
        simulate_degradation(ELECTRIC_BUS, n_cycles=300, deg_rate=0.0005)
        return

    if args.cold:
        simulate_cold_temperature(TWO_WHEELER)
        simulate_cold_temperature(ELECTRIC_BUS)
        return

    if args.driving:
        simulate_driving_degradation(TWO_WHEELER, simulation_years=10)
        simulate_driving_degradation(ELECTRIC_BUS, simulation_years=10)
        return

    if args.manufacturers:
        simulate_manufacturer_comparison(simulation_years=8)
        return

    if args.datasets:
        from .datasets import (
            load_all_synthetic,
            print_validation_report,
            validate_bms_model,
        )
        print("\n  Generating synthetic Kaggle dataset equivalents...")
        datasets = load_all_synthetic()
        print(f"  Datasets ready: {list(datasets.keys())}")
        for name, df in datasets.items():
            print(f"    {name:<25} {len(df):>8,} rows  {list(df.columns)}")
        results = validate_bms_model(datasets)
        print_validation_report(results)
        return

    if args.vehicle:
        VEHICLE_MAP[args.vehicle]()
    else:
        histories = {}
        histories["2W"]       = simulate_two_wheeler()
        histories["4W_RETRO"] = simulate_four_wheeler_retrofit()
        histories["BUS"]      = simulate_electric_bus()

        if args.plot:
            _plot(histories)


if __name__ == "__main__":
    main()

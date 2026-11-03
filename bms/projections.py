"""
Lifecycle projections and comparison tables for the BMS simulation CLI.

Analytical (no per-second BMS stepping) SOH projections under real-world driving
patterns, the Ather / Ola / Tesla / BYD manufacturer comparison, and the
multi-vehicle capability summary table. Re-exported from :mod:`bms.simulation`.
"""

from __future__ import annotations

from .driving_profiles import (
    DRIVING_PROFILES,
    annual_soh_loss_pct,
    compute_stress_factors,
    efc_per_year,
    months_to_eol,
    soh_at_month,
)
from .manufacturer_profiles import (
    ATHER_450X,
    ATHER_DRIVING,
    BYD_DRIVING,
    BYD_HAN_EV,
    OLA_DRIVING,
    OLA_S1_PRO,
    TESLA_DRIVING,
    TESLA_MODEL3_SR,
)
from .vehicles import VehicleProfile


def _left_aligned_format(widths) -> str:
    """Return a `str.format` template of left-aligned columns with the given widths."""
    return "  " + "  ".join(f"{{:<{w}}}" for w in widths)


def _soh_marker(soh: float, warn_pct: float, crit_pct: float) -> str:
    """Table marker: '!' below critical SOH, '~' below warning SOH, else ''."""
    if soh < crit_pct:
        return "!"
    if soh < warn_pct:
        return "~"
    return ""


# ---------------------------------------------------------------------------
# Driving conditions degradation simulation
# ---------------------------------------------------------------------------

def simulate_driving_degradation(
    profile: VehicleProfile,
    conditions_list=None,
    base_rate_per_efc: float = 1e-4,
    calendar_aging_pct_year: float = 0.5,
    simulation_years: int = 10,
    print_every_years: int = 1,
) -> dict:
    """
    Project SOH evolution under different real-world driving patterns.

    Uses an analytical stress model (no per-second BMS steps) combining:
      - C-rate stress     : f_C  = mean(√C_discharge, √C_charge)
      - Temperature stress: f_T  = 1.5^((T-25)/10) above 25°C; +3%/°C below
      - DoD stress        : f_DoD = DoD²
      - Calendar ageing   : fixed term added regardless of cycling

    Parameters
    ----------
    profile : VehicleProfile
        Pack being evaluated (provides chemistry and SOH thresholds).
    conditions_list : list[DrivingConditions] | None
        Profiles to compare.  Default: all 6 built-in profiles.
    base_rate_per_efc : float
        Degradation fraction per EFC at reference (1C, 25°C, 100% DoD).
        Default 1e-4 (0.01%/EFC) — calibrated so city driving ≈ 1.4%/yr.
    calendar_aging_pct_year : float
        Time-dependent SOH loss [%/yr] regardless of cycling (SEI growth).
    simulation_years : int
        Duration to project [years].
    print_every_years : int
        Row frequency in the SOH-over-time table.

    Returns
    -------
    dict  keyed by profile name → list of (year, soh_pct) tuples.
    """
    if conditions_list is None:
        conditions_list = list(DRIVING_PROFILES.values())

    warn_pct = profile.pack_cfg.soh_warning_pct
    crit_pct = profile.pack_cfg.soh_critical_pct

    _print_driving_banner(profile, base_rate_per_efc, calendar_aging_pct_year, warn_pct)
    results = _print_driving_summary(
        conditions_list, warn_pct, base_rate_per_efc, calendar_aging_pct_year)
    _print_soh_over_time(
        conditions_list, results, simulation_years, print_every_years, warn_pct, crit_pct)
    print(f"\n  Legend: ~ = SOH below warning ({warn_pct}%)   ! = SOH below critical ({crit_pct}%)")
    _print_stress_model_note(calendar_aging_pct_year)

    return results


def _print_driving_banner(profile, base_rate_per_efc, calendar_aging_pct_year, warn_pct) -> None:
    print(f"\n{'='*86}")
    print(f"  DRIVING CONDITIONS DEGRADATION — {profile.name}")
    print(f"  Chemistry: {profile.chemistry.symbol}  |  "
          f"base_rate={base_rate_per_efc*100:.4f}%/EFC  |  "
          f"calendar_aging={calendar_aging_pct_year}%/yr  |  "
          f"EOL threshold: SOH={warn_pct}%")
    print(f"{'='*86}")


def _print_driving_summary(conditions_list, warn_pct, base_rate_per_efc, calendar_aging_pct_year) -> dict:
    """Print the per-profile stress table; return the initial per-profile results dict."""
    h = ["Profile", "C-dis", "T(°C)", "DoD%", "FC%", "StressX", "EFC/yr", "EOL(yr)"]
    w = [24, 7, 7, 6, 5, 9, 8, 9]
    fmt = _left_aligned_format(w)
    print(fmt.format(*h))
    print("  " + "-" * 84)

    results = {}
    for cond in conditions_list:
        f_c, f_T, f_dod, total = compute_stress_factors(cond)
        efcs = efc_per_year(cond)
        loss = annual_soh_loss_pct(base_rate_per_efc, cond, calendar_aging_pct_year)
        eol_m = months_to_eol(warn_pct, base_rate_per_efc, cond, calendar_aging_pct_year)
        eol_str = f"{eol_m/12:.1f}" if eol_m < 120 * 12 else "Never"
        print(fmt.format(
            cond.name[:24],
            f"{cond.avg_discharge_c_rate:.2f}",
            f"{cond.avg_pack_temp_c:.0f}",
            f"{cond.dod_per_trip*100:.0f}%",
            f"{cond.fast_charge_fraction*100:.0f}%",
            f"{total:.3f}×",
            f"{efcs:.0f}",
            eol_str,
        ))
        results[cond.name] = {"annual_loss_pct": loss, "eol_months": eol_m, "soh_by_year": []}
    return results


def _print_soh_over_time(conditions_list, results, simulation_years, print_every_years,
                         warn_pct, crit_pct) -> None:
    """Print the SOH-over-time table, appending (year, soh) points into ``results``."""
    print()
    months_steps = [m for m in range(0, simulation_years * 12 + 1,
                                     max(1, print_every_years * 12))]

    name_width = 6  # "Year"
    col_width = 10
    header_row = f"  {'Year':>{name_width}}" + "".join(
        f"  {c.name[:col_width]:>{col_width}}" for c in conditions_list
    )
    print(header_row)
    print("  " + "-" * (name_width + (col_width + 2) * len(conditions_list) + 2))

    for month in months_steps:
        year = month / 12
        row = f"  {year:>{name_width}.1f}"
        for cond in conditions_list:
            soh = max(0.0, 100.0 - results[cond.name]["annual_loss_pct"] * year)
            results[cond.name]["soh_by_year"].append((year, round(soh, 1)))
            marker = _soh_marker(soh, warn_pct, crit_pct)
            cell = f"{soh:.1f}{marker}"
            row += f"  {cell:>{col_width}}"
        print(row)


def _print_stress_model_note(calendar_aging_pct_year) -> None:
    print(f"""
  Stress model (per profile):
    f_C   = (√C_discharge + √C_charge_blend) / 2   [C-rate damage]
    f_T   = 1.5^((T−25)/10) if T≥25°C; 1+0.03×(25−T) if T<25°C  [Arrhenius]
    f_DoD = DoD²                                    [volume-change fatigue]
    eff_rate/EFC = base_rate × f_C × f_T × f_DoD
    annual_loss  = eff_rate × EFC/yr × 100%  +  {calendar_aging_pct_year}% calendar

  Interpretation:
    Aggressive driving degrades the pack faster on all three axes simultaneously:
    high C-rate + hot pack + deep DoD. The cold-climate profile shows that
    sub-zero temperatures add a lithium-plating penalty even at modest C-rates.
""")


# ---------------------------------------------------------------------------
# Manufacturer comparison
# ---------------------------------------------------------------------------

MANUFACTURER_VEHICLES = {
    "Ather 450X":  (ATHER_450X,      ATHER_DRIVING),
    "Ola S1 Pro":  (OLA_S1_PRO,       OLA_DRIVING),
    "Tesla M3 SR": (TESLA_MODEL3_SR,  TESLA_DRIVING),
    "BYD Han EV":  (BYD_HAN_EV,      BYD_DRIVING),
}

def simulate_manufacturer_comparison(
    base_rate_per_efc: float = 1e-4,
    calendar_aging_pct_year: float = 0.5,
    simulation_years: int = 8,
) -> dict:
    """
    Side-by-side BMS and degradation comparison for Ather, Ola, Tesla, and BYD.

    Uses manufacturer-specific driving conditions that reflect each vehicle's
    target market, typical usage pattern, and cooling system capability.

    Parameters
    ----------
    base_rate_per_efc : float
        Base degradation rate at reference conditions (1C, 25°C, 100% DoD).
    calendar_aging_pct_year : float
        Calendar SOH loss [%/yr].
    simulation_years : int
        Projection horizon [years].

    Returns
    -------
    dict keyed by short name → {profile, driving_cond, soh_by_year, eol_months}.
    """
    vehicles = MANUFACTURER_VEHICLES

    print(f"\n{'='*90}")
    print("  MANUFACTURER BMS COMPARISON — Ather · Ola Electric · Tesla · BYD")
    print(f"{'='*90}")

    _print_pack_specs(vehicles)
    results = _print_manufacturer_stress(
        vehicles, base_rate_per_efc, calendar_aging_pct_year)
    _print_manufacturer_soh(
        vehicles, results, base_rate_per_efc, calendar_aging_pct_year, simulation_years)

    print("\n  ~ = SOH below warning threshold   ! = below critical threshold")
    print("\n  Key BMS differentiators:")
    print("    Ather 450X  : Cloud-connected ADS; warp/eco modes; air-cooled (hot pack)")
    print("    Ola S1 Pro  : MoveOS 4; liquid cooling reduces pack temp vs Ather")
    print("    Tesla M3 SR : Octovalve thermal; Supercharger V3 (2.8C); trip planning")
    print("    BYD Han EV  : Cell-to-Body (CTB); nail-test safe; 120 kW DC; heat pump\n")

    return results


def _print_pack_specs(vehicles) -> None:
    headers = ["Vehicle", "Chemistry", "Pack", "Voltage", "Energy", "Motor", "Cooling", "Charge"]
    rows = []
    for name, (vp, _) in vehicles.items():
        rows.append([
            name,
            vp.chemistry.symbol,
            f"{vp.n_cells_series}S{vp.n_cells_parallel}P",
            f"{vp.nominal_pack_voltage_v:.0f} V",
            f"{vp.nominal_pack_energy_kwh:.1f} kWh",
            f"{vp.motor_power_kw:.0f} kW",
            "Active" if vp.has_active_cooling else "Passive",
            f"{vp.charging_stages}-stage",
        ])
    col_w = [14, 10, 10, 9, 10, 9, 9, 9]
    fmt = _left_aligned_format(col_w)
    print(fmt.format(*headers))
    print("  " + "-" * 88)
    for r in rows:
        print(fmt.format(*r))


def _print_manufacturer_stress(vehicles, base_rate_per_efc, calendar_aging_pct_year) -> dict:
    """Print the driving-conditions stress table; return the initial per-vehicle results."""
    print(f"\n  Typical driving conditions & degradation stress (base_rate={base_rate_per_efc*100:.4f}%/EFC):")
    h2 = ["Vehicle", "Driving scenario", "C-dis", "T_pack", "DoD%", "FC%", "Stress×", "EFC/yr", "EOL(yr)"]
    w2 = [14, 32, 7, 7, 6, 5, 9, 8, 9]
    fmt2 = _left_aligned_format(w2)
    print(fmt2.format(*h2))
    print("  " + "-" * 106)

    results = {}
    for name, (vp, dc) in vehicles.items():
        _, _, _, stress = compute_stress_factors(dc)
        efcs = dc.trips_per_day * 365.25 * dc.dod_per_trip
        loss = annual_soh_loss_pct(base_rate_per_efc, dc, calendar_aging_pct_year)
        eol_m = months_to_eol(vp.pack_cfg.soh_warning_pct, base_rate_per_efc, dc, calendar_aging_pct_year)
        eol_str = f"{eol_m/12:.1f}" if eol_m < 200 * 12 else "∞"
        print(fmt2.format(
            name, dc.description[:32],
            f"{dc.avg_discharge_c_rate:.2f}", f"{dc.avg_pack_temp_c:.0f}°C",
            f"{dc.dod_per_trip*100:.0f}%", f"{dc.fast_charge_fraction*100:.0f}%",
            f"{stress:.3f}×", f"{efcs:.0f}", eol_str,
        ))
        results[name] = {
            "profile": vp,
            "driving": dc,
            "annual_loss_pct": loss,
            "eol_months": eol_m,
            "soh_by_year": [],
        }
    return results


def _print_manufacturer_soh(vehicles, results, base_rate_per_efc, calendar_aging_pct_year,
                            simulation_years) -> None:
    """Print the projected-SOH table, appending (year, soh) points into ``results``."""
    print(f"\n  Projected SOH (%) over {simulation_years} years:")
    col = 13
    header_row = f"  {'Year':>6}" + "".join(f"  {n:>{col}}" for n in vehicles)
    print(header_row)
    print("  " + "-" * (8 + (col + 2) * len(vehicles)))

    for yr in range(simulation_years + 1):
        row = f"  {yr:>6}"
        for name, (vp, dc) in vehicles.items():
            soh = soh_at_month(yr * 12, base_rate_per_efc, dc, calendar_aging_pct_year)
            results[name]["soh_by_year"].append((yr, round(soh, 1)))
            warn = vp.pack_cfg.soh_warning_pct
            crit = vp.pack_cfg.soh_critical_pct
            marker = _soh_marker(soh, warn, crit)
            cell = f"{soh:.1f}{marker}"
            row += f"  {cell:>{col}}"
        print(row)


# ---------------------------------------------------------------------------
# Comparison summary table
# ---------------------------------------------------------------------------

def print_comparison_table():
    print("\n" + "=" * 90)
    print("  BMS MULTI-VEHICLE CAPABILITY COMPARISON")
    print("=" * 90)
    headers = ["Vehicle", "Chemistry", "Pack", "Voltage", "Energy", "Motor", "Charge"]
    rows = [
        ["2-Wheeler (Scooter)", "LFP",       "15S1P",  "48V",    "1.4 kWh",  "1.5 kW",  "CC-CV"],
        ["4W Retrofit (PbA)",   "Lead-Acid", "36S1P",  "72V",    "9.4 kWh",  "10 kW",   "CC-CV-Float"],
        ["4W Modern LFP",       "LFP",       "32S1P",  "102V",   "6.1 kWh",  "30 kW",   "CC-CV"],
        ["Electric Bus",        "LFP",       "128S2P", "409V",   "163.8 kWh","150 kW",  "CC-CV"],
    ]
    col_w = [22, 12, 8, 8, 11, 9, 14]
    fmt = _left_aligned_format(col_w)
    print(fmt.format(*headers))
    print("  " + "-" * 88)
    for r in rows:
        print(fmt.format(*r))
    print("=" * 90)

    print("""
  Key BMS algorithms (common across all vehicles):
    SOC estimation  : Extended Kalman Filter on 2-RC Thevenin model
    SOH estimation  : Capacity fade (Ah throughput) + R0 growth (dV/dI)
    Cell balancing  : Passive bleed-down with dead-band hysteresis
    Charging        : Stage 1 CC bulk → Stage 2 CV topping → (Stage 3 Float for PbA)
    Power limits    : SOC + temperature + SOH derating
    Thermal         : PWM fan controller; heater below T_min
    Protection      : OV / UV / OC / OT / isolation fault → contactor open
    Alerts          : Driver warnings (low SOC, high temp, cell imbalance)

  Supported chemistries (bms.chemistries):
    Established : LFP · Lead-Acid (PbA) · NMC
    Emerging    : Na-Ion · LMFP · LTO · Li-Sulfur · Solid-State NMC

  Lifecycle simulations (--degradation / --cold / --driving):
    Degradation : Capacity fade + R0 growth model over N cycles (Wöhler model)
    Cold temp   : Discharge performance and BMS fault sweep from +25°C to −30°C
    Driving     : SOH projection for 6 real-world driving patterns over 10 years
""")

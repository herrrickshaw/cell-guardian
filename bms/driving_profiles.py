"""
Driving condition profiles and battery degradation stress model.

Three physical mechanisms govern cycle-life under real-world driving:

  1. C-rate stress  (discharge + charge)
     Higher current densities increase electrode volume change and SEI growth
     per cycle. Empirical square-root relationship: f_C = C^0.5.

  2. Temperature stress  (Arrhenius-based)
     Above 25 °C: degradation rate scales with 1.5× per 10 °C (Q₁₀ = 1.5).
     Below 25 °C: lithium-plating risk during charging adds a cold penalty
     of +3 %/°C below the reference temperature.

  3. Depth-of-Discharge (DoD) stress
     Deeper cycles force larger electrode volume swings, cracking SEI and
     active material. Power-law: f_DoD = DoD².

  4. Calendar ageing (independent of cycling)
     SEI layer grows even at rest; ~0.5 %/year for LFP at 25 °C.

Effective degradation rate per equivalent full cycle (EFC):
    eff_rate = base_rate × f_C × f_T × f_DoD

Total annual SOH loss:
    ΔSOH/yr = eff_rate × EFC_per_year × 100  +  calendar_aging_pct

Reference conditions for base_rate (1.0×): 1C discharge/charge, 25 °C, 100% DoD.

Usage
-----
    from bms.driving_profiles import CITY, HIGHWAY, compute_stress_factors
    from bms.vehicles import TWO_WHEELER
    from bms.simulation import simulate_driving_degradation
    simulate_driving_degradation(TWO_WHEELER, [CITY, HIGHWAY])
"""

import math
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DrivingConditions:
    """
    Specification for one real-world driving pattern.

    Attributes
    ----------
    name : str
        Human-readable profile name.
    avg_discharge_c_rate : float
        Mean discharge C-rate during driving (power / nominal capacity).
    avg_pack_temp_c : float
        Mean pack temperature during a trip [°C].
    dod_per_trip : float
        Fraction of usable capacity consumed per trip [0–1].
    trips_per_day : int
        Charge/discharge cycles completed per day.
    fast_charge_fraction : float
        Fraction of charging events using DC fast charging [0–1].
    fast_charge_c_rate : float
        C-rate applied during DC fast charging.
    description : str
        Free-text summary of this driving scenario.
    """
    name: str
    avg_discharge_c_rate: float
    avg_pack_temp_c: float
    dod_per_trip: float
    trips_per_day: int
    fast_charge_fraction: float
    fast_charge_c_rate: float = 3.0
    description: str = ""


# ---------------------------------------------------------------------------
# Preset profiles
# ---------------------------------------------------------------------------

GENTLE = DrivingConditions(
    name="Gentle City Driver",
    avg_discharge_c_rate=0.30,
    avg_pack_temp_c=25.0,
    dod_per_trip=0.30,
    trips_per_day=2,
    fast_charge_fraction=0.00,
    fast_charge_c_rate=3.0,
    description=(
        "Short daily commutes (30% DoD); Level-2 AC charging only; "
        "mild climate (25°C). Minimal stress on every axis."
    ),
)

CITY = DrivingConditions(
    name="Average City Driver",
    avg_discharge_c_rate=0.50,
    avg_pack_temp_c=30.0,
    dod_per_trip=0.50,
    trips_per_day=2,
    fast_charge_fraction=0.20,
    fast_charge_c_rate=3.0,
    description=(
        "Mixed city driving (50% DoD); 20% of charges via DC fast (3C); "
        "warm summer temperatures (~30°C pack)."
    ),
)

HIGHWAY = DrivingConditions(
    name="Highway Commuter",
    avg_discharge_c_rate=1.00,
    avg_pack_temp_c=35.0,
    dod_per_trip=0.70,
    trips_per_day=1,
    fast_charge_fraction=0.50,
    fast_charge_c_rate=3.0,
    description=(
        "Long highway commutes (70% DoD / trip); half of charges via 3C DC; "
        "sustained speed keeps pack at 35°C."
    ),
)

AGGRESSIVE = DrivingConditions(
    name="Aggressive Driver",
    avg_discharge_c_rate=2.00,
    avg_pack_temp_c=40.0,
    dod_per_trip=0.80,
    trips_per_day=2,
    fast_charge_fraction=0.80,
    fast_charge_c_rate=3.0,
    description=(
        "Sport driving, hard acceleration (2C avg); 80% DoD per trip; "
        "mostly 3C DC fast charging; pack runs hot (40°C)."
    ),
)

COLD_CLIMATE = DrivingConditions(
    name="Cold Climate Driver",
    avg_discharge_c_rate=0.50,
    avg_pack_temp_c=5.0,
    dod_per_trip=0.60,
    trips_per_day=2,
    fast_charge_fraction=0.10,
    fast_charge_c_rate=3.0,
    description=(
        "Cold winters (pack at 5°C); cabin heater adds load; 60% DoD. "
        "Lithium-plating risk during cold charging dominates degradation."
    ),
)

REGEN_HEAVY = DrivingConditions(
    name="Stop-and-Go + Regen",
    avg_discharge_c_rate=0.40,
    avg_pack_temp_c=28.0,
    dod_per_trip=0.40,
    trips_per_day=3,
    fast_charge_fraction=0.05,
    fast_charge_c_rate=3.0,
    description=(
        "Dense city traffic with frequent regenerative braking (3 trips/day); "
        "shallow DoD (40%) but high cycle count. "
        "Regen pulses modelled as micro-cycles adding marginal stress."
    ),
)

DRIVING_PROFILES = {
    "gentle":     GENTLE,
    "city":       CITY,
    "highway":    HIGHWAY,
    "aggressive": AGGRESSIVE,
    "cold":       COLD_CLIMATE,
    "regen":      REGEN_HEAVY,
}


# ---------------------------------------------------------------------------
# Stress model
# ---------------------------------------------------------------------------

def compute_stress_factors(conditions: DrivingConditions) -> Tuple[float, float, float, float]:
    """
    Compute the three degradation stress multipliers for a DrivingConditions.

    Returns
    -------
    (f_c_rate, f_temp, f_dod, total_multiplier)
    """
    # Blended charge C-rate (mix of slow AC and DC fast charging)
    c_slow = 0.30          # typical Level-2 AC charge rate
    c_chg_avg = (
        (1.0 - conditions.fast_charge_fraction) * c_slow
        + conditions.fast_charge_fraction * conditions.fast_charge_c_rate
    )

    # C-rate stress: square-root of discharge × charge average (empirical)
    f_c = (math.sqrt(conditions.avg_discharge_c_rate) + math.sqrt(c_chg_avg)) / 2.0

    # Temperature stress
    T = conditions.avg_pack_temp_c
    if T >= 25.0:
        # Arrhenius-based: Q₁₀ = 1.5 (degradation rate 1.5× per 10°C)
        f_T = 1.5 ** ((T - 25.0) / 10.0)
    else:
        # Cold penalty: lithium plating during charging (+3% per °C below 25°C)
        f_T = 1.0 + 0.03 * (25.0 - T)

    # Depth-of-discharge stress (power-law exponent = 2)
    f_dod = conditions.dod_per_trip ** 2.0

    return f_c, f_T, f_dod, f_c * f_T * f_dod


def efc_per_year(conditions: DrivingConditions) -> float:
    """Equivalent full cycles accumulated per year for this driving pattern."""
    return conditions.trips_per_day * 365.25 * conditions.dod_per_trip


def effective_deg_rate_per_efc(
    base_rate_per_efc: float,
    conditions: DrivingConditions,
) -> float:
    """
    Effective capacity-fade rate per equivalent full cycle [fraction/EFC].

    Parameters
    ----------
    base_rate_per_efc : float
        Degradation rate at reference conditions (1C, 25°C, 100% DoD).
    conditions : DrivingConditions
        Driving pattern to evaluate.
    """
    _, _, _, multiplier = compute_stress_factors(conditions)
    return base_rate_per_efc * multiplier


def annual_soh_loss_pct(
    base_rate_per_efc: float,
    conditions: DrivingConditions,
    calendar_aging_pct_year: float = 0.5,
) -> float:
    """Total SOH loss per year (cycle aging + calendar aging) [%/year]."""
    cycle = effective_deg_rate_per_efc(base_rate_per_efc, conditions) * efc_per_year(conditions) * 100.0
    return cycle + calendar_aging_pct_year


def months_to_eol(
    soh_threshold_pct: float,
    base_rate_per_efc: float,
    conditions: DrivingConditions,
    calendar_aging_pct_year: float = 0.5,
) -> float:
    """
    Projected months until SOH falls below soh_threshold_pct.
    Returns float('inf') if the threshold is never reached.
    """
    loss_per_year = annual_soh_loss_pct(base_rate_per_efc, conditions, calendar_aging_pct_year)
    if loss_per_year <= 0.0:
        return float("inf")
    years = (100.0 - soh_threshold_pct) / loss_per_year
    return years * 12.0


def soh_at_month(
    month: int,
    base_rate_per_efc: float,
    conditions: DrivingConditions,
    calendar_aging_pct_year: float = 0.5,
) -> float:
    """SOH [%] after `month` months of driving under `conditions`."""
    loss_per_month = annual_soh_loss_pct(base_rate_per_efc, conditions, calendar_aging_pct_year) / 12.0
    return max(0.0, 100.0 - loss_per_month * month)

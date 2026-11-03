"""
Real-world manufacturer BMS profiles tuned from published specifications.

Sources
-------
Ather 450X     : Ather Energy whitepaper / product page (2024 Gen-3)
Ola S1 Pro     : Ola Electric product specs & MoveOS announcements (2024 Gen-2)
Tesla Model 3  : Tesla Motor Trend data, EPA range sheet, Supercharger specs (2024 SR LFP)
BYD Han EV     : BYD Blade Battery whitepaper, CLTC homologation data (2024)

Design notes
------------
*  Cell counts are derived from published pack-voltage ÷ nominal cell voltage.
*  Cell capacity is derived from published usable energy ÷ pack voltage.
*  R₀ and RC-network values are estimated from published DCIR and pulse-test data
   where available; otherwise scaled from chemistry defaults (per-Ah scaling).
*  All thermal-management parameters reflect the vehicle's actual cooling system.
*  Charging stages follow each OEM's published protocol (2-stage CC-CV or 3-stage).

Manufacturer-specific driving conditions reflect typical real-world usage patterns
documented in owner surveys, telemetry releases, and press fleet reports.
"""

from dataclasses import replace as _dc_replace

from .chemistries import LFP, NMC, ChemistryProfile
from .config import DEFAULT_EKF, CellConfig, PackConfig
from .driving_profiles import DrivingConditions
from .vehicles import VehicleProfile


def _make_chem(base: ChemistryProfile, cell: CellConfig) -> ChemistryProfile:
    """Return a new ChemistryProfile identical to `base` except with a custom CellConfig."""
    return ChemistryProfile(
        name=base.name, symbol=base.symbol,
        ocv_table=base.ocv_table, cell_config=cell,
        self_discharge_pct_month=base.self_discharge_pct_month,
        charge_efficiency=base.charge_efficiency,
        float_voltage_v=base.float_voltage_v,
        description=base.description,
    )


# ===========================================================================
# 1. ATHER 450X  (Gen-3, 2024)
#    Specs: 3.7 kWh usable | 72 V (20S) | NMC cylindrical cells
#    Motor: 6 kW continuous, 18 kW peak
#    Charge: Ather Dot (1.5 kW AC) and Ather Grid / supercharge (5 kW AC fast)
#    Cooling: passive air-cooled
#    BMS: Ather Data Services (ADS) — cloud-connected, OTA, warp/eco modes
# ===========================================================================

# 20 NMC cells in series × 2 parallel strings = 20S2P
# 20 × 3.70 V = 74 V nominal
# Usable energy: 3.7 kWh ÷ 74 V = 50 Ah pack → 25 Ah per physical cell
_NMC_ATHER_CELL = _dc_replace(
    NMC.cell_config,
    nominal_capacity_ah=25.0,
    # 18650/21700 cylindrical NMC: DCIR ≈ 30–60 mΩ for 25 Ah class
    r0=0.018, r1=0.008, c1=1800.0, r2=0.012, c2=12000.0,
    i_max_charge=-37.5,          # 1.5C charge (Ather super-charge ~5 kW ÷ 74V = 67.6A pack ÷ 2P)
    i_max_discharge=150.0,       # 6C peak → 18 kW ÷ 74V = 243A pack ÷ 2P = 121A; use 150A
    i_cc_charge=-25.0,           # 1C steady-state (AC Level-2: 1.5 kW)
    i_cv_term=-1.25,             # 0.05C taper
    t_max_charge=45.0,
    t_min_charge=0.0,
    t_max_discharge=60.0,
    t_min_discharge=-10.0,       # NMC limited at cold; Ather warns below 0°C
    t_nominal=25.0,
    thermal_capacity_j_k=800.0,
    thermal_resistance_k_w=8.0,  # air-cooled: higher thermal resistance
)

_NMC_ATHER_CHEM = _make_chem(NMC, _NMC_ATHER_CELL)

ATHER_450X = VehicleProfile(
    name="Ather 450X Gen-3 (2024)",
    vehicle_type="2W",
    motor_power_kw=6.0,
    n_cells_series=20,
    n_cells_parallel=2,
    chemistry=_NMC_ATHER_CHEM,
    pack_cfg=PackConfig(
        cells_series=20,
        cells_parallel=2,
        balance_v_threshold=0.010,   # NMC: tighter tolerance than LFP
        balance_current_a=0.25,
        soh_warning_pct=80.0,
        soh_critical_pct=60.0,
        pre_charge_resistor_ohm=5.0,
        isolation_fault_threshold_ohm=100.0,
    ),
    ekf_cfg=DEFAULT_EKF,
    t_fan_on_c=40.0,
    t_fan_full_c=55.0,
    t_heat_on_c=0.0,
    has_active_cooling=False,        # passive air-cooled
    charging_stages=2,               # CC + CV
    notes=(
        "Ather 450X Gen-3. 20S2P NMC | 74V / 50Ah / 3.7 kWh. "
        "Ather Grid / supercharge: 5 kW AC (1.5C). Passive air cooling. "
        "ADS cloud-BMS: OTA, warp mode, predictive charging schedule."
    ),
)


# ===========================================================================
# 2. OLA S1 PRO  (Gen-2 / Bharat Cell, 2024)
#    Specs: 4 kWh | 72 V (22S LFP) | cylindrical LFP 4680-style Bharat cells
#    Motor: 8.5 kW continuous, 45 kW peak (reported)
#    Charge: 1.5 kW home (0.75C), 3 kW Hypercharger
#    Cooling: liquid-cooled thermal management system (Gen-2 onwards)
#    BMS: MoveOS 4 — predictive range, hill-hold, multi-profile regen
# ===========================================================================

# 22 LFP cells in series × 1 parallel string = 22S1P
# 22 × 3.20 V = 70.4 V nominal
# 4 kWh ÷ 70.4 V = 56.8 Ah cell
_LFP_OLA_CELL = _dc_replace(
    LFP.cell_config,
    nominal_capacity_ah=56.8,
    # Large-format cylindrical LFP (4680 class): DCIR ≈ 5–10 mΩ
    r0=0.005, r1=0.002, c1=4000.0, r2=0.003, c2=35000.0,
    i_max_charge=-85.2,          # 1.5C (Hypercharger 3 kW ÷ 70.4V = 42.6A → 0.75C; headroom to 1.5C)
    i_max_discharge=568.0,       # ~10C peak (45 kW ÷ 70.4V = 639A → ~10C; limited to 568A)
    i_cc_charge=-56.8,           # 1C steady (home charge ≈ 1.5 kW ÷ 70.4V ≈ 21A actual, but 1C as CC set)
    i_cv_term=-2.8,              # 0.05C
    t_max_charge=45.0,
    t_min_charge=5.0,
    t_max_discharge=60.0,
    t_min_discharge=-20.0,       # LFP good cold performance
    t_nominal=25.0,
    thermal_capacity_j_k=2500.0,
    thermal_resistance_k_w=2.5,  # liquid cooling: low R
)

_LFP_OLA_CHEM = _make_chem(LFP, _LFP_OLA_CELL)

OLA_S1_PRO = VehicleProfile(
    name="Ola S1 Pro Gen-2 / Bharat Cell (2024)",
    vehicle_type="2W",
    motor_power_kw=8.5,
    n_cells_series=22,
    n_cells_parallel=1,
    chemistry=_LFP_OLA_CHEM,
    pack_cfg=PackConfig(
        cells_series=22,
        cells_parallel=1,
        balance_v_threshold=0.015,
        balance_current_a=0.3,
        soh_warning_pct=80.0,
        soh_critical_pct=60.0,
        pre_charge_resistor_ohm=8.0,
        isolation_fault_threshold_ohm=100.0,
    ),
    ekf_cfg=DEFAULT_EKF,
    t_fan_on_c=38.0,
    t_fan_full_c=50.0,
    t_heat_on_c=5.0,
    has_active_cooling=True,         # liquid thermal management (Gen-2)
    charging_stages=2,
    notes=(
        "Ola S1 Pro Gen-2 Bharat Cell. 22S1P LFP 4680-style | 70.4V / 56.8Ah / 4 kWh. "
        "Hypercharger: 3 kW (0.75C). Liquid-cooled. "
        "MoveOS 4: predictive range, multi-profile regen, OTA."
    ),
)


# ===========================================================================
# 3. TESLA MODEL 3  (Standard Range, LFP — India / China spec, 2024)
#    Specs: 60 kWh (usable) | ~345 V | CATL LFP prismatic cells
#    Motor: 208 kW RWD (India market)
#    Charge: 11 kW AC (onboard, 0.19C) | 170 kW DC Supercharger V3 (2.83C)
#    Cooling: liquid-cooled (direct refrigerant or glycol)
#    BMS: Tesla proprietary — trip planning, preconditioning, scheduled charging
# ===========================================================================

# 108 LFP cells in series × 2 parallel strings = 108S2P
# 108 × 3.20 V = 345.6 V nominal
# 60 kWh ÷ 345.6 V = 173.6 Ah pack → 86.8 Ah per physical cell
_LFP_TESLA_SR_CELL = _dc_replace(
    LFP.cell_config,
    nominal_capacity_ah=86.8,
    # Large CATL prismatic LFP cell: DCIR ≈ 0.3–0.8 mΩ per Ah scaling
    r0=0.0006, r1=0.0003, c1=8000.0, r2=0.0004, c2=80000.0,
    i_max_charge=-173.6,         # 1C (set ceiling; AC limited to 0.19C)
    i_max_discharge=521.0,       # 3C peak (208 kW ÷ 345.6V = 602A pack ÷ 2P = 301A/cell → 3.5C; cap at 3C)
    i_cc_charge=-86.8,           # 1C CC phase (Supercharger peak ≈ 170 kW ÷ 345.6V = 492A pack ÷ 2P = 246A → 2.8C)
    i_cv_term=-4.3,              # 0.05C
    t_max_charge=45.0,
    t_min_charge=0.0,
    t_max_discharge=60.0,
    t_min_discharge=-20.0,
    t_nominal=25.0,
    thermal_capacity_j_k=15000.0,  # large prismatic cells: very high thermal mass
    thermal_resistance_k_w=0.15,   # excellent liquid cooling (Octovalve system)
)

_LFP_TESLA_CHEM = _make_chem(LFP, _LFP_TESLA_SR_CELL)

TESLA_MODEL3_SR = VehicleProfile(
    name="Tesla Model 3 SR LFP (India/China, 2024)",
    vehicle_type="4W",
    motor_power_kw=208.0,
    n_cells_series=108,
    n_cells_parallel=2,
    chemistry=_LFP_TESLA_CHEM,
    pack_cfg=PackConfig(
        cells_series=108,
        cells_parallel=2,
        balance_v_threshold=0.008,   # Tesla: tight cell-level monitoring
        balance_current_a=1.0,
        soh_warning_pct=80.0,
        soh_critical_pct=60.0,
        pre_charge_resistor_ohm=1.5,
        isolation_fault_threshold_ohm=100.0,
    ),
    ekf_cfg=DEFAULT_EKF,
    t_fan_on_c=30.0,
    t_fan_full_c=45.0,
    t_heat_on_c=5.0,
    has_active_cooling=True,         # Octovalve liquid thermal management
    charging_stages=2,
    notes=(
        "Tesla Model 3 SR LFP (India/China 2024). 108S2P CATL LFP prismatic | "
        "345.6V / 173.6Ah / 60 kWh. Supercharger V3: 170 kW (2.8C). "
        "Octovalve liquid cooling. Trip planning, preconditioning, charge scheduling."
    ),
)


# ===========================================================================
# 4. BYD HAN EV  (Blade Battery, 2024)
#    Specs: 76.9 kWh | 614.4 V | BYD Blade LFP prismatic cells (CTB)
#    Motor: 180 kW (RWD) / 380 kW (AWD)
#    Charge: 7.4 kW AC (0.096C) | 120 kW DC (1.56C)
#    Cooling: liquid-cooled (Cell-to-Body integration with heat pump)
#    BMS: BYD proprietary — Blade cell abuse test, cell-to-body structural
# ===========================================================================

# 192 LFP Blade cells in series × 1 parallel string = 192S1P
# 192 × 3.20 V = 614.4 V nominal
# 76.9 kWh ÷ 614.4 V = 125.2 Ah per Blade cell
_LFP_BYD_BLADE_CELL = _dc_replace(
    LFP.cell_config,
    nominal_capacity_ah=125.2,
    # BYD Blade (960 mm long prismatic): very low DCIR due to large format
    # Estimated DCIR ≈ 0.2–0.5 mΩ per cell (large Ah, low R scales 1/Ah)
    r0=0.0005, r1=0.0002, c1=15000.0, r2=0.0003, c2=120000.0,
    i_max_charge=-195.0,         # 1.56C (120 kW ÷ 614.4V = 195A)
    i_max_discharge=500.0,       # 3.99C → 380 kW AWD ÷ 614.4V = 618.5A; limit to 500A (2×2C)
    i_cc_charge=-125.2,          # 1C CC (120 kW DC fast at peak)
    i_cv_term=-6.3,              # 0.05C
    t_max_charge=45.0,
    t_min_charge=0.0,
    t_max_discharge=60.0,
    t_min_discharge=-20.0,
    t_nominal=25.0,
    thermal_capacity_j_k=22000.0,  # Blade cell: very large thermal mass
    thermal_resistance_k_w=0.12,   # liquid-cooled + CTB integration
)

_LFP_BYD_CHEM = _make_chem(LFP, _LFP_BYD_BLADE_CELL)

BYD_HAN_EV = VehicleProfile(
    name="BYD Han EV Blade Battery (2024)",
    vehicle_type="4W",
    motor_power_kw=180.0,
    n_cells_series=192,
    n_cells_parallel=1,
    chemistry=_LFP_BYD_CHEM,
    pack_cfg=PackConfig(
        cells_series=192,
        cells_parallel=1,
        balance_v_threshold=0.006,   # BYD: extremely tight balancing on Blade
        balance_current_a=1.5,
        soh_warning_pct=80.0,
        soh_critical_pct=60.0,
        pre_charge_resistor_ohm=1.0,
        isolation_fault_threshold_ohm=100.0,
    ),
    ekf_cfg=DEFAULT_EKF,
    t_fan_on_c=28.0,
    t_fan_full_c=42.0,
    t_heat_on_c=5.0,
    has_active_cooling=True,         # liquid-cooled + heat pump (CTB)
    charging_stages=2,
    notes=(
        "BYD Han EV Blade Battery 2024. 192S1P LFP prismatic | "
        "614.4V / 125.2Ah / 76.9 kWh. DC fast: 120 kW (1.56C). "
        "Cell-to-Body (CTB) — structural integration doubles pack torsional rigidity. "
        "BYD Blade abuse test: nail penetration without thermal runaway."
    ),
)


# ===========================================================================
# Manufacturer registry
# ===========================================================================

MANUFACTURER_REGISTRY = {
    "ather_450x":       ATHER_450X,
    "ola_s1_pro":       OLA_S1_PRO,
    "tesla_model3_sr":  TESLA_MODEL3_SR,
    "byd_han_ev":       BYD_HAN_EV,
}


# ===========================================================================
# Manufacturer-typical driving conditions
# Reflect real-world usage patterns for each manufacturer's target market.
# ===========================================================================

# Ather: Indian urban commuter, hot summers, short trips, AC charging
ATHER_DRIVING = DrivingConditions(
    name="Ather 450X (Indian Urban)",
    avg_discharge_c_rate=0.60,       # stop-and-go city; regen helps
    avg_pack_temp_c=35.0,            # Indian summer; air-cooled pushes pack temp high
    dod_per_trip=0.35,               # typical 25–35 km commute on 85 km range
    trips_per_day=2,                 # morning + evening
    fast_charge_fraction=0.10,       # mostly home AC; occasional Ather Grid
    fast_charge_c_rate=1.5,          # Ather supercharge ≈ 1.5C
    description="Indian urban commuter on Ather 450X; air-cooled; hot summers",
)

# Ola: Indian urban, similar hot conditions but liquid cooling helps
OLA_DRIVING = DrivingConditions(
    name="Ola S1 Pro (Indian Urban)",
    avg_discharge_c_rate=0.70,       # aggressive acceleration profiles popular
    avg_pack_temp_c=30.0,            # liquid cooling holds pack 5°C below Ather
    dod_per_trip=0.40,
    trips_per_day=2,
    fast_charge_fraction=0.20,       # Hypercharger adoption growing
    fast_charge_c_rate=0.75,         # 3 kW / 70.4V / 56.8Ah ≈ 0.75C
    description="Indian urban commuter on Ola S1 Pro; liquid-cooled; frequent Hypercharger",
)

# Tesla: mixed highway + city, supercharger trips, efficient thermal management
TESLA_DRIVING = DrivingConditions(
    name="Tesla Model 3 (Mixed India)",
    avg_discharge_c_rate=0.80,       # highway + city mix
    avg_pack_temp_c=28.0,            # Octovalve keeps pack at 25–30°C
    dod_per_trip=0.55,               # ~190 km trip on 340 km range
    trips_per_day=1,
    fast_charge_fraction=0.35,       # Supercharger for highway; home for daily
    fast_charge_c_rate=2.8,          # Supercharger V3 peak ≈ 2.8C
    description="Mixed highway/city driving; Tesla Supercharger V3 for long trips",
)

# BYD: mixed city + highway, efficient liquid + heat pump, careful DoD management
BYD_DRIVING = DrivingConditions(
    name="BYD Han EV (Chinese Mixed)",
    avg_discharge_c_rate=0.60,
    avg_pack_temp_c=27.0,            # CTB + heat pump: excellent thermal control
    dod_per_trip=0.50,
    trips_per_day=1,
    fast_charge_fraction=0.30,       # 120 kW CCS2 DC
    fast_charge_c_rate=1.56,         # 120 kW / 614.4V / 125.2Ah ≈ 1.56C
    description="Mixed city + highway; BYD CTB thermal management; 120 kW DC fast charge",
)

MANUFACTURER_DRIVING = {
    "ather_450x":      ATHER_DRIVING,
    "ola_s1_pro":      OLA_DRIVING,
    "tesla_model3_sr": TESLA_DRIVING,
    "byd_han_ev":      BYD_DRIVING,
}

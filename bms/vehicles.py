"""
Vehicle-specific BMS profiles.

Each profile bundles the pack topology, chemistry, thermal strategy, and
charging parameters required by BMSController for a specific vehicle class.

Vehicle types
-------------
TWO_WHEELER         Electric scooter / e-bike (48V LFP, ~1.5 kW)
FOUR_WHEELER_RETRO  Retrofit IC→EV car (72V Lead-Acid, 10 kW)  ← IJAREEIE paper
FOUR_WHEELER_MODERN Modern 4-wheeler (102.4V LFP, 30 kW)
ELECTRIC_BUS        City / intercity e-bus (409.6V LFP, 150 kW)

Usage
-----
    from bms.vehicles import FOUR_WHEELER_RETRO, make_bms_controller
    bms = make_bms_controller(FOUR_WHEELER_RETRO, soc_init=0.50)
    state = bms.step(current=50.0, dt=1.0)
"""

from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace

from .chemistries import LEAD_ACID, LFP, ChemistryProfile
from .chemistries import ChemistryProfile as _CP
from .config import DEFAULT_EKF, CellConfig, EKFConfig, PackConfig


@dataclass
class VehicleProfile:
    """
    Complete specification for one vehicle class.

    Attributes
    ----------
    name : str
        Human-readable vehicle name.
    vehicle_type : str
        '2W' | '4W' | 'BUS'
    motor_power_kw : float
        Rated traction motor power [kW].
    n_cells_series : int
        Cells in series per string.
    n_cells_parallel : int
        Parallel cell strings.
    chemistry : ChemistryProfile
        Electrochemical profile (OCV table, limits, …).
    pack_cfg : PackConfig
        Pack-level topology and protection.
    ekf_cfg : EKFConfig
        Kalman filter noise tuning.
    t_fan_on_c : float
        Temperature threshold to start cooling fan [°C].
    t_fan_full_c : float
        Temperature for full fan speed [°C].
    t_heat_on_c : float
        Temperature below which heater activates [°C].
    has_active_cooling : bool
        True = liquid / forced-air; False = passive / natural convection.
    charging_stages : int
        2 = CC+CV (LFP / NMC); 3 = CC+CV+Float (Lead-Acid).
    notes : str
        Application notes / source reference.
    """
    name: str
    vehicle_type: str
    motor_power_kw: float
    n_cells_series: int
    n_cells_parallel: int
    chemistry: ChemistryProfile
    pack_cfg: PackConfig
    ekf_cfg: EKFConfig = field(default_factory=lambda: DEFAULT_EKF)
    t_fan_on_c: float = 35.0
    t_fan_full_c: float = 50.0
    t_heat_on_c: float = 5.0
    has_active_cooling: bool = False
    charging_stages: int = 2
    notes: str = ""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def cell_config(self) -> CellConfig:
        return self.chemistry.cell_config

    @property
    def nominal_pack_voltage_v(self) -> float:
        return self.n_cells_series * self.cell_config.v_nominal

    @property
    def nominal_pack_capacity_ah(self) -> float:
        return self.cell_config.nominal_capacity_ah * self.n_cells_parallel

    @property
    def nominal_pack_energy_kwh(self) -> float:
        return (self.nominal_pack_voltage_v * self.nominal_pack_capacity_ah) / 1000.0

    @property
    def max_discharge_current_a(self) -> float:
        return self.cell_config.i_max_discharge * self.n_cells_parallel

    @property
    def max_charge_current_a(self) -> float:
        return self.cell_config.i_max_charge * self.n_cells_parallel

    def summary(self) -> str:
        lines = [
            f"{'='*58}",
            f"  {self.name}  [{self.vehicle_type}]",
            f"{'='*58}",
            f"  Chemistry   : {self.chemistry.symbol} — {self.chemistry.name}",
            f"  Pack        : {self.n_cells_series}S{self.n_cells_parallel}P",
            f"  Voltage     : {self.nominal_pack_voltage_v:.1f} V (nominal)",
            f"  Capacity    : {self.nominal_pack_capacity_ah:.0f} Ah",
            f"  Energy      : {self.nominal_pack_energy_kwh:.1f} kWh",
            f"  Motor       : {self.motor_power_kw:.0f} kW",
            f"  Discharge   : {self.max_discharge_current_a:.0f} A max",
            f"  Charge      : {self.max_charge_current_a:.0f} A max",
            f"  Charging    : {self.charging_stages}-stage",
            f"  Cooling     : {'Active' if self.has_active_cooling else 'Passive/Fan'}",
            f"  Fan on at   : {self.t_fan_on_c}°C",
            f"  {self.notes}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preset: 2-Wheeler — Electric Scooter / E-Bike (48V LFP)
# Typical: Ola S1, Ather 450, Hero Photon
# ---------------------------------------------------------------------------
TWO_WHEELER = VehicleProfile(
    name="Electric Scooter / E-Bike (48V LFP)",
    vehicle_type="2W",
    motor_power_kw=1.5,
    n_cells_series=15,              # 15 × 3.2 V = 48 V
    n_cells_parallel=1,
    chemistry=LFP,
    pack_cfg=PackConfig(
        cells_series=15,
        cells_parallel=1,
        balance_v_threshold=0.015,  # slightly looser — small pack
        balance_current_a=0.2,
        soh_warning_pct=80.0,
        soh_critical_pct=60.0,
        pre_charge_resistor_ohm=10.0,
        isolation_fault_threshold_ohm=100.0,
    ),
    t_fan_on_c=40.0,
    t_fan_full_c=55.0,
    t_heat_on_c=0.0,
    has_active_cooling=False,
    charging_stages=2,
    notes=(
        "Simple air-cooled pack. CC-CV charge. "
        "Typical Indian e-scooter: ~1.5–2.5 kWh, 60–80 km range."
    ),
)

# ---------------------------------------------------------------------------
# Preset: 4-Wheeler Retrofit — Lead-Acid Retrofit EV (72V PbA)
# Source: IJAREEIE Vol.8, Issue 4, 2019
#   6 × 12V Lead-Acid modules = 72V pack
#   Each 12V module = 6 cells × 2V → 36 cells total
#   130Ah, 10kW BLDC, 3-stage charging, fan at 40°C
# ---------------------------------------------------------------------------

# Override lead-acid cell config with paper-specific current limits
_PBA_4W_CELL = _dc_replace(
    LEAD_ACID.cell_config,
    nominal_capacity_ah=130.0,
    i_max_charge=-10.0,         # charger in paper: up to 10 A
    i_max_discharge=130.0,      # 1 C
    i_cc_charge=-10.0,
    i_cv_term=-1.0,
)

_LEAD_ACID_4W = _CP(
    name=LEAD_ACID.name,
    symbol=LEAD_ACID.symbol,
    ocv_table=LEAD_ACID.ocv_table,
    cell_config=_PBA_4W_CELL,
    self_discharge_pct_month=LEAD_ACID.self_discharge_pct_month,
    charge_efficiency=LEAD_ACID.charge_efficiency,
    float_voltage_v=LEAD_ACID.float_voltage_v,
    description=LEAD_ACID.description,
)

FOUR_WHEELER_RETRO = VehicleProfile(
    name="Retrofit EV Car — Lead-Acid (72V / 130Ah)",
    vehicle_type="4W",
    motor_power_kw=10.0,
    n_cells_series=36,              # 6 modules × 6 cells × 2V = 72V
    n_cells_parallel=1,
    chemistry=_LEAD_ACID_4W,
    pack_cfg=PackConfig(
        cells_series=36,
        cells_parallel=1,
        balance_v_threshold=0.050,  # lead-acid tolerates larger spread
        balance_current_a=0.5,
        soh_warning_pct=70.0,       # lead-acid degrades faster
        soh_critical_pct=50.0,
        pre_charge_resistor_ohm=10.0,
        isolation_fault_threshold_ohm=100.0,
    ),
    t_fan_on_c=35.0,                # paper: temp above 40°C activates cooling
    t_fan_full_c=45.0,
    t_heat_on_c=10.0,               # paper: min operating temp 10°C
    has_active_cooling=False,        # paper: external cooling fan (passive assist)
    charging_stages=3,              # CC → Topping (CV) → Float
    notes=(
        "Source: IJAREEIE 2019 — battery pack + BMS for reliable/safe EV. "
        "72V 130Ah lead-acid powers 10kW BLDC. 3-stage CCCVF charging."
    ),
)

# ---------------------------------------------------------------------------
# Preset: 4-Wheeler Modern — LFP 102.4V (Compact EV)
# Typical: Tata Tiago EV, MG Comet, small city car
# ---------------------------------------------------------------------------
_LFP_4W_CELL = _dc_replace(
    LFP.cell_config,
    nominal_capacity_ah=60.0,
    i_max_charge=-60.0,         # 1C
    i_max_discharge=240.0,      # 4C
    i_cc_charge=-60.0,
    i_cv_term=-3.0,
)

_LFP_4W_CHEM = _CP(
    name=LFP.name, symbol=LFP.symbol,
    ocv_table=LFP.ocv_table,
    cell_config=_LFP_4W_CELL,
    self_discharge_pct_month=LFP.self_discharge_pct_month,
    charge_efficiency=LFP.charge_efficiency,
    float_voltage_v=LFP.float_voltage_v,
    description=LFP.description,
)

FOUR_WHEELER_MODERN = VehicleProfile(
    name="Modern Compact EV (102.4V LFP)",
    vehicle_type="4W",
    motor_power_kw=30.0,
    n_cells_series=32,              # 32 × 3.2 V = 102.4 V
    n_cells_parallel=1,
    chemistry=_LFP_4W_CHEM,
    pack_cfg=PackConfig(
        cells_series=32,
        cells_parallel=1,
        balance_v_threshold=0.010,
        balance_current_a=0.5,
        soh_warning_pct=80.0,
        soh_critical_pct=60.0,
        pre_charge_resistor_ohm=5.0,
        isolation_fault_threshold_ohm=100.0,
    ),
    t_fan_on_c=38.0,
    t_fan_full_c=50.0,
    t_heat_on_c=5.0,
    has_active_cooling=True,
    charging_stages=2,
    notes="Modern compact city EV. LFP for safety. Active air/liquid cooling.",
)

# ---------------------------------------------------------------------------
# Preset: Electric Bus — 409.6V LFP (City / Intercity)
# Typical: Olectra C9, Tata Starbus EV, BEST/DTC fleet buses
# ---------------------------------------------------------------------------
_LFP_BUS_CELL = _dc_replace(
    LFP.cell_config,
    nominal_capacity_ah=200.0,   # 2P combined: 2 × 100 Ah
    # 2P parallel combination: R_eff = R_cell / 2, and large-format cells
    # already have lower R than the small-cell default.
    r0=0.0005, r1=0.0003, c1=3000.0,
    r2=0.0005, c2=100000.0,
    i_max_charge=-100.0,         # 0.5C per physical cell (100 Ah)
    i_max_discharge=400.0,       # 2C per physical cell
    i_cc_charge=-100.0,
    i_cv_term=-10.0,
    t_max_discharge=55.0,        # buses run harder
    t_min_discharge=-20.0,
    t_max_charge=40.0,
    thermal_capacity_j_k=8000.0,
    thermal_resistance_k_w=0.3,  # liquid cooling: low R keeps cell below 55°C
)

_LFP_BUS_CHEM = _CP(
    name=LFP.name, symbol=LFP.symbol,
    ocv_table=LFP.ocv_table,
    cell_config=_LFP_BUS_CELL,
    self_discharge_pct_month=LFP.self_discharge_pct_month,
    charge_efficiency=LFP.charge_efficiency,
    float_voltage_v=LFP.float_voltage_v,
    description=LFP.description,
)

ELECTRIC_BUS = VehicleProfile(
    name="Electric City Bus (409.6V LFP)",
    vehicle_type="BUS",
    motor_power_kw=150.0,
    n_cells_series=128,             # 128 × 3.2 V = 409.6 V
    n_cells_parallel=2,             # 2P for 400 Ah total
    chemistry=_LFP_BUS_CHEM,
    pack_cfg=PackConfig(
        cells_series=128,
        cells_parallel=2,
        balance_v_threshold=0.008,  # tighter for large pack
        balance_current_a=1.0,
        soh_warning_pct=80.0,
        soh_critical_pct=70.0,      # buses need higher reliability
        pre_charge_resistor_ohm=2.0,
        isolation_fault_threshold_ohm=100.0,
    ),
    t_fan_on_c=30.0,
    t_fan_full_c=45.0,
    t_heat_on_c=5.0,
    has_active_cooling=True,        # liquid cooling mandatory at 150 kW
    charging_stages=2,
    notes=(
        "City/intercity electric bus. 128S2P LFP = 409.6V / 400Ah / 163.8 kWh. "
        "Liquid cooling. Centralized BMS with distributed cell monitoring."
    ),
)

# Registry
VEHICLE_REGISTRY = {
    "2W": TWO_WHEELER,
    "4W_RETRO": FOUR_WHEELER_RETRO,
    "4W_MODERN": FOUR_WHEELER_MODERN,
    "BUS": ELECTRIC_BUS,
}


def make_bms_controller(profile: VehicleProfile, soc_init=0.50, ambient_temp_c: float = 25.0):
    """
    Instantiate a BMSController configured for the given vehicle profile.

    Parameters
    ----------
    profile : VehicleProfile
        One of the presets from this module.
    soc_init : float or list[float]
        Initial state of charge [0–1].
    ambient_temp_c : float
        Ambient temperature [°C].

    Returns
    -------
    BMSController
    """
    from .bms_controller import BMSController

    return BMSController(
        n_cells=profile.n_cells_series,
        cell_cfg=profile.cell_config,
        pack_cfg=profile.pack_cfg,
        ekf_cfg=profile.ekf_cfg,
        soc_init=soc_init,
        ambient_temp_c=ambient_temp_c,
        ocv_table=profile.chemistry.ocv_table,
        vehicle_profile=profile,
    )

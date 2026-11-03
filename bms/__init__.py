"""
BMS — Battery Management System Python package.

Modules
-------
config            Cell / pack / EKF parameters
cell_model        2-RC Thevenin equivalent circuit model
soc_estimator     Extended Kalman Filter SOC estimation
soh_estimator     State-of-Health (capacity & resistance fade)
cell_balancer     Passive cell balancing controller
power_limits      Charge / discharge power limit calculator
supervisor        Supervisory state machine (CCCV, faults, contactors)
thermal           PWM fan / heater thermal management controller
chemistries       LFP / Lead-Acid / NMC / Na-Ion / LMFP / LTO / Li-S / SS-NMC profiles
vehicles          Vehicle presets (2W, 4W, BUS) + factory function
driving_profiles  Real-world driving conditions + degradation stress model
manufacturer_profiles  Ather / Ola / Tesla / BYD real-world tuned profiles
bms_controller    Top-level pack controller (integrates all modules)
simulation        Multi-vehicle, degradation, cold-temp, and driving simulations
datasets          Kaggle dataset integration: synthetic generators + validation
"""

from .bms_controller import BMSController, BMSPackState
from .chemistries import (
    CHEMISTRY_REGISTRY,
    LEAD_ACID,
    LFP,
    LI_SULFUR,
    LMFP,
    LTO,
    NA_ION,
    NMC,
    SOLID_STATE,
    ChemistryProfile,
)
from .config import DEFAULT_CELL, DEFAULT_EKF, DEFAULT_PACK, CellConfig, EKFConfig, PackConfig
from .driving_profiles import (
    AGGRESSIVE,
    CITY,
    COLD_CLIMATE,
    DRIVING_PROFILES,
    GENTLE,
    HIGHWAY,
    REGEN_HEAVY,
    DrivingConditions,
    annual_soh_loss_pct,
    compute_stress_factors,
    efc_per_year,
    effective_deg_rate_per_efc,
    months_to_eol,
    soh_at_month,
)
from .manufacturer_profiles import (
    ATHER_450X,
    ATHER_DRIVING,
    BYD_DRIVING,
    BYD_HAN_EV,
    MANUFACTURER_DRIVING,
    MANUFACTURER_REGISTRY,
    OLA_DRIVING,
    OLA_S1_PRO,
    TESLA_DRIVING,
    TESLA_MODEL3_SR,
)
from .supervisor import AlertCode, BMSState, FaultCode
from .thermal import FanController, FanFaultCode, ThermalManagementState
from .vehicles import (
    ELECTRIC_BUS,
    FOUR_WHEELER_MODERN,
    FOUR_WHEELER_RETRO,
    TWO_WHEELER,
    VEHICLE_REGISTRY,
    VehicleProfile,
    make_bms_controller,
)

__all__ = [
    # Core controller
    "BMSController",
    "BMSPackState",
    # State machine
    "BMSState",
    "FaultCode",
    "AlertCode",
    # Configuration
    "CellConfig",
    "PackConfig",
    "EKFConfig",
    "DEFAULT_CELL",
    "DEFAULT_PACK",
    "DEFAULT_EKF",
    # Chemistries — established
    "ChemistryProfile",
    "LFP",
    "LEAD_ACID",
    "NMC",
    # Chemistries — emerging / next-generation
    "NA_ION",
    "LMFP",
    "LTO",
    "LI_SULFUR",
    "SOLID_STATE",
    "CHEMISTRY_REGISTRY",
    # Thermal
    "FanController",
    "ThermalManagementState",
    "FanFaultCode",
    # Vehicles
    "VehicleProfile",
    "TWO_WHEELER",
    "FOUR_WHEELER_RETRO",
    "FOUR_WHEELER_MODERN",
    "ELECTRIC_BUS",
    "VEHICLE_REGISTRY",
    "make_bms_controller",
    # Manufacturer profiles
    "ATHER_450X",
    "OLA_S1_PRO",
    "TESLA_MODEL3_SR",
    "BYD_HAN_EV",
    "ATHER_DRIVING",
    "OLA_DRIVING",
    "TESLA_DRIVING",
    "BYD_DRIVING",
    "MANUFACTURER_REGISTRY",
    "MANUFACTURER_DRIVING",
    # Driving profiles
    "DrivingConditions",
    "GENTLE",
    "CITY",
    "HIGHWAY",
    "AGGRESSIVE",
    "COLD_CLIMATE",
    "REGEN_HEAVY",
    "DRIVING_PROFILES",
    "compute_stress_factors",
    "efc_per_year",
    "effective_deg_rate_per_efc",
    "annual_soh_loss_pct",
    "months_to_eol",
    "soh_at_month",
]

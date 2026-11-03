"""
BMS configuration parameters for a 16S LiFePO4 battery pack.

Chemistry: Lithium Iron Phosphate (LFP)
Pack topology: 16 cells in series, 1 cell in parallel (16S1P)
Nominal pack voltage: 51.2 V  (16 × 3.2 V)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CellConfig:
    """Electrochemical parameters for a single LFP cell."""

    # Capacity
    nominal_capacity_ah: float = 100.0       # Ah
    soc_min: float = 0.05                    # 5 % lower cut-off
    soc_max: float = 0.95                    # 95 % upper cut-off

    # Voltage limits
    v_nominal: float = 3.2                   # V
    v_max: float = 3.65                      # V  (charge cut-off)
    v_min: float = 2.50                      # V  (discharge cut-off)

    # Thevenin equivalent-circuit elements (at 25 °C, 50 % SOC)
    r0: float = 0.002                        # Ω  series resistance
    r1: float = 0.001                        # Ω  RC-branch 1 (fast dynamics)
    c1: float = 3000.0                       # F
    r2: float = 0.002                        # Ω  RC-branch 2 (slow diffusion)
    c2: float = 30000.0                      # F

    # Current limits (positive = discharge convention)
    i_max_charge: float = -50.0              # A  (0.5 C charge)
    i_max_discharge: float = 200.0           # A  (2 C discharge)
    i_cc_charge: float = -50.0              # A  constant-current charge
    i_cv_term: float = -5.0                 # A  CV phase termination current

    # Temperature limits (°C)
    t_max_charge: float = 45.0
    t_min_charge: float = 0.0
    t_max_discharge: float = 60.0
    t_min_discharge: float = -20.0
    t_nominal: float = 25.0

    # Thermal model (lumped capacitance)
    thermal_capacity_j_k: float = 1500.0    # J/K  heat capacity of cell
    thermal_resistance_k_w: float = 5.0     # K/W  cell-to-ambient thermal resistance


@dataclass(frozen=True)
class PackConfig:
    """Pack-level topology and protection limits."""

    cells_series: int = 16
    cells_parallel: int = 1

    # Cell balancing
    balance_v_threshold: float = 0.010      # V  imbalance threshold (10 mV)
    balance_current_a: float = 0.5          # A  passive bypass current

    # SOH thresholds
    soh_warning_pct: float = 80.0
    soh_critical_pct: float = 60.0

    # Contactor / isolation
    pre_charge_resistor_ohm: float = 10.0   # Ω  inrush-limiting resistor
    isolation_fault_threshold_ohm: float = 100.0  # kΩ  minimum acceptable isolation


@dataclass(frozen=True)
class EKFConfig:
    """Extended Kalman Filter tuning parameters."""

    # Initial state uncertainty (diagonal of P0)
    p0_soc: float = 0.01
    p0_vrc1: float = 1e-4
    p0_vrc2: float = 1e-4

    # Process noise (diagonal of Q)
    q_soc: float = 1e-6
    q_vrc1: float = 1e-5
    q_vrc2: float = 1e-5

    # Measurement noise variance (R)
    r_voltage: float = 1e-4                 # V²


# OCV–SOC lookup table for LFP at 25 °C
# Columns: [SOC (0–1), OCV (V)]
OCV_SOC_TABLE = np.array([
    [0.00, 2.500],
    [0.05, 3.000],
    [0.10, 3.120],
    [0.20, 3.180],
    [0.30, 3.210],
    [0.40, 3.230],
    [0.50, 3.240],
    [0.60, 3.250],
    [0.70, 3.260],
    [0.80, 3.280],
    [0.90, 3.320],
    [0.95, 3.400],
    [1.00, 3.650],
])

DEFAULT_CELL = CellConfig()
DEFAULT_PACK = PackConfig()
DEFAULT_EKF = EKFConfig()

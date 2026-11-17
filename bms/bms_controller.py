"""
Top-level Battery Management System controller.

Integrates:
  - N-cell pack (individual CellModel instances)
  - Per-cell EKF SOC estimator
  - Pack-level SOH estimator
  - Passive cell balancer
  - Power limits calculator
  - Supervisory state machine

Usage:
    from bms.bms_controller import BMSController, BMSPackState
    bms = BMSController(n_cells=16)
    state = bms.step(current=50.0, dt=1.0)   # 50 A discharge, 1-second step
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .config import CellConfig, PackConfig, EKFConfig, DEFAULT_CELL, DEFAULT_PACK, DEFAULT_EKF
from .cell_model import CellModel, OCV_SOC_TABLE
from .soc_estimator import EKFSOCEstimator
from .soh_estimator import SOHEstimator, SOHState
from .cell_balancer import CellBalancer, BalancingState
from .power_limits import PowerLimitsCalculator, PowerLimits
from .supervisor import SupervisoryController, SupervisorOutput, BMSState, AlertCode
from .thermal import FanController, ThermalManagementState


@dataclass
class BMSPackState:
    """Full observable state snapshot returned each control cycle."""
    # Time
    timestamp_s: float

    # Pack-level aggregates
    pack_voltage_v: float
    pack_current_a: float
    pack_soc_mean: float
    pack_soc_min: float
    pack_soc_max: float
    pack_temp_max_c: float
    pack_temp_min_c: float

    # Per-cell data
    cell_voltages_v: List[float]
    cell_soc_estimates: List[float]
    cell_temperatures_c: List[float]

    # Sub-system states
    soh: SOHState
    balancing: BalancingState
    power_limits: PowerLimits
    supervisor: SupervisorOutput
    thermal: ThermalManagementState

    # Fault summary
    fault_active: bool
    fault_description: str
    alerts: List[AlertCode]


class BMSController:
    """
    Pack-level Battery Management System.

    Parameters
    ----------
    n_cells : int
        Number of cells in series.
    cell_cfg : CellConfig
        Electrochemical cell parameters.
    pack_cfg : PackConfig
        Pack topology and protection parameters.
    ekf_cfg : EKFConfig
        Kalman filter noise covariances.
    soc_init : float or list[float]
        Initial SOC per cell (scalar → same for all, list → per cell).
    ambient_temp_c : float
        Ambient / coolant temperature for thermal model.
    """

    def __init__(
        self,
        n_cells: int = 16,
        cell_cfg: CellConfig = DEFAULT_CELL,
        pack_cfg: PackConfig = DEFAULT_PACK,
        ekf_cfg: EKFConfig = DEFAULT_EKF,
        soc_init=0.80,
        ambient_temp_c: float = 25.0,
        ocv_table=None,
        vehicle_profile=None,           # optional VehicleProfile for metadata
    ):
        self._n = n_cells
        self._cell_cfg = cell_cfg
        self._pack_cfg = pack_cfg
        self._ambient = ambient_temp_c
        self._time = 0.0
        self._vehicle_profile = vehicle_profile
        self._ocv_table = ocv_table if ocv_table is not None else OCV_SOC_TABLE

        # Per-cell initial SOC (allow slight spread to demonstrate balancing)
        if isinstance(soc_init, (int, float)):
            socs = [float(soc_init)] * n_cells
        else:
            socs = [float(s) for s in soc_init]

        # Instantiate subsystems
        self._cells: List[CellModel] = [
            CellModel(cell_cfg, soc_init=s, temp_init=ambient_temp_c, ocv_table=self._ocv_table)
            for s in socs
        ]
        self._ekf: List[EKFSOCEstimator] = [
            EKFSOCEstimator(cell, ekf_cfg, soc_init=s)
            for cell, s in zip(self._cells, socs)
        ]
        self._soh = SOHEstimator(
            nominal_capacity_ah=cell_cfg.nominal_capacity_ah,
            r0_fresh=cell_cfg.r0,
        )
        self._balancer = CellBalancer(
            n_cells=n_cells,
            v_threshold=pack_cfg.balance_v_threshold,
            bypass_current_a=pack_cfg.balance_current_a,
        )
        self._power_limits = PowerLimitsCalculator(cell_cfg, n_series=n_cells)

        # Determine charging stages from vehicle profile (default 2 = CC+CV)
        _stages = getattr(vehicle_profile, 'charging_stages', 2) if vehicle_profile else 2
        _float_v = getattr(vehicle_profile, 'chemistry', None)
        _float_v = _float_v.float_voltage_v if _float_v else cell_cfg.v_nominal

        self._supervisor = SupervisoryController(
            cv_voltage_per_cell=cell_cfg.v_max,
            cv_term_current_a=abs(cell_cfg.i_cv_term),
            cc_current_a=cell_cfg.i_max_charge,
            float_voltage_per_cell=_float_v,
            charging_stages=_stages,
            ov_fault_v=cell_cfg.v_max + 0.20,
            uv_fault_v=cell_cfg.v_min - 0.10,
            t_max_fault_c=cell_cfg.t_max_discharge,
            t_min_fault_c=cell_cfg.t_min_discharge,
        )

        # Fan / thermal controller (parameters from vehicle profile if available)
        t_fan_on = getattr(vehicle_profile, 't_fan_on_c', 35.0) if vehicle_profile else 35.0
        t_fan_full = getattr(vehicle_profile, 't_fan_full_c', 50.0) if vehicle_profile else 50.0
        t_heat = getattr(vehicle_profile, 't_heat_on_c', 5.0) if vehicle_profile else 5.0
        self._fan = FanController(t_fan_on=t_fan_on, t_fan_full=t_fan_full, t_heat_on=t_heat)

    # ------------------------------------------------------------------
    # Main control cycle
    # ------------------------------------------------------------------

    def step(
        self,
        current: float,
        dt: float,
        charge_requested: bool = False,
        discharge_requested: bool = False,
    ) -> BMSPackState:
        """
        Advance the BMS by one time step.

        Parameters
        ----------
        current : float
            Applied pack current [A]. Positive = discharge, negative = charge.
        dt : float
            Time step duration [s].
        charge_requested / discharge_requested : bool
            External command signals fed into the supervisory state machine.

        Returns
        -------
        BMSPackState
        """
        self._time += dt

        # Step each cell model; add bypass current for cells being balanced
        cell_states = []
        for i, (cell, ekf) in enumerate(zip(self._cells, self._ekf)):
            bypass_i = self._balancer.effective_current(i)
            effective_i = current + bypass_i
            state = cell.step(effective_i, dt, self._ambient)
            cell_states.append(state)

        # Collect cell voltages and temperatures
        v_cells = [s.v_terminal for s in cell_states]
        t_cells = [s.temperature for s in cell_states]

        # EKF SOC estimation
        soc_estimates = [
            ekf.update(current, v, dt)
            for ekf, v in zip(self._ekf, v_cells)
        ]

        # Pack-level aggregates
        pack_voltage = sum(v_cells)
        pack_soc_mean = float(np.mean(soc_estimates))
        pack_soc_min = float(np.min(soc_estimates))
        pack_soc_max = float(np.max(soc_estimates))
        t_max = float(np.max(t_cells))
        t_min = float(np.min(t_cells))

        # SOH estimation (uses weakest cell = min voltage)
        soh_state = self._soh.update(current, float(np.min(v_cells)), dt)

        # Cell balancing
        bal_state = self._balancer.update(
            cell_voltages=v_cells,
            soc_estimates=soc_estimates,
            balancing_enabled=(abs(current) < 10.0),  # only balance at low current
        )

        # Power limits
        limits = self._power_limits.compute(
            soc=pack_soc_mean,
            temperature=t_max,
            v_pack=pack_voltage,
            soh_pct=soh_state.soh_capacity_pct,
            temperature_min=t_min,
        )

        # Supervisory state machine
        sv_out = self._supervisor.update(
            v_cell_max=max(v_cells),
            v_cell_min=min(v_cells),
            t_cell_max=t_max,
            soc_mean=pack_soc_mean,
            current=current,
            v_spread=bal_state.v_spread,
            charge_requested=charge_requested,
            discharge_requested=discharge_requested,
            t_cell_min=t_min,
        )

        # Thermal management / fan controller
        thermal_state = self._fan.update(temperature=t_max)

        fault_active = sv_out.fault_code.value != 0
        return BMSPackState(
            timestamp_s=self._time,
            pack_voltage_v=round(pack_voltage, 4),
            pack_current_a=current,
            pack_soc_mean=round(pack_soc_mean, 4),
            pack_soc_min=round(pack_soc_min, 4),
            pack_soc_max=round(pack_soc_max, 4),
            pack_temp_max_c=round(t_max, 2),
            pack_temp_min_c=round(t_min, 2),
            cell_voltages_v=[round(v, 4) for v in v_cells],
            cell_soc_estimates=[round(s, 4) for s in soc_estimates],
            cell_temperatures_c=[round(t, 2) for t in t_cells],
            soh=soh_state,
            balancing=bal_state,
            power_limits=limits,
            supervisor=sv_out,
            thermal=thermal_state,
            fault_active=fault_active,
            fault_description=sv_out.fault_code.name if fault_active else "NONE",
            alerts=sv_out.alerts,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def bms_state(self) -> BMSState:
        return self._supervisor.state

    @property
    def n_cells(self) -> int:
        return self._n

    def reset_fault(self) -> None:
        """Clear fault state and transition to IDLE (if fault is recoverable)."""
        from .supervisor import FaultCode
        self._supervisor._fault_code = FaultCode.NONE
        self._supervisor._transition(BMSState.IDLE)

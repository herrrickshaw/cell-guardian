"""
Supervisory state machine for the BMS.

States:
    INIT            — startup self-check
    IDLE            — contactors open, no current flow
    PRE_CHARGE      — pre-charge resistor in circuit (inrush limiting)
    CHARGING_CC     — Stage 1: constant-current bulk charge
    CHARGING_CV     — Stage 2: constant-voltage / topping charge (taper)
    CHARGING_FLOAT  — Stage 3: float / maintenance charge (lead-acid only)
    DISCHARGING     — supplying load
    BALANCING       — post-charge passive cell balancing
    FAULT           — non-recoverable fault; contactors open
    SHUTDOWN        — controlled shutdown sequence

3-stage charging (per IJAREEIE 2019 lead-acid paper):
    Stage 1 (CC / Bulk):    Constant current to 80 % SOC or V_absorption
    Stage 2 (CV / Topping): Constant voltage at V_absorption; current tapers
    Stage 3 (Float):        Voltage held at V_float (<V_absorption); indefinite

Fault → driver alert system:
    AlertCode enumerates driver-visible warnings (low SOC, high temp, …).

Convention: charge current is negative, discharge current is positive.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List


class BMSState(Enum):
    INIT = auto()
    IDLE = auto()
    PRE_CHARGE = auto()
    CHARGING_CC = auto()
    CHARGING_CV = auto()
    CHARGING_FLOAT = auto()     # Stage 3: float / maintenance charge
    DISCHARGING = auto()
    BALANCING = auto()
    FAULT = auto()
    SHUTDOWN = auto()


class FaultCode(Enum):
    NONE = 0
    OVER_VOLTAGE = auto()
    UNDER_VOLTAGE = auto()
    OVER_TEMPERATURE = auto()
    UNDER_TEMPERATURE = auto()
    OVER_CURRENT = auto()
    ISOLATION_FAULT = auto()
    INTERNAL_ERROR = auto()


class AlertCode(Enum):
    """Driver-visible warning codes (non-fatal, informational)."""
    NONE = 0
    LOW_SOC = auto()            # SOC approaching minimum
    HIGH_TEMPERATURE = auto()   # temperature above comfort zone
    SOH_WARNING = auto()        # capacity health below warning threshold
    CELL_IMBALANCE = auto()     # cells drifted beyond balance threshold
    CHARGING_COMPLETE = auto()  # charge cycle finished


@dataclass
class SupervisorOutput:
    state: BMSState
    fault_code: FaultCode
    alerts: List[AlertCode]      # driver-visible warnings (can be multiple)
    contactor_main: bool         # True = closed
    contactor_pre_charge: bool   # True = closed
    charge_enable: bool
    discharge_enable: bool
    balancing_enable: bool
    requested_current: float     # A  (negative = charge, positive = discharge)
    status_message: str


class SupervisoryController:
    """
    Event-driven BMS supervisory state machine.

    Parameters
    ----------
    cell_config : CellConfig (optional via kwargs)
        Uses v_max, v_min, i_max_charge, i_max_discharge, t_max_*.
    cv_voltage : float
        Constant-voltage phase target [V per cell].
    cv_termination_current : float
        CC→CV phase transition current magnitude [A].
    pre_charge_timeout_s : float
        Maximum time allowed in PRE_CHARGE state [s].
    """

    # Voltage thresholds (per cell, V)
    _OV_FAULT = 3.70
    _UV_FAULT = 2.40
    _OV_WARN = 3.65
    _UV_WARN = 2.50

    # Temperature thresholds (°C)
    _T_MAX_FAULT = 60.0
    _T_MIN_FAULT = -25.0

    def __init__(
        self,
        cv_voltage_per_cell: float = 3.65,
        cv_term_current_a: float = 5.0,
        cc_current_a: float = -50.0,        # negative = charge
        float_voltage_per_cell: float = 3.40,
        charging_stages: int = 2,           # 2=CC+CV, 3=CC+CV+Float (lead-acid)
        pre_charge_timeout_ticks: int = 10, # max ticks in PRE_CHARGE before fault
        ov_fault_v: float = 3.70,           # per-cell over-voltage fault threshold
        uv_fault_v: float = 2.40,           # per-cell under-voltage fault threshold
        t_max_fault_c: float = 60.0,
        t_min_fault_c: float = -25.0,
    ):
        self._cv_voltage = cv_voltage_per_cell
        self._cv_term = cv_term_current_a
        self._cc_current = cc_current_a
        self._float_voltage = float_voltage_per_cell
        self._charging_stages = charging_stages
        self._pre_charge_timeout = pre_charge_timeout_ticks
        self._OV_FAULT = ov_fault_v
        self._UV_FAULT = uv_fault_v
        self._T_MAX_FAULT = t_max_fault_c
        self._T_MIN_FAULT = t_min_fault_c

        self._state = BMSState.INIT
        self._fault_code = FaultCode.NONE
        self._pre_charge_ticks: int = 0
        self._pre_charge_for_discharge: bool = False
        self._init_ticks = 0

    # ------------------------------------------------------------------

    def update(
        self,
        v_cell_max: float,
        v_cell_min: float,
        t_cell_max: float,
        soc_mean: float,
        current: float,
        v_spread: float,
        charge_requested: bool,
        discharge_requested: bool,
        isolation_ok: bool = True,
        t_cell_min: Optional[float] = None,
    ) -> SupervisorOutput:
        """
        Evaluate one supervisory cycle and advance the state machine.

        Parameters
        ----------
        v_cell_max / v_cell_min : float
            Extremal cell terminal voltages [V].
        t_cell_max : float
            Hottest cell temperature [°C].
        soc_mean : float
            Pack mean SOC [0, 1].
        current : float
            Pack current [A]. Positive = discharge.
        v_spread : float
            Max – min cell voltage spread [V].
        charge_requested / discharge_requested : bool
            External command signals.
        isolation_ok : bool
            Isolation monitoring result.
        t_cell_min : float, optional
            Coldest cell temperature [°C], used for the under-temperature
            fault. Defaults to ``t_cell_max`` (single-sensor packs).

        Returns
        -------
        SupervisorOutput
        """
        # -- Global fault checks (highest priority) -------------------------
        t_min = t_cell_max if t_cell_min is None else t_cell_min
        fault = self._check_faults(v_cell_max, v_cell_min, t_cell_max, current, isolation_ok, t_min)
        if fault != FaultCode.NONE and self._state not in (BMSState.FAULT, BMSState.SHUTDOWN):
            self._transition(BMSState.FAULT)
            self._fault_code = fault

        # -- Driver alerts (non-fatal) ----------------------------------------
        alerts = self._check_alerts(soc_mean, t_cell_max, v_spread)

        # -- State transitions -----------------------------------------------
        if self._state == BMSState.INIT:
            self._output = self._handle_init(alerts)

        elif self._state == BMSState.IDLE:
            self._output = self._handle_idle(charge_requested, discharge_requested, alerts)

        elif self._state == BMSState.PRE_CHARGE:
            self._output = self._handle_pre_charge(v_cell_min, alerts)

        elif self._state == BMSState.CHARGING_CC:
            self._output = self._handle_charging_cc(v_cell_max, current, soc_mean, charge_requested, alerts)

        elif self._state == BMSState.CHARGING_CV:
            self._output = self._handle_charging_cv(current, v_spread, charge_requested, alerts)

        elif self._state == BMSState.CHARGING_FLOAT:
            self._output = self._handle_charging_float(current, v_spread, charge_requested, alerts)

        elif self._state == BMSState.DISCHARGING:
            self._output = self._handle_discharging(soc_mean, discharge_requested, alerts)

        elif self._state == BMSState.BALANCING:
            self._output = self._handle_balancing(v_spread, discharge_requested, alerts)

        elif self._state == BMSState.FAULT:
            self._output = self._handle_fault()

        elif self._state == BMSState.SHUTDOWN:
            self._output = self._handle_shutdown()

        return self._output

    @property
    def state(self) -> BMSState:
        return self._state

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_init(self, alerts) -> SupervisorOutput:
        self._init_ticks += 1
        if self._init_ticks >= 3:
            self._transition(BMSState.IDLE)
        return self._make_output(
            contactor_main=False, contactor_pre=False,
            charge_en=False, discharge_en=False, balancing_en=False,
            i_req=0.0, msg="Initializing...", alerts=alerts
        )

    def _handle_idle(self, charge_req: bool, discharge_req: bool, alerts) -> SupervisorOutput:
        if charge_req:
            self._transition(BMSState.PRE_CHARGE)
            self._pre_charge_ticks = 0
            self._pre_charge_for_discharge = False
        elif discharge_req:
            self._transition(BMSState.PRE_CHARGE)
            self._pre_charge_ticks = 0
            self._pre_charge_for_discharge = True
        return self._make_output(
            contactor_main=False, contactor_pre=False,
            charge_en=False, discharge_en=False, balancing_en=False,
            i_req=0.0, msg="Idle — contactors open", alerts=alerts
        )

    def _handle_pre_charge(self, v_cell_min: float, alerts) -> SupervisorOutput:
        self._pre_charge_ticks += 1
        if self._pre_charge_ticks > self._pre_charge_timeout:
            self._transition(BMSState.FAULT)
            self._fault_code = FaultCode.INTERNAL_ERROR
            return self._handle_fault()
        if self._pre_charge_ticks >= 2:
            next_st = BMSState.DISCHARGING if self._pre_charge_for_discharge else BMSState.CHARGING_CC
            self._transition(next_st)
        return self._make_output(
            contactor_main=False, contactor_pre=True,
            charge_en=False, discharge_en=False, balancing_en=False,
            i_req=0.0, msg="Pre-charging bus...", alerts=alerts
        )

    def _handle_charging_cc(self, v_max: float, current: float, soc: float, charge_req: bool, alerts) -> SupervisorOutput:
        if not charge_req:
            self._transition(BMSState.IDLE)
        elif v_max >= self._cv_voltage or soc >= 0.95:
            self._transition(BMSState.CHARGING_CV)
        return self._make_output(
            contactor_main=True, contactor_pre=False,
            charge_en=True, discharge_en=False, balancing_en=False,
            i_req=self._cc_current,
            msg=f"CC charge (Stage 1)  I={self._cc_current:.1f} A  SOC={soc*100:.0f}%",
            alerts=alerts
        )

    def _handle_charging_cv(self, current: float, v_spread: float, charge_req: bool, alerts) -> SupervisorOutput:
        if not charge_req:
            self._transition(BMSState.IDLE)
        elif abs(current) <= self._cv_term:
            if self._charging_stages >= 3:
                self._transition(BMSState.CHARGING_FLOAT)
            else:
                next_state = BMSState.BALANCING if v_spread > 0.010 else BMSState.IDLE
                self._transition(next_state)
        return self._make_output(
            contactor_main=True, contactor_pre=False,
            charge_en=True, discharge_en=False, balancing_en=False,
            i_req=0.0,
            msg=f"CV/Topping (Stage 2)  I_act={current:.1f} A  V_target={self._cv_voltage:.2f} V/cell",
            alerts=alerts
        )

    def _handle_charging_float(self, current: float, v_spread: float, charge_req: bool, alerts) -> SupervisorOutput:
        """Stage 3: float / maintenance charge (lead-acid paper requirement)."""
        if not charge_req:
            self._transition(BMSState.IDLE)
        float_alerts = [AlertCode.CHARGING_COMPLETE] + [a for a in alerts if a != AlertCode.CHARGING_COMPLETE]
        return self._make_output(
            contactor_main=True, contactor_pre=False,
            charge_en=True, discharge_en=False, balancing_en=False,
            i_req=0.0,
            msg=f"Float charge (Stage 3)  V_float={self._float_voltage:.2f} V/cell  I={current:.2f} A",
            alerts=float_alerts
        )

    def _handle_discharging(self, soc: float, discharge_req: bool, alerts) -> SupervisorOutput:
        if not discharge_req or soc <= 0.05:
            self._transition(BMSState.IDLE)
        return self._make_output(
            contactor_main=True, contactor_pre=False,
            charge_en=False, discharge_en=True, balancing_en=False,
            i_req=0.0, msg=f"Discharging  SOC={soc*100:.1f}%", alerts=alerts
        )

    def _handle_balancing(self, v_spread: float, discharge_req: bool, alerts) -> SupervisorOutput:
        if v_spread < 0.005 or discharge_req:
            self._transition(BMSState.IDLE)
        return self._make_output(
            contactor_main=False, contactor_pre=False,
            charge_en=False, discharge_en=False, balancing_en=True,
            i_req=0.0, msg=f"Balancing  DV={v_spread*1000:.1f} mV", alerts=alerts
        )

    def _handle_fault(self) -> SupervisorOutput:
        return self._make_output(
            contactor_main=False, contactor_pre=False,
            charge_en=False, discharge_en=False, balancing_en=False,
            i_req=0.0, msg=f"FAULT: {self._fault_code.name}", alerts=[]
        )

    def _handle_shutdown(self) -> SupervisorOutput:
        return self._make_output(
            contactor_main=False, contactor_pre=False,
            charge_en=False, discharge_en=False, balancing_en=False,
            i_req=0.0, msg="Shutdown", alerts=[]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_alerts(self, soc: float, t_max: float, v_spread: float) -> List[AlertCode]:
        alerts = []
        if soc < 0.15:
            alerts.append(AlertCode.LOW_SOC)
        if t_max > self._T_MAX_FAULT - 10.0:
            alerts.append(AlertCode.HIGH_TEMPERATURE)
        if v_spread > 0.050:
            alerts.append(AlertCode.CELL_IMBALANCE)
        return alerts

    def _check_faults(
        self, v_max: float, v_min: float, t_max: float, current: float, isolation_ok: bool,
        t_min: Optional[float] = None,
    ) -> FaultCode:
        if t_min is None:
            t_min = t_max
        if v_max > self._OV_FAULT:
            return FaultCode.OVER_VOLTAGE
        if v_min < self._UV_FAULT:
            return FaultCode.UNDER_VOLTAGE
        if t_max > self._T_MAX_FAULT:
            return FaultCode.OVER_TEMPERATURE
        if t_min < self._T_MIN_FAULT:
            return FaultCode.UNDER_TEMPERATURE
        if not isolation_ok:
            return FaultCode.ISOLATION_FAULT
        return FaultCode.NONE

    def _transition(self, new_state: BMSState) -> None:
        self._state = new_state

    def _make_output(
        self, *, contactor_main: bool, contactor_pre: bool,
        charge_en: bool, discharge_en: bool, balancing_en: bool,
        i_req: float, msg: str, alerts=None
    ) -> SupervisorOutput:
        return SupervisorOutput(
            state=self._state,
            fault_code=self._fault_code,
            alerts=alerts or [],
            contactor_main=contactor_main,
            contactor_pre_charge=contactor_pre,
            charge_enable=charge_en,
            discharge_enable=discharge_en,
            balancing_enable=balancing_en,
            requested_current=i_req,
            status_message=msg,
        )

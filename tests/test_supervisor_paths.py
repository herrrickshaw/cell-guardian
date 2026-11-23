"""State-machine boundary tests for ``SupervisoryController`` (offline, deterministic)."""
from __future__ import annotations

import pytest

from bms.supervisor import AlertCode, BMSState, FaultCode, SupervisoryController

NOMINAL = dict(
    v_cell_max=3.30, v_cell_min=3.20, t_cell_max=25.0, soc_mean=0.5,
    current=0.0, v_spread=0.001, charge_requested=False, discharge_requested=False,
)


def step(sv, **over):
    args = dict(NOMINAL)
    args.update(over)
    return sv.update(**args)


def to_idle(sv):
    for _ in range(3):
        step(sv)
    assert sv.state == BMSState.IDLE


def to_cc(sv):
    to_idle(sv)
    step(sv, charge_requested=True)  # IDLE -> PRE_CHARGE
    step(sv, charge_requested=True)  # tick 1
    step(sv, charge_requested=True)  # tick 2 -> CC
    assert sv.state == BMSState.CHARGING_CC


def to_cv(sv, **kw):
    to_cc(sv)
    step(sv, charge_requested=True, v_cell_max=3.65, **kw)
    assert sv.state == BMSState.CHARGING_CV


class TestInitAndIdle:
    def test_init_takes_three_ticks(self):
        sv = SupervisoryController()
        out = step(sv)
        assert sv.state == BMSState.INIT and out.status_message == "Initializing..."
        step(sv)
        assert sv.state == BMSState.INIT
        step(sv)
        assert sv.state == BMSState.IDLE

    def test_init_outputs_everything_off(self):
        out = step(SupervisoryController())
        assert not (out.contactor_main or out.contactor_pre_charge or out.charge_enable
                    or out.discharge_enable or out.balancing_enable)
        assert out.requested_current == 0.0

    def test_idle_without_request_stays(self):
        sv = SupervisoryController()
        to_idle(sv)
        out = step(sv)
        assert sv.state == BMSState.IDLE and out.status_message.startswith("Idle")

    def test_idle_discharge_request_precharges_then_discharges(self):
        sv = SupervisoryController()
        to_idle(sv)
        step(sv, discharge_requested=True)
        assert sv.state == BMSState.PRE_CHARGE
        step(sv, discharge_requested=True)
        assert sv.state == BMSState.PRE_CHARGE
        out = step(sv, discharge_requested=True)
        assert sv.state == BMSState.DISCHARGING
        assert out.contactor_pre_charge and not out.contactor_main

    def test_charge_request_wins_over_discharge(self):
        sv = SupervisoryController()
        to_idle(sv)
        step(sv, charge_requested=True, discharge_requested=True)
        step(sv, charge_requested=True, discharge_requested=True)
        step(sv, charge_requested=True, discharge_requested=True)
        assert sv.state == BMSState.CHARGING_CC

    def test_precharge_flag_reset_between_requests(self):
        sv = SupervisoryController()
        to_idle(sv)
        step(sv, discharge_requested=True)
        step(sv)
        step(sv)
        assert sv.state == BMSState.DISCHARGING
        step(sv)  # no discharge request -> IDLE
        assert sv.state == BMSState.IDLE
        step(sv, charge_requested=True)
        step(sv, charge_requested=True)
        step(sv, charge_requested=True)
        assert sv.state == BMSState.CHARGING_CC


class TestPreChargeTimeout:
    def test_timeout_boundary_is_strictly_greater(self):
        sv = SupervisoryController(pre_charge_timeout_ticks=2)
        to_idle(sv)
        step(sv, charge_requested=True)
        step(sv, charge_requested=True)
        out = step(sv, charge_requested=True)  # ticks == 2 == timeout: still ok
        assert out.fault_code == FaultCode.NONE and sv.state == BMSState.CHARGING_CC

    def test_timeout_exceeded_faults(self):
        sv = SupervisoryController(pre_charge_timeout_ticks=1)
        to_idle(sv)
        step(sv, charge_requested=True)
        step(sv, charge_requested=True)  # tick 1, not > 1
        assert sv.state == BMSState.PRE_CHARGE
        out = step(sv, charge_requested=True)  # tick 2 > 1
        assert sv.state == BMSState.FAULT
        assert out.fault_code == FaultCode.INTERNAL_ERROR
        assert out.status_message == "FAULT: INTERNAL_ERROR"
        assert not out.contactor_main and not out.contactor_pre_charge


class TestChargingStages:
    def test_cc_drops_to_idle_when_charge_not_requested(self):
        sv = SupervisoryController()
        to_cc(sv)
        step(sv, charge_requested=False)
        assert sv.state == BMSState.IDLE

    @pytest.mark.parametrize("v_max,soc,expected", [
        (3.64, 0.94, BMSState.CHARGING_CC),
        (3.65, 0.50, BMSState.CHARGING_CV),
        (3.00, 0.95, BMSState.CHARGING_CV),
        (3.00, 0.9499, BMSState.CHARGING_CC),
    ])
    def test_cc_to_cv_boundaries(self, v_max, soc, expected):
        sv = SupervisoryController(cv_voltage_per_cell=3.65)
        to_cc(sv)
        step(sv, charge_requested=True, v_cell_max=v_max, soc_mean=soc)
        assert sv.state == expected

    def test_cc_output_reports_configured_current(self):
        sv = SupervisoryController(cc_current_a=-42.0)
        to_cc(sv)
        out = step(sv, charge_requested=True)
        assert out.requested_current == -42.0
        assert out.contactor_main and out.charge_enable and not out.discharge_enable
        assert "CC charge" in out.status_message and "-42.0" in out.status_message

    def test_cv_stays_while_current_above_termination(self):
        sv = SupervisoryController(cv_term_current_a=5.0)
        to_cv(sv)
        out = step(sv, charge_requested=True, current=-5.01)
        assert sv.state == BMSState.CHARGING_CV
        assert out.requested_current == 0.0 and "CV/Topping" in out.status_message

    def test_cv_termination_current_uses_magnitude_and_inclusive(self):
        sv = SupervisoryController(cv_term_current_a=5.0)
        to_cv(sv)
        step(sv, charge_requested=True, current=-5.0, v_spread=0.001)
        assert sv.state == BMSState.IDLE

    def test_cv_balancing_threshold(self):
        sv = SupervisoryController(cv_term_current_a=5.0)
        to_cv(sv)
        step(sv, charge_requested=True, current=-1.0, v_spread=0.010)
        assert sv.state == BMSState.IDLE  # exactly 10 mV: not > 0.010

        sv2 = SupervisoryController(cv_term_current_a=5.0)
        to_cv(sv2)
        step(sv2, charge_requested=True, current=-1.0, v_spread=0.0101)
        assert sv2.state == BMSState.BALANCING

    def test_cv_drops_to_idle_when_charge_not_requested(self):
        sv = SupervisoryController()
        to_cv(sv)
        step(sv, charge_requested=False, current=-100.0)
        assert sv.state == BMSState.IDLE

    def test_three_stage_goes_to_float(self):
        sv = SupervisoryController(charging_stages=3, cv_term_current_a=5.0)
        to_cv(sv)
        step(sv, charge_requested=True, current=-1.0, v_spread=0.5)
        assert sv.state == BMSState.CHARGING_FLOAT

    def test_two_stage_never_floats(self):
        sv = SupervisoryController(charging_stages=2, cv_term_current_a=5.0)
        to_cv(sv)
        step(sv, charge_requested=True, current=-1.0)
        assert sv.state != BMSState.CHARGING_FLOAT

    def test_float_adds_complete_alert_first_without_duplicate(self):
        sv = SupervisoryController(charging_stages=3, cv_term_current_a=5.0, float_voltage_per_cell=3.40)
        to_cv(sv)
        step(sv, charge_requested=True, current=-1.0)
        assert sv.state == BMSState.CHARGING_FLOAT
        out = step(sv, charge_requested=True, current=-0.5, soc_mean=0.1)
        assert out.alerts == [AlertCode.CHARGING_COMPLETE, AlertCode.LOW_SOC]
        assert out.charge_enable and out.contactor_main and out.requested_current == 0.0
        assert "Float charge" in out.status_message and "3.40" in out.status_message
        assert sv.state == BMSState.CHARGING_FLOAT

    def test_float_leaves_when_charge_not_requested(self):
        sv = SupervisoryController(charging_stages=3, cv_term_current_a=5.0)
        to_cv(sv)
        step(sv, charge_requested=True, current=-1.0)
        step(sv, charge_requested=False)
        assert sv.state == BMSState.IDLE


class TestDischargeAndBalancing:
    def _discharging(self):
        sv = SupervisoryController()
        to_idle(sv)
        for _ in range(3):
            step(sv, discharge_requested=True)
        assert sv.state == BMSState.DISCHARGING
        return sv

    def test_discharging_output(self):
        sv = self._discharging()
        out = step(sv, discharge_requested=True, soc_mean=0.5)
        assert out.discharge_enable and out.contactor_main and not out.charge_enable
        assert "50.0%" in out.status_message

    def test_discharging_soc_floor_is_inclusive(self):
        sv = self._discharging()
        step(sv, discharge_requested=True, soc_mean=0.0501)
        assert sv.state == BMSState.DISCHARGING
        step(sv, discharge_requested=True, soc_mean=0.05)
        assert sv.state == BMSState.IDLE

    def test_discharging_stops_when_not_requested(self):
        sv = self._discharging()
        step(sv, discharge_requested=False)
        assert sv.state == BMSState.IDLE

    def _balancing(self):
        sv = SupervisoryController(cv_term_current_a=5.0)
        to_cv(sv)
        step(sv, charge_requested=True, current=-1.0, v_spread=0.02)
        assert sv.state == BMSState.BALANCING
        return sv

    def test_balancing_output_and_stay(self):
        sv = self._balancing()
        out = step(sv, v_spread=0.005)
        assert sv.state == BMSState.BALANCING
        assert out.balancing_enable and not out.contactor_main
        assert "5.0 mV" in out.status_message

    def test_balancing_done_below_5mv(self):
        sv = self._balancing()
        step(sv, v_spread=0.0049)
        assert sv.state == BMSState.IDLE

    def test_balancing_aborted_by_discharge_request(self):
        sv = self._balancing()
        step(sv, v_spread=0.02, discharge_requested=True)
        assert sv.state == BMSState.IDLE


class TestFaults:
    @pytest.mark.parametrize("over,code", [
        (dict(v_cell_max=3.7001), FaultCode.OVER_VOLTAGE),
        (dict(v_cell_min=2.399), FaultCode.UNDER_VOLTAGE),
        (dict(t_cell_max=60.01), FaultCode.OVER_TEMPERATURE),
        (dict(t_cell_min=-25.01), FaultCode.UNDER_TEMPERATURE),
        (dict(isolation_ok=False), FaultCode.ISOLATION_FAULT),
    ])
    def test_each_fault_trips(self, over, code):
        sv = SupervisoryController()
        to_idle(sv)
        out = step(sv, **over)
        assert sv.state == BMSState.FAULT and out.fault_code == code
        assert out.status_message == f"FAULT: {code.name}"

    @pytest.mark.parametrize("over", [
        dict(v_cell_max=3.70),
        dict(v_cell_min=2.40),
        dict(t_cell_max=60.0),
        dict(t_cell_min=-25.0),
        dict(isolation_ok=True),
    ])
    def test_limits_themselves_are_not_faults(self, over):
        sv = SupervisoryController()
        to_idle(sv)
        out = step(sv, **over)
        assert sv.state == BMSState.IDLE and out.fault_code == FaultCode.NONE

    def test_single_sensor_cold_uses_tmax(self):
        sv = SupervisoryController()
        to_idle(sv)
        out = step(sv, t_cell_max=-26.0)
        assert out.fault_code == FaultCode.UNDER_TEMPERATURE

    def test_t_cell_min_overrides_tmax_for_cold_check(self):
        sv = SupervisoryController()
        to_idle(sv)
        out = step(sv, t_cell_max=20.0, t_cell_min=-30.0)
        assert out.fault_code == FaultCode.UNDER_TEMPERATURE

    def test_fault_priority_order(self):
        sv = SupervisoryController()
        to_idle(sv)
        out = step(sv, v_cell_max=4.0, v_cell_min=1.0, t_cell_max=99.0, isolation_ok=False)
        assert out.fault_code == FaultCode.OVER_VOLTAGE
        sv2 = SupervisoryController()
        to_idle(sv2)
        out = step(sv2, v_cell_min=1.0, t_cell_max=99.0, isolation_ok=False)
        assert out.fault_code == FaultCode.UNDER_VOLTAGE
        sv3 = SupervisoryController()
        to_idle(sv3)
        out = step(sv3, t_cell_max=99.0, isolation_ok=False)
        assert out.fault_code == FaultCode.OVER_TEMPERATURE

    def test_fault_is_latched_and_first_code_kept(self):
        sv = SupervisoryController()
        to_idle(sv)
        step(sv, isolation_ok=False)
        out = step(sv, v_cell_max=4.0)  # a different fault does not overwrite the latch
        assert sv.state == BMSState.FAULT and out.fault_code == FaultCode.ISOLATION_FAULT
        out = step(sv)  # nominal inputs do not clear it
        assert sv.state == BMSState.FAULT and out.fault_code == FaultCode.ISOLATION_FAULT

    def test_fault_output_has_no_alerts_and_opens_everything(self):
        sv = SupervisoryController()
        to_idle(sv)
        out = step(sv, isolation_ok=False, soc_mean=0.01)
        assert out.alerts == []
        assert not (out.contactor_main or out.charge_enable or out.discharge_enable or out.balancing_enable)

    def test_custom_thresholds_are_used(self):
        sv = SupervisoryController(ov_fault_v=3.5, uv_fault_v=2.9, t_max_fault_c=40.0, t_min_fault_c=0.0)
        to_idle(sv)
        assert step(sv, v_cell_max=3.51).fault_code == FaultCode.OVER_VOLTAGE
        sv = SupervisoryController(ov_fault_v=3.5, uv_fault_v=2.9, t_max_fault_c=40.0, t_min_fault_c=0.0)
        to_idle(sv)
        assert step(sv, v_cell_min=2.89).fault_code == FaultCode.UNDER_VOLTAGE
        sv = SupervisoryController(t_max_fault_c=40.0)
        to_idle(sv)
        assert step(sv, t_cell_max=40.5).fault_code == FaultCode.OVER_TEMPERATURE
        sv = SupervisoryController(t_min_fault_c=0.0)
        to_idle(sv)
        assert step(sv, t_cell_max=-0.5).fault_code == FaultCode.UNDER_TEMPERATURE


class TestAlerts:
    @pytest.mark.parametrize("over,alerts", [
        (dict(soc_mean=0.1499), [AlertCode.LOW_SOC]),
        (dict(soc_mean=0.15), []),
        (dict(t_cell_max=50.01), [AlertCode.HIGH_TEMPERATURE]),
        (dict(t_cell_max=50.0), []),
        (dict(v_spread=0.0501), [AlertCode.CELL_IMBALANCE]),
        (dict(v_spread=0.050), []),
        (dict(soc_mean=0.0, t_cell_max=55.0, v_spread=0.1),
         [AlertCode.LOW_SOC, AlertCode.HIGH_TEMPERATURE, AlertCode.CELL_IMBALANCE]),
    ])
    def test_alert_thresholds(self, over, alerts):
        sv = SupervisoryController()
        to_idle(sv)
        assert step(sv, **over).alerts == alerts

    def test_high_temp_alert_tracks_custom_fault_limit(self):
        sv = SupervisoryController(t_max_fault_c=40.0)
        to_idle(sv)
        assert step(sv, t_cell_max=30.5).alerts == [AlertCode.HIGH_TEMPERATURE]
        assert step(sv, t_cell_max=29.5).alerts == []

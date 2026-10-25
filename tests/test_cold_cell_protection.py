"""Cold-side protection must look at the COLDEST cell, hot-side at the hottest.

A pack with one cell at -30 C and the rest at 25 C is below the LFP discharge
(-20 C) and charge (0 C) limits. The supervisor and power limits were fed only
t_max, so this pack was neither faulted nor charge-blocked.
"""
import pytest

from bms.bms_controller import BMSController
from bms.power_limits import PowerLimitsCalculator
from bms.supervisor import BMSState, SupervisoryController


def _pack_with_one_cold_cell(cold_c=-30.0):
    bms = BMSController(n_cells=4, soc_init=0.6, ambient_temp_c=25.0)
    bms._cells[2]._temperature = cold_c          # simulate a cell with a failed heater / cold corner
    return bms


def test_one_cold_cell_trips_under_temperature_fault():
    bms = _pack_with_one_cold_cell()
    state = bms.step(current=0.0, dt=1.0)
    assert state.pack_temp_min_c < -20.0
    assert state.fault_active
    assert state.fault_description == "UNDER_TEMPERATURE"


def test_one_cold_cell_blocks_charging():
    bms = _pack_with_one_cold_cell(cold_c=-5.0)  # below LFP t_min_charge = 0 C, above discharge limit
    state = bms.step(current=0.0, dt=1.0)
    assert state.power_limits.i_max_charge == pytest.approx(0.0)


def test_power_limits_cold_checks_use_min_temperature():
    calc = PowerLimitsCalculator()
    lim = calc.compute(soc=0.5, temperature=25.0, v_pack=51.2, temperature_min=-25.0)
    assert lim.i_max_discharge == 0.0
    assert lim.i_max_charge == 0.0
    assert "under-temp-discharge-block" in lim.derate_reason


def test_power_limits_default_min_temperature_is_backward_compatible():
    calc = PowerLimitsCalculator()
    a = calc.compute(soc=0.5, temperature=5.0, v_pack=51.2)
    b = calc.compute(soc=0.5, temperature=5.0, v_pack=51.2, temperature_min=5.0)
    assert a == b


def test_supervisor_under_temp_uses_t_cell_min():
    sv = SupervisoryController()
    kw = dict(v_cell_max=3.3, v_cell_min=3.25, t_cell_max=25.0, soc_mean=0.5, current=0.0,
              v_spread=0.005, charge_requested=False, discharge_requested=False)
    for _ in range(4):
        sv.update(**kw)
    assert sv.state == BMSState.IDLE
    sv.update(**kw, t_cell_min=-30.0)
    assert sv.state == BMSState.FAULT

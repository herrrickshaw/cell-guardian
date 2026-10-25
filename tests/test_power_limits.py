"""PowerLimitsCalculator boundary values, hand-computed from DEFAULT_CELL.

DEFAULT_CELL: 200 A discharge / 50 A charge, SOC window 5-95 %, discharge
-20..60 C, charge 0..45 C.
"""
import pytest

from bms.power_limits import PowerLimitsCalculator

V_PACK = 51.2


@pytest.fixture
def calc():
    return PowerLimitsCalculator()


def test_nominal_point_gives_full_c_rate_and_power(calc):
    lim = calc.compute(soc=0.5, temperature=25.0, v_pack=V_PACK)
    assert (lim.i_max_discharge, lim.i_max_charge) == (200.0, -50.0)
    assert lim.p_max_discharge_w == pytest.approx(200 * V_PACK)
    assert lim.p_max_charge_w == pytest.approx(-50 * V_PACK)
    assert lim.derate_reason == "nominal"


@pytest.mark.parametrize("soc,expected", [(0.15, 200.0), (0.10, 100.0), (0.05, 0.0), (0.0, 0.0)])
def test_discharge_tapers_linearly_over_last_10pct_above_soc_min(calc, soc, expected):
    assert calc.compute(soc=soc, temperature=25.0, v_pack=V_PACK).i_max_discharge == pytest.approx(expected)


@pytest.mark.parametrize("soc,expected", [(0.90, -50.0), (0.925, -25.0), (0.95, 0.0), (1.0, 0.0)])
def test_charge_tapers_linearly_over_last_5pct_below_soc_max(calc, soc, expected):
    assert calc.compute(soc=soc, temperature=25.0, v_pack=V_PACK).i_max_charge == pytest.approx(expected)


def test_hot_discharge_derate_and_charge_block(calc):
    lim = calc.compute(soc=0.5, temperature=57.5, v_pack=V_PACK)    # 2.5 C below 60 C
    assert lim.i_max_discharge == pytest.approx(100.0)
    assert lim.i_max_charge == pytest.approx(0.0)                   # above 45 C


def test_hot_charge_derate(calc):
    lim = calc.compute(soc=0.5, temperature=42.5, v_pack=V_PACK)    # 2.5 C below 45 C
    assert lim.i_max_charge == pytest.approx(-25.0)
    assert lim.i_max_discharge == pytest.approx(200.0)


@pytest.mark.parametrize("t,expected", [(10.0, -50.0), (5.0, -25.0), (0.5, -5.0), (-0.1, 0.0)])
def test_cold_charge_derate_with_10pct_floor(calc, t, expected):
    # factor = max(0.1, (t - 0)/10) between 0 and 10 C, blocked below 0 C
    assert calc.compute(soc=0.5, temperature=t, v_pack=V_PACK).i_max_charge == pytest.approx(expected)


def test_discharge_blocked_below_minus_20c(calc):
    assert calc.compute(soc=0.5, temperature=-20.5, v_pack=V_PACK).i_max_discharge == 0.0
    assert calc.compute(soc=0.5, temperature=-19.5, v_pack=V_PACK).i_max_discharge == pytest.approx(200.0)


def test_soh_scales_both_limits(calc):
    lim = calc.compute(soc=0.5, temperature=25.0, v_pack=V_PACK, soh_pct=80.0)
    assert (lim.i_max_discharge, lim.i_max_charge) == (pytest.approx(160.0), pytest.approx(-40.0))


def test_combined_soc_and_temperature_factors_multiply(calc):
    lim = calc.compute(soc=0.10, temperature=57.5, v_pack=V_PACK)   # 0.5 x 0.5
    assert lim.i_max_discharge == pytest.approx(50.0)
    assert "low-SOC-discharge-derate" in lim.derate_reason
    assert "high-temp-discharge-derate" in lim.derate_reason

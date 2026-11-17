"""SOHEstimator: hand-computed capacity-fade and R0-EMA values."""
import pytest

from bms.bms_controller import BMSController
from bms.soh_estimator import SOHEstimator


def test_first_sample_does_not_fabricate_resistance_growth():
    """No previous voltage exists on the first call, so no dV/dI measurement can be made.

    Before the fix the previous voltage defaulted to 0 V, so the first current step
    measured |0 - 3.2 V| / 50 A = 64 mOhm (clipped to 50 mOhm) and a brand-new cell
    reported ~80.6 % resistance SOH after one sample.
    """
    est = SOHEstimator(r0_fresh=0.002)
    st = est.update(current=50.0, v_terminal=3.2, dt=1.0)
    assert st.r0_estimate == pytest.approx(0.002)
    assert st.soh_resistance_pct == pytest.approx(100.0)


def test_fresh_pack_resistance_soh_stays_near_100_on_first_drive():
    bms = BMSController(n_cells=4, soc_init=0.8)
    state = None
    for _ in range(10):
        state = bms.step(current=50.0, dt=1.0, discharge_requested=True)
    assert state.soh.soh_resistance_pct > 95.0


def test_linear_capacity_fade_hand_computed():
    # 100 A for 10 h = 1000 Ah throughput = 5 equivalent full cycles (2 x 100 Ah each)
    est = SOHEstimator(nominal_capacity_ah=100.0, degradation_rate=2e-4)
    st = est.update(current=100.0, v_terminal=3.2, dt=36000.0)
    assert st.total_ah_throughput == pytest.approx(1000.0)
    assert st.cycle_count_equiv == pytest.approx(5.0)
    assert st.soh_capacity_pct == pytest.approx(100.0 * (1 - 2e-4 * 5))   # 99.9 %
    assert st.capacity_estimate_ah == pytest.approx(99.9)


def test_charge_and_discharge_both_count_as_throughput():
    est = SOHEstimator(nominal_capacity_ah=50.0)
    est.update(current=-25.0, v_terminal=3.3, dt=3600.0)
    st = est.update(current=25.0, v_terminal=3.2, dt=3600.0)
    assert st.total_ah_throughput == pytest.approx(50.0)
    assert st.cycle_count_equiv == pytest.approx(0.5)


def test_capacity_soh_floors_at_zero():
    est = SOHEstimator(nominal_capacity_ah=1.0, degradation_rate=0.5)
    st = est.update(current=1.0, v_terminal=3.0, dt=3600.0 * 100)   # 50 EFC x 0.5 >> 1
    assert st.soh_capacity_pct == 0.0


def test_r0_ema_update_hand_computed():
    est = SOHEstimator(r0_fresh=0.002, r0_ema_alpha=0.01)
    est.update(current=0.0, v_terminal=3.30, dt=1.0)            # seed previous sample
    st = est.update(current=100.0, v_terminal=2.80, dt=1.0)     # dV = 0.5 V, dI = 100 A -> 5 mOhm
    expected = 0.99 * 0.002 + 0.01 * 0.005                      # 2.03 mOhm
    assert st.r0_estimate == pytest.approx(expected)
    assert st.soh_resistance_pct == pytest.approx(0.002 / expected * 100.0)


def test_small_current_change_is_ignored():
    est = SOHEstimator(r0_fresh=0.002)
    est.update(current=10.0, v_terminal=3.30, dt=1.0)
    st = est.update(current=10.9, v_terminal=3.00, dt=1.0)      # dI = 0.9 A < 1 A threshold
    assert st.r0_estimate == pytest.approx(0.002)


def test_r0_measurement_is_clipped_to_50_mohm():
    est = SOHEstimator(r0_fresh=0.002, r0_ema_alpha=1.0)        # alpha 1 -> estimate == measurement
    est.update(current=0.0, v_terminal=3.3, dt=1.0)
    st = est.update(current=2.0, v_terminal=2.3, dt=1.0)        # 1 V / 2 A = 0.5 Ohm -> clipped
    assert st.r0_estimate == pytest.approx(0.05)
    assert st.soh_resistance_pct == pytest.approx(4.0)

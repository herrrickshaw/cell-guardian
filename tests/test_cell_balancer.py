"""CellBalancer: passive bleed-down decisions."""
import pytest

from bms.cell_balancer import CellBalancer


def test_cells_above_min_plus_half_threshold_are_bled():
    bal = CellBalancer(n_cells=3, v_threshold=0.010, bypass_current_a=0.5)
    st = bal.update([3.300, 3.304, 3.320], [0.5, 0.5, 0.5])
    # target 3.300 + dead band 0.005: only 3.320 exceeds it
    assert st.is_balancing
    assert st.bypass_switches == [False, False, True]
    assert st.v_spread == pytest.approx(0.020)
    assert [bal.effective_current(i) for i in range(3)] == [0.0, 0.0, 0.5]


def test_spread_below_threshold_does_nothing():
    bal = CellBalancer(n_cells=2, v_threshold=0.010)
    st = bal.update([3.300, 3.305], [0.5, 0.5])
    assert not st.is_balancing and st.bypass_switches == [False, False]


def test_low_soc_cell_is_never_bled():
    bal = CellBalancer(n_cells=3, v_threshold=0.010, min_soc_to_balance=0.20)
    st = bal.update([3.30, 3.33, 3.34], [0.50, 0.15, 0.50])
    assert st.bypass_switches == [False, False, True]


def test_external_disable_clears_previously_set_switches():
    bal = CellBalancer(n_cells=2, v_threshold=0.010, bypass_current_a=0.5)
    bal.update([3.30, 3.34], [0.5, 0.5])
    assert bal.effective_current(1) == 0.5
    st = bal.update([3.30, 3.34], [0.5, 0.5], balancing_enabled=False)
    assert st.bypass_switches == [False, False]
    assert bal.effective_current(1) == 0.0


def test_soc_spread_reported():
    bal = CellBalancer(n_cells=2)
    st = bal.update([3.30, 3.30], [0.40, 0.55])
    assert st.soc_spread == pytest.approx(0.15)

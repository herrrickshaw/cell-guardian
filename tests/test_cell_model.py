"""CellModel: one discrete step checked against the closed-form ZOH equations."""
import numpy as np
import pytest

from bms.cell_model import CellModel, docv_dsoc, ocv_from_soc
from bms.config import DEFAULT_CELL


def test_single_step_matches_zero_order_hold_closed_form():
    cfg = DEFAULT_CELL
    cell = CellModel(soc_init=0.8, temp_init=25.0)
    st = cell.step(current=100.0, dt=1.0, ambient_temp=25.0)

    soc = 0.8 - 100.0 * 1.0 / (100.0 * 3600.0)
    vrc1 = cfg.r1 * (1 - np.exp(-1.0 / (cfg.r1 * cfg.c1))) * 100.0
    vrc2 = cfg.r2 * (1 - np.exp(-1.0 / (cfg.r2 * cfg.c2))) * 100.0
    ocv = 3.26 + 0.02 * (soc - 0.7) / 0.1                 # table segment 0.7 -> 0.8
    v = ocv - cfg.r0 * 100.0 - vrc1 - vrc2
    heat = 100.0 ** 2 * cfg.r0 + vrc1 ** 2 / cfg.r1 + vrc2 ** 2 / cfg.r2
    temp = 25.0 + heat * 1.0 / cfg.thermal_capacity_j_k

    assert st.soc == pytest.approx(soc)
    assert st.vrc1 == pytest.approx(vrc1)
    assert st.vrc2 == pytest.approx(vrc2)
    assert st.ocv == pytest.approx(ocv)
    assert st.v_terminal == pytest.approx(v)
    assert st.temperature == pytest.approx(temp)


def test_rc_voltages_relax_to_zero_at_rest():
    cell = CellModel(soc_init=0.5)
    for _ in range(60):
        cell.step(100.0, 1.0)
    for _ in range(3000):
        st = cell.step(0.0, 1.0)
    assert abs(st.vrc1) < 1e-9 and abs(st.vrc2) < 1e-6
    assert st.v_terminal == pytest.approx(st.ocv, abs=1e-6)


def test_cold_raises_resistance_linearly():
    warm = CellModel(temp_init=25.0)
    cold = CellModel(temp_init=5.0)
    vw = warm.step(100.0, 1e-6).v_terminal
    vc = cold.step(100.0, 1e-6).v_terminal
    # dt -> 0: only R0 acts; cold R0 = 0.002 * (1 + 0.005*20) = 0.0022 Ohm
    assert vw - vc == pytest.approx(100.0 * 0.0002, rel=1e-3)


def test_idle_hot_cell_cools_toward_ambient():
    cell = CellModel(temp_init=40.0)
    st = cell.step(0.0, 10.0, ambient_temp=25.0)
    # dT = -(40-25)/R_th * dt / C_th = -(15/5)*10/1500 = -0.02 K
    assert st.temperature == pytest.approx(40.0 - 0.02)


def test_ocv_interpolation_and_derivative_hand_values():
    assert ocv_from_soc(0.25) == pytest.approx(3.195)        # midway 3.18 -> 3.21
    assert docv_dsoc(0.25) == pytest.approx(0.30)            # 0.03 V / 0.1
    assert docv_dsoc(0.0) == pytest.approx(10.0, rel=1e-6)   # one-sided at the lower end: 0.5 V / 0.05


def test_ekf_matrices_use_requested_linearisation_point():
    cell = CellModel(soc_init=0.95)
    _, H_true, _, _ = cell.get_ekf_matrices(0.0, 1.0)
    _, H_est, ocv, _ = cell.get_ekf_matrices(0.0, 1.0, soc=0.25)
    assert H_est[0, 0] == pytest.approx(0.30)
    assert ocv == pytest.approx(3.195)
    assert H_true[0, 0] != pytest.approx(0.30)

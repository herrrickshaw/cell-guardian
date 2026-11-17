"""EKF SOC estimator: the filter may only use the measurement, the input and its own estimate.

The plant (CellModel) holds the TRUE state. A real BMS cannot read it, so the
estimator's output must not change when only the plant's hidden SOC changes.
"""
import numpy as np
import pytest

from bms.cell_model import CellModel, ocv_from_soc
from bms.config import DEFAULT_CELL, DEFAULT_EKF
from bms.soc_estimator import EKFSOCEstimator


def _one_step(plant_soc, est_soc=0.55, current=0.0, v_meas=3.25, dt=1.0):
    plant = CellModel(soc_init=plant_soc)
    ekf = EKFSOCEstimator(plant, soc_init=est_soc)
    ekf.update(current, v_meas, dt)
    return ekf


def test_estimate_independent_of_plant_true_soc():
    """Same estimate, same input, same measurement -> same output, whatever the hidden true SOC."""
    a = _one_step(plant_soc=0.15)   # steep OCV region (slope 0.6 V/SOC)
    b = _one_step(plant_soc=0.85)   # different slope (0.4 V/SOC)
    assert a.soc_estimate == pytest.approx(b.soc_estimate, abs=1e-12)
    np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-15)


def test_single_update_matches_hand_computed_gain():
    """One step at I=0 from x=[0.55,0,0]; Jacobian taken at the ESTIMATE (slope 0.1 V/SOC in 0.5-0.6)."""
    cfg, e = DEFAULT_CELL, DEFAULT_EKF
    dt = 1.0
    a1 = np.exp(-dt / (cfg.r1 * cfg.c1))
    a2 = np.exp(-dt / (cfg.r2 * cfg.c2))
    p11 = e.p0_soc + e.q_soc
    p22 = a1 ** 2 * e.p0_vrc1 + e.q_vrc1
    p33 = a2 ** 2 * e.p0_vrc2 + e.q_vrc2
    slope = (3.250 - 3.240) / 0.10                 # OCV table between 0.5 and 0.6
    s = slope ** 2 * p11 + p22 + p33 + e.r_voltage  # cross terms are zero (P diagonal)
    innovation = 3.25 - ocv_from_soc(0.55)         # 3.25 - 3.245 = 0.005 V
    expected = 0.55 + slope * p11 / s * innovation

    ekf = _one_step(plant_soc=0.95)                # plant's true SOC sits on a much steeper segment
    assert ekf.soc_estimate == pytest.approx(expected, rel=1e-6)


def test_zero_innovation_leaves_soc_unchanged_at_rest():
    ekf = _one_step(plant_soc=0.30, est_soc=0.55, v_meas=ocv_from_soc(0.55))
    assert ekf.soc_estimate == pytest.approx(0.55, abs=1e-12)


def test_coulomb_counting_prediction_without_voltage_information():
    """Huge measurement noise -> the filter reduces to coulomb counting: dSOC = I*dt/(Q*3600)."""
    from dataclasses import replace
    plant = CellModel(soc_init=0.8)
    ekf = EKFSOCEstimator(plant, ekf_cfg=replace(DEFAULT_EKF, r_voltage=1e12), soc_init=0.8)
    for _ in range(360):
        ekf.update(100.0, 3.0, 10.0)               # 100 Ah drawn from a 0.8 x 100 Ah cell
    assert ekf.soc_estimate == pytest.approx(0.0, abs=1e-6)   # clipped at the lower bound
    ekf2 = EKFSOCEstimator(plant, ekf_cfg=replace(DEFAULT_EKF, r_voltage=1e12), soc_init=0.8)
    for _ in range(36):
        ekf2.update(100.0, 3.0, 10.0)              # 10 Ah -> 0.10
    assert ekf2.soc_estimate == pytest.approx(0.70, abs=1e-6)


def test_converges_from_wrong_initial_estimate_on_sloped_region():
    """Plant at 15 % (steep LFP knee), estimator starts at 40 %: must close most of the gap at rest."""
    plant = CellModel(soc_init=0.15)
    ekf = EKFSOCEstimator(plant, soc_init=0.40)
    for _ in range(300):
        st = plant.step(0.0, 1.0)
        ekf.update(0.0, st.v_terminal, 1.0)
    assert abs(ekf.soc_estimate - plant.soc) < 0.03

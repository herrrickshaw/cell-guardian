"""
Extended Kalman Filter (EKF) for State-of-Charge estimation.

State vector: x = [SOC, Vrc1, Vrc2]^T
Input:        u = I  (current, positive = discharge)
Measurement:  y = V_terminal

Process model (discrete):
    x[k+1] = f(x[k], u[k])
    SOC[k+1]  = SOC[k] - I·dt / (Q·3600)
    Vrc1[k+1] = α1·Vrc1 + R1(1-α1)·I
    Vrc2[k+1] = α2·Vrc2 + R2(1-α2)·I

Measurement model:
    h(x) = OCV(SOC) - R0·I - Vrc1 - Vrc2

EKF two-step recursion:
    Predict: x̂⁻ = f(x̂, u),  P⁻ = F·P·Fᵀ + Q
    Update:  K  = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹
             x̂ = x̂⁻ + K·(y - h(x̂⁻))
             P  = (I - K·H)·P⁻
"""

import numpy as np
from .config import CellConfig, EKFConfig, DEFAULT_CELL, DEFAULT_EKF
from .cell_model import CellModel, ocv_from_soc


class EKFSOCEstimator:
    """
    Extended Kalman Filter that tracks [SOC, Vrc1, Vrc2] for a single cell.

    Parameters
    ----------
    cell_model : CellModel
        The plant model used to obtain F and H Jacobians at each step.
    ekf_cfg : EKFConfig
        Noise covariance tuning parameters.
    soc_init : float
        Initial SOC estimate [0, 1].
    """

    def __init__(
        self,
        cell_model: CellModel,
        ekf_cfg: EKFConfig = DEFAULT_EKF,
        soc_init: float = 0.80,
    ):
        self._cell = cell_model
        self._cfg = ekf_cfg

        # State estimate
        self._x = np.array([soc_init, 0.0, 0.0])

        # Error covariance
        self._P = np.diag([ekf_cfg.p0_soc, ekf_cfg.p0_vrc1, ekf_cfg.p0_vrc2])

        # Process noise covariance
        self._Q = np.diag([ekf_cfg.q_soc, ekf_cfg.q_vrc1, ekf_cfg.q_vrc2])

        # Measurement noise variance (scalar)
        self._R = ekf_cfg.r_voltage

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, current: float, v_measured: float, dt: float) -> float:
        """
        Run one EKF predict-update cycle.

        Parameters
        ----------
        current : float
            Measured cell current [A]. Positive = discharge.
        v_measured : float
            Measured terminal voltage [V].
        dt : float
            Time step [s].

        Returns
        -------
        float
            Updated SOC estimate [0, 1].
        """
        # ---- Predict ----
        x_pred = self._predict_state(self._x, current, dt)
        # Linearise at the predicted ESTIMATE, never at the plant's true SOC
        F, H, ocv, r0 = self._cell.get_ekf_matrices(current, dt, soc=float(x_pred[0]))
        P_pred = F @ self._P @ F.T + self._Q

        # ---- Update ----
        v_pred = self._cell.ocv_for_soc(x_pred[0]) - r0 * current - x_pred[1] - x_pred[2]
        innovation = v_measured - v_pred

        S = float((H @ P_pred @ H.T)[0, 0]) + self._R   # scalar innovation variance
        K = (P_pred @ H.T) / S                          # 3×1 gain

        self._x = x_pred + K.flatten() * innovation
        self._x[0] = float(np.clip(self._x[0], 0.0, 1.0))
        self._x[1] = float(self._x[1])
        self._x[2] = float(self._x[2])

        I3 = np.eye(3)
        self._P = (I3 - K @ H) @ P_pred

        return self.soc_estimate

    @property
    def soc_estimate(self) -> float:
        return float(self._x[0])

    @property
    def vrc1_estimate(self) -> float:
        return float(self._x[1])

    @property
    def vrc2_estimate(self) -> float:
        return float(self._x[2])

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_state(self, x: np.ndarray, current: float, dt: float) -> np.ndarray:
        """Propagate state forward using the nonlinear process model."""
        soc, vrc1, vrc2 = x
        cfg: CellConfig = self._cell.cfg

        # RC time constants (use nominal params — no temp derating inside EKF)
        tau1 = cfg.r1 * cfg.c1
        tau2 = cfg.r2 * cfg.c2
        alpha1 = np.exp(-dt / tau1)
        alpha2 = np.exp(-dt / tau2)

        new_soc = float(np.clip(soc - (current * dt) / (cfg.nominal_capacity_ah * 3600.0), 0.0, 1.0))
        new_vrc1 = alpha1 * vrc1 + cfg.r1 * (1.0 - alpha1) * current
        new_vrc2 = alpha2 * vrc2 + cfg.r2 * (1.0 - alpha2) * current

        return np.array([new_soc, new_vrc1, new_vrc2])

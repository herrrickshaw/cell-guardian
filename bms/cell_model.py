"""
2-RC Thevenin equivalent circuit model for a single battery cell.

Circuit topology (positive current = discharge)::

    R0          R1          R2
+--RRRR--+--RRRR--+--RRRR--+
|         |   |     |   |     |
OCV(soc) C1 Vrc1   C2 Vrc2  V_terminal
|         |         |         |
+---------+---------+---------+

Terminal voltage:
    V_t = OCV(SOC) - R0·I - Vrc1 - Vrc2

Discrete-time state equations (ZOH exact):
    SOC[k+1]  = SOC[k] - (I · dt) / (Q · 3600)
    Vrc1[k+1] = α1 · Vrc1[k] + R1·(1 - α1) · I      α1 = exp(-dt / τ1)
    Vrc2[k+1] = α2 · Vrc2[k] + R2·(1 - α2) · I      α2 = exp(-dt / τ2)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .config import CellConfig, OCV_SOC_TABLE, DEFAULT_CELL


def ocv_from_soc(soc: float, table: np.ndarray = OCV_SOC_TABLE) -> float:
    """Interpolate OCV from the lookup table. Clamps SOC to [0, 1]."""
    soc_clamped = float(np.clip(soc, table[0, 0], table[-1, 0]))
    return float(np.interp(soc_clamped, table[:, 0], table[:, 1]))


def docv_dsoc(soc: float, table: np.ndarray = OCV_SOC_TABLE, eps: float = 1e-4) -> float:
    """Numerical derivative dOCV/dSOC for EKF Jacobian."""
    soc_hi = min(soc + eps, table[-1, 0])
    soc_lo = max(soc - eps, table[0, 0])
    return (ocv_from_soc(soc_hi, table) - ocv_from_soc(soc_lo, table)) / (soc_hi - soc_lo)


@dataclass
class CellState:
    """Observable + internal state of one cell at a single time step."""
    soc: float          # state of charge [0, 1]
    vrc1: float         # RC-branch 1 voltage [V]
    vrc2: float         # RC-branch 2 voltage [V]
    v_terminal: float   # measured terminal voltage [V]
    ocv: float          # open-circuit voltage [V]
    temperature: float  # cell temperature [°C]
    current: float      # instantaneous current [A], positive = discharge


class CellModel:
    """
    Discrete-time 2-RC Thevenin cell model.

    Parameters
    ----------
    config : CellConfig
        Cell chemistry and equivalent-circuit parameters.
    soc_init : float
        Initial state of charge [0, 1].
    temp_init : float
        Initial cell temperature [°C].
    """

    def __init__(
        self,
        config: CellConfig = DEFAULT_CELL,
        soc_init: float = 0.80,
        temp_init: float = 25.0,
        ocv_table: Optional[np.ndarray] = None,
    ):
        self.cfg = config
        self._ocv_table = ocv_table if ocv_table is not None else OCV_SOC_TABLE
        self._soc = float(np.clip(soc_init, 0.0, 1.0))
        self._vrc1 = 0.0
        self._vrc2 = 0.0
        self._temperature = temp_init
        self._current = 0.0
        self._v_terminal = self.ocv_for_soc(self._soc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, current: float, dt: float, ambient_temp: float = 25.0) -> CellState:
        """
        Advance the cell model by one time step.

        Parameters
        ----------
        current : float
            Applied current [A]. Positive = discharge, negative = charge.
        dt : float
            Time step [s].
        ambient_temp : float
            Ambient / coolant temperature [°C].

        Returns
        -------
        CellState
            Cell state after the step.
        """
        cfg = self.cfg

        # Equivalent-circuit elements (temperature derating — simplified linear)
        r0, r1, c1, r2, c2 = self._param_at_temp()

        # Discrete-time zero-order-hold coefficients
        tau1 = r1 * c1
        tau2 = r2 * c2
        alpha1 = np.exp(-dt / tau1) if tau1 > 0 else 0.0
        alpha2 = np.exp(-dt / tau2) if tau2 > 0 else 0.0

        # State propagation
        new_soc = self._soc - (current * dt) / (cfg.nominal_capacity_ah * 3600.0)
        new_soc = float(np.clip(new_soc, 0.0, 1.0))

        new_vrc1 = alpha1 * self._vrc1 + r1 * (1.0 - alpha1) * current
        new_vrc2 = alpha2 * self._vrc2 + r2 * (1.0 - alpha2) * current

        # Terminal voltage
        ocv = self.ocv_for_soc(new_soc)
        v_terminal = ocv - r0 * current - new_vrc1 - new_vrc2

        # Thermal model: lumped capacitance (C_th · dT/dt = P_heat − (T − T_amb)/R_th)
        p_heat = (current**2 * r0
                  + (new_vrc1**2 / r1 if r1 > 0 else 0.0)
                  + (new_vrc2**2 / r2 if r2 > 0 else 0.0))
        p_cool = (self._temperature - ambient_temp) / cfg.thermal_resistance_k_w
        new_temp = self._temperature + (p_heat - p_cool) * dt / cfg.thermal_capacity_j_k

        # Commit
        self._soc = new_soc
        self._vrc1 = new_vrc1
        self._vrc2 = new_vrc2
        self._v_terminal = v_terminal
        self._temperature = new_temp
        self._current = current

        return CellState(
            soc=new_soc,
            vrc1=new_vrc1,
            vrc2=new_vrc2,
            v_terminal=v_terminal,
            ocv=ocv,
            temperature=new_temp,
            current=current,
        )

    @property
    def soc(self) -> float:
        return self._soc

    @property
    def v_terminal(self) -> float:
        return self._v_terminal

    @property
    def temperature(self) -> float:
        return self._temperature

    def ocv_for_soc(self, soc: float) -> float:
        """Look up OCV for given SOC using this cell's chemistry table."""
        return ocv_from_soc(soc, self._ocv_table)

    @property
    def state(self) -> np.ndarray:
        """Return [SOC, Vrc1, Vrc2] as a numpy vector."""
        return np.array([self._soc, self._vrc1, self._vrc2])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _param_at_temp(self):
        """Return (R0, R1, C1, R2, C2) with linear temperature correction.

        Cold increases resistance (delta_t < 0 → scale > 1); hot decreases it.
        Scale is clamped to [0.5, ∞) so resistance never more than doubles at cold.
        """
        cfg = self.cfg
        delta_t = self._temperature - cfg.t_nominal
        scale = max(1.0 - 0.005 * delta_t, 0.5)
        return cfg.r0 * scale, cfg.r1 * scale, cfg.c1, cfg.r2 * scale, cfg.c2

    def get_ekf_matrices(self, current: float, dt: float, soc: Optional[float] = None):
        """
        Return (F, H, ocv, r0) needed by the EKF at the current operating point.
        F  — state transition Jacobian (3×3)
        H  — measurement Jacobian (1×3)

        ``soc`` is the linearisation point. An estimator must pass its own
        (predicted) SOC estimate here; the default falls back to the plant's
        true SOC, which only a simulation can know.
        """
        if soc is None:
            soc = self._soc
        r0, r1, c1, r2, c2 = self._param_at_temp()
        tau1 = r1 * c1
        tau2 = r2 * c2
        alpha1 = np.exp(-dt / tau1) if tau1 > 0 else 0.0
        alpha2 = np.exp(-dt / tau2) if tau2 > 0 else 0.0

        F = np.array([
            [1.0, 0.0, 0.0],
            [0.0, alpha1, 0.0],
            [0.0, 0.0, alpha2],
        ])

        dOCV_dSOC = docv_dsoc(soc, table=self._ocv_table)
        H = np.array([[dOCV_dSOC, -1.0, -1.0]])

        ocv = self.ocv_for_soc(soc)
        return F, H, ocv, r0

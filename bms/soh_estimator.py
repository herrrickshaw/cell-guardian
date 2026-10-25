"""
State-of-Health (SOH) estimator.

Two independent SOH metrics are tracked:
1. SOH_capacity (%)  — usable capacity relative to nameplate (capacity fade)
2. SOH_resistance (%) — R0 relative to fresh cell (resistance growth)

Capacity SOH:
    Tracked via cycle counting and total charge throughput (Ah in/out).
    SOH_Q = (Q_usable / Q_nominal) × 100 %
    Q_usable decreases roughly linearly with total Ah throughput.

Resistance SOH:
    R0 is estimated from short current pulses using:
        ΔV = ΔI × R0   →   R0_est = ΔV / ΔI
    The exponentially-weighted moving average is updated each step.
    SOH_R = (R0_fresh / R0_est) × 100 %
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SOHState:
    soh_capacity_pct: float     # capacity SOH [0–100 %]
    soh_resistance_pct: float   # resistance SOH [0–100 %]
    r0_estimate: float          # estimated series resistance [Ω]
    capacity_estimate_ah: float # estimated usable capacity [Ah]
    total_ah_throughput: float  # cumulative Ah cycled since new
    cycle_count_equiv: float    # equivalent full cycles


class SOHEstimator:
    """
    Online SOH estimator for a battery cell.

    Parameters
    ----------
    nominal_capacity_ah : float
        Nameplate capacity [Ah].
    r0_fresh : float
        Series resistance of a fresh cell [Ω].
    degradation_rate : float
        Capacity loss per equivalent full cycle [0–1].
        Default 0.02 % / cycle → ~2 % capacity loss per 100 cycles.
    r0_ema_alpha : float
        Smoothing factor for R0 exponential moving average.
    """

    def __init__(
        self,
        nominal_capacity_ah: float = 100.0,
        r0_fresh: float = 0.002,
        degradation_rate: float = 2e-4,
        r0_ema_alpha: float = 0.01,
    ):
        self._q_nominal = nominal_capacity_ah
        self._r0_fresh = r0_fresh
        self._deg_rate = degradation_rate
        self._alpha = r0_ema_alpha

        self._total_ah = 0.0
        self._r0_est = r0_fresh

        # Previous step values for ΔV/ΔI estimation
        self._prev_current = 0.0
        self._prev_voltage = 0.0
        self._has_prev = False   # no dV/dI measurement possible before a first sample

    # ------------------------------------------------------------------

    def update(self, current: float, v_terminal: float, dt: float) -> SOHState:
        """
        Update SOH estimates for one time step.

        Parameters
        ----------
        current : float
            Cell current [A]. Positive = discharge.
        v_terminal : float
            Cell terminal voltage [V].
        dt : float
            Time step [s].

        Returns
        -------
        SOHState
        """
        # Charge throughput (absolute Ah)
        delta_ah = abs(current) * dt / 3600.0
        self._total_ah += delta_ah

        # Equivalent full cycles (each full cycle = 2 × Q_nominal Ah through)
        equiv_cycles = self._total_ah / (2.0 * self._q_nominal)

        # Capacity fade (linear model — replace with empirical model if available)
        q_fade_fraction = self._deg_rate * equiv_cycles
        q_usable = self._q_nominal * max(0.0, 1.0 - q_fade_fraction)
        soh_q = (q_usable / self._q_nominal) * 100.0

        # R0 estimation via current-step ΔV/ΔI
        delta_i = current - self._prev_current
        if self._has_prev and abs(delta_i) > 1.0:        # only when step > 1 A
            delta_v = self._prev_voltage - v_terminal    # voltage drops on discharge
            r0_meas = abs(delta_v / delta_i) if abs(delta_i) > 0 else self._r0_est
            r0_meas = float(np.clip(r0_meas, 0.0, 0.05))   # sanity bounds
            self._r0_est = (1.0 - self._alpha) * self._r0_est + self._alpha * r0_meas

        soh_r = (self._r0_fresh / self._r0_est) * 100.0
        soh_r = float(np.clip(soh_r, 0.0, 100.0))

        self._prev_current = current
        self._prev_voltage = v_terminal
        self._has_prev = True

        return SOHState(
            soh_capacity_pct=soh_q,
            soh_resistance_pct=soh_r,
            r0_estimate=self._r0_est,
            capacity_estimate_ah=q_usable,
            total_ah_throughput=self._total_ah,
            cycle_count_equiv=equiv_cycles,
        )

    @property
    def r0_estimate(self) -> float:
        return self._r0_est

    @property
    def total_ah_throughput(self) -> float:
        return self._total_ah

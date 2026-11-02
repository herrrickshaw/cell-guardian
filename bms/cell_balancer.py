"""
Passive cell balancing controller.

Strategy: bleed-down balancing (passive)
  - Compare terminal voltages (or SOC estimates) across all cells in a module.
  - When the spread exceeds `balance_v_threshold`, enable the bypass switch
    for cells whose voltage exceeds (V_min + threshold/2), bleeding charge
    through a fixed bypass resistor until the spread collapses.
  - Balancing is disabled during charging or high-current discharge (safety).

Cell balancing logic (mirrors the Stateflow diagram in the whitepaper):
    init  →  [spread > δV]  →  balancing  →  [spread ≤ δV]  →  finish
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BalancingState:
    is_balancing: bool
    bypass_switches: List[bool]   # True = bypass enabled for that cell
    v_spread: float               # max - min terminal voltage [V]
    soc_spread: float             # max - min SOC estimate


class CellBalancer:
    """
    Passive (dissipative) cell balancer for a series string.

    Parameters
    ----------
    n_cells : int
        Number of cells in series.
    v_threshold : float
        Voltage spread [V] above which balancing is triggered.
    bypass_current_a : float
        Bypass current through the bleed resistor per cell [A].
    min_soc_to_balance : float
        Do not balance below this SOC (protect weakest cell).
    """

    def __init__(
        self,
        n_cells: int,
        v_threshold: float = 0.010,
        bypass_current_a: float = 0.5,
        min_soc_to_balance: float = 0.20,
    ):
        self._n = n_cells
        self._threshold = v_threshold
        self._bypass_i = bypass_current_a
        self._min_soc = min_soc_to_balance
        self._switches = [False] * n_cells

    # ------------------------------------------------------------------

    def update(
        self,
        cell_voltages: List[float],
        soc_estimates: List[float],
        balancing_enabled: bool = True,
    ) -> BalancingState:
        """
        Compute the bypass switch commands for all cells.

        Parameters
        ----------
        cell_voltages : list[float]
            Terminal voltage of each cell [V].
        soc_estimates : list[float]
            SOC estimate for each cell [0, 1].
        balancing_enabled : bool
            External enable signal (False during high-rate discharge).

        Returns
        -------
        BalancingState
        """
        v_arr = np.array(cell_voltages)
        s_arr = np.array(soc_estimates)

        v_spread = float(v_arr.max() - v_arr.min())
        s_spread = float(s_arr.max() - s_arr.min())

        if not balancing_enabled or v_spread <= self._threshold:
            self._switches = [False] * self._n
            return BalancingState(
                is_balancing=False,
                bypass_switches=list(self._switches),
                v_spread=v_spread,
                soc_spread=s_spread,
            )

        # Determine target voltage: weakest (lowest) cell sets the floor
        v_target = float(v_arr.min())

        # Enable bypass for cells above target + dead-band to avoid chatter
        dead_band = self._threshold * 0.5
        for i in range(self._n):
            too_low_soc = soc_estimates[i] < self._min_soc
            needs_bleed = cell_voltages[i] > (v_target + dead_band)
            self._switches[i] = needs_bleed and not too_low_soc

        return BalancingState(
            is_balancing=any(self._switches),
            bypass_switches=list(self._switches),
            v_spread=v_spread,
            soc_spread=s_spread,
        )

    def effective_current(self, cell_index: int) -> float:
        """Extra bypass discharge current for cell [A] (positive = discharge)."""
        return self._bypass_i if self._switches[cell_index] else 0.0

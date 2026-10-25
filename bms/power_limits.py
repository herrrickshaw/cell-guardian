"""
Charge / discharge power limit calculator.

Power limits protect the pack from:
  - Over/under voltage  (SOC-based current limits)
  - Over-temperature    (temperature derating)
  - Under-temperature   (reduced charge rate at low temps)
  - Rate limits         (C-rate ceiling)

The allowed charge current is negative (convention: discharge positive).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .config import CellConfig, DEFAULT_CELL


@dataclass
class PowerLimits:
    i_max_discharge: float   # A  maximum discharge current (positive)
    i_max_charge: float      # A  maximum charge current (negative)
    p_max_discharge_w: float # W  maximum discharge power
    p_max_charge_w: float    # W  maximum charge power
    derate_reason: str       # human-readable reason for any derating


class PowerLimitsCalculator:
    """
    Compute instantaneous charge/discharge current limits.

    Parameters
    ----------
    config : CellConfig
        Cell chemistry and limits.
    n_series : int
        Number of cells in series (scales voltage for power calc).
    """

    def __init__(self, config: CellConfig = DEFAULT_CELL, n_series: int = 16):
        self.cfg = config
        self.n_series = n_series

    def compute(
        self,
        soc: float,
        temperature: float,
        v_pack: float,
        soh_pct: float = 100.0,
        temperature_min: Optional[float] = None,
    ) -> PowerLimits:
        """
        Compute power limits for the current operating point.

        Parameters
        ----------
        soc : float
            Pack-level (mean) SOC [0, 1].
        temperature : float
            Maximum cell temperature [°C].
        v_pack : float
            Pack terminal voltage [V].
        soh_pct : float
            Capacity SOH [%] – reduces limits as cell ages.
        temperature_min : float, optional
            Minimum cell temperature [°C] for the cold-side limits.
            Defaults to ``temperature`` (single-sensor packs).

        Returns
        -------
        PowerLimits
        """
        cfg = self.cfg
        reasons = []

        # Base limits (C-rate ceiling)
        i_dis_base = abs(cfg.i_max_discharge)      # positive
        i_chg_base = abs(cfg.i_max_charge)         # magnitude

        # SOC-based derating ------------------------------------------------
        # Discharge: taper to zero as SOC approaches soc_min
        soc_dis_window = 0.10   # start tapering 10 % above soc_min
        if soc < cfg.soc_min + soc_dis_window:
            soc_factor_dis = max(0.0, (soc - cfg.soc_min) / soc_dis_window)
            i_dis_base *= soc_factor_dis
            if soc_factor_dis < 1.0:
                reasons.append("low-SOC-discharge-derate")

        # Charge: taper to zero as SOC approaches soc_max
        soc_chg_window = 0.05
        if soc > cfg.soc_max - soc_chg_window:
            soc_factor_chg = max(0.0, (cfg.soc_max - soc) / soc_chg_window)
            i_chg_base *= soc_factor_chg
            if soc_factor_chg < 1.0:
                reasons.append("high-SOC-charge-derate")

        # Temperature derating -----------------------------------------------
        t = temperature
        t_cold = temperature if temperature_min is None else temperature_min
        temp_factor_dis = 1.0
        temp_factor_chg = 1.0

        if t > cfg.t_max_discharge:
            temp_factor_dis = 0.0
            reasons.append("over-temp-discharge-block")
        elif t > cfg.t_max_discharge - 5.0:
            temp_factor_dis = (cfg.t_max_discharge - t) / 5.0
            reasons.append("high-temp-discharge-derate")

        if t > cfg.t_max_charge:
            temp_factor_chg = 0.0
            reasons.append("over-temp-charge-block")
        elif t > cfg.t_max_charge - 5.0:
            temp_factor_chg = (cfg.t_max_charge - t) / 5.0
            reasons.append("high-temp-charge-derate")

        if t_cold < cfg.t_min_discharge:
            temp_factor_dis = 0.0
            reasons.append("under-temp-discharge-block")

        if t_cold < cfg.t_min_charge:
            temp_factor_chg = 0.0
            reasons.append("under-temp-charge-block")
        elif t_cold < cfg.t_min_charge + 10.0:
            temp_factor_chg *= max(0.1, (t_cold - cfg.t_min_charge) / 10.0)
            reasons.append("low-temp-charge-derate")

        # SOH derating (aged cell can deliver less power safely) ---------------
        soh_factor = soh_pct / 100.0
        i_dis_base *= soh_factor
        i_chg_base *= soh_factor

        # Apply temperature factors
        i_dis = i_dis_base * temp_factor_dis
        i_chg = i_chg_base * temp_factor_chg

        reason_str = ", ".join(reasons) if reasons else "nominal"

        return PowerLimits(
            i_max_discharge=round(i_dis, 2),
            i_max_charge=round(-i_chg, 2),       # negative sign = charge
            p_max_discharge_w=round(i_dis * v_pack, 1),
            p_max_charge_w=round(-i_chg * v_pack, 1),
            derate_reason=reason_str,
        )

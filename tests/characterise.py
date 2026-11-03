"""Characterisation digests for the BMS simulations and dataset tooling.

Runs the seeded, offline code paths (no Kaggle download, no network) and reduces
stdout and return values to SHA-256 digests.  The committed golden file was produced
from the pre-refactor code, so any numeric or formatting drift fails the test.

Regenerate (only for an intentional behaviour change):
    python -m tests.characterise tests/golden/characterisation.json [--slow]
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from typing import Callable, Dict


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _history_digest(history) -> dict:
    rows = [
        repr((s.timestamp_s, s.pack_soc_mean, s.pack_voltage_v, s.pack_current_a,
              s.pack_temp_max_c, s.thermal.fan_pwm_pct, s.soh.soh_capacity_pct,
              s.supervisor.state.name, tuple(a.name for a in s.alerts),
              tuple(s.cell_voltages_v)))
        for s in history
    ]
    return {"n": len(history), "sha": _sha("\n".join(rows))}


def _normalise(obj):
    """Round floats to 9 significant digits so digests survive numpy/pandas version drift."""
    if isinstance(obj, dict):
        return {k: _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    if isinstance(obj, float):
        return float(f"{obj:.9g}")
    if hasattr(obj, "item"):
        return _normalise(obj.item())
    return obj


def _digest(ret):
    if isinstance(ret, list) and ret and hasattr(ret[0], "pack_soc_mean"):
        return _history_digest(ret)
    return _sha(repr(ret))


def _cases(include_slow: bool) -> Dict[str, Callable[[], object]]:
    from bms import simulation as S
    from bms.datasets import (
        load_all_synthetic,
        loaders,
        print_validation_report,
        validate_bms_model,
    )
    from bms.vehicles import ELECTRIC_BUS, TWO_WHEELER

    def datasets():
        return {k: {"cols": list(v.columns), "n": len(v),
                    "sha": _sha(v.to_csv(index=False))}
                for k, v in load_all_synthetic().items()}

    def validation():
        res = validate_bms_model(load_all_synthetic())
        print_validation_report(res)
        return _sha(repr(_normalise(res)))

    def loader_rows():
        names = ["load_nasa", "load_degradation", "load_ev_charging", "load_rul",
                 "load_bms_telemetry", "load_distributed_bms"]
        return {n: len(getattr(loaders, n)("/nonexistent/dir")) for n in names}

    cases = {
        "2w": lambda: _digest(S.simulate_two_wheeler()),
        "deg_2w": lambda: _digest(S.simulate_degradation(TWO_WHEELER)),
        "deg_bus": lambda: _digest(S.simulate_degradation(
            ELECTRIC_BUS, n_cycles=300, deg_rate=0.0005)),
        "cold_2w": lambda: _digest(S.simulate_cold_temperature(TWO_WHEELER)),
        "driving_2w": lambda: _digest(S.simulate_driving_degradation(TWO_WHEELER)),
        "driving_bus": lambda: _digest(S.simulate_driving_degradation(
            ELECTRIC_BUS, simulation_years=10)),
        "manuf": lambda: _digest(S.simulate_manufacturer_comparison()),
        "table": lambda: _digest(S.print_comparison_table()),
        "datasets": datasets,
        "validation": validation,
        "loaders": loader_rows,
    }
    if include_slow:
        cases["bus"] = lambda: _digest(S.simulate_electric_bus())
        cases["4w"] = lambda: _digest(S.simulate_four_wheeler_retrofit())
    return cases


def capture(include_slow: bool = False) -> dict:
    """Run every case; return {case: {stdout_sha, stdout_len, ret}}."""
    out = {}
    for name, fn in _cases(include_slow).items():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ret = fn()
        out[name] = {"stdout_sha": _sha(buf.getvalue()),
                     "stdout_len": len(buf.getvalue()), "ret": ret}
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as fh:
        json.dump(capture("--slow" in sys.argv), fh, indent=1, sort_keys=True)

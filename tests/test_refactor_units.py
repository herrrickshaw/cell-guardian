"""Unit tests for the pure helpers extracted from bms.simulation and bms.datasets."""
from __future__ import annotations

import pandas as pd
import pytest

from bms import simulation as S
from bms.datasets import loaders, validation


@pytest.mark.parametrize("soh,expected", [(95.0, ""), (79.0, "~"), (59.0, "!"), (80.0, "")])
def test_soh_marker(soh, expected):
    assert S._soh_marker(soh, warn_pct=80.0, crit_pct=60.0) == expected


@pytest.mark.parametrize("soh,expected", [(90.0, "OK"), (75.0, "WARNING"), (50.0, "CRITICAL")])
def test_degradation_status(soh, expected):
    assert S._degradation_status(soh, 80.0, 60.0) == expected


def test_degradation_point_matches_closed_form():
    soh_q, q_use, soh_r, r0, dr0 = S._degradation_point(
        cycle=100, q_nominal=50.0, r0_fresh_mohm=10.0, deg_rate=0.001, r0_growth_rate=0.001)
    assert soh_q == pytest.approx(90.0)
    assert q_use == pytest.approx(45.0)
    assert r0 == pytest.approx(11.0)
    assert soh_r == pytest.approx(10.0 / 11.0 * 100.0)
    assert dr0 == pytest.approx(10.0)


def test_degradation_capacity_floors_at_zero():
    assert S._degradation_point(5000, 1.0, 1.0, 0.001, 0.0)[0] == 0.0


def test_left_aligned_format():
    assert S._left_aligned_format([3, 5]).format("a", "b") == "  a    b    "


def test_loader_falls_back_to_generator(tmp_path, capsys):
    marker = pd.DataFrame({"x": [1, 2]})
    out = loaders._load_or_synthesize("t", str(tmp_path), lambda: marker)
    assert out is marker
    assert "real data not found" in capsys.readouterr().out


def test_loader_reads_first_csv(tmp_path, capsys):
    (tmp_path / "sub").mkdir()
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(tmp_path / "sub" / "d.csv", index=False)
    out = loaders._load_or_synthesize("t", str(tmp_path), lambda: pytest.fail("no fallback"))
    assert len(out) == 3
    assert "loaded real data" in capsys.readouterr().out


def test_report_skips_missing_sections(capsys):
    validation.print_validation_report({})
    text = capsys.readouterr().out
    assert "VALIDATION REPORT" in text and "1. EKF SOC" not in text


def test_report_error_branch(capsys):
    validation.print_validation_report({"soc_estimation": {"error": "boom"}})
    assert "ERROR: boom" in capsys.readouterr().out

"""Seeded numeric + stdout regression against digests captured from the pre-refactor code."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.characterise import capture

GOLDEN = json.loads((Path(__file__).parent / "golden" / "characterisation.json").read_text())
FAST = {k: v for k, v in GOLDEN.items() if k not in ("bus", "4w")}


@pytest.fixture(scope="module")
def current():
    return capture(include_slow=False)


@pytest.mark.parametrize("case", sorted(FAST))
def test_matches_pre_refactor_golden(current, case):
    assert current[case] == FAST[case]


@pytest.mark.slow
@pytest.mark.parametrize("case", ["bus", "4w"])
def test_full_vehicle_runs_match_golden(case):
    from tests import characterise
    cases = characterise._cases(include_slow=True)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = cases[case]()
    assert {"stdout_sha": characterise._sha(buf.getvalue()),
            "stdout_len": len(buf.getvalue()), "ret": ret} == GOLDEN[case]

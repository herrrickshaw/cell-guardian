"""Offline tests for the Kaggle dataset loaders (real-CSV branch vs synthetic fallback)."""
from __future__ import annotations

import pandas as pd
import pytest

from bms.datasets import loaders

# (loader function, generator attribute it must fall back to, log tag, default sub-directory)
CASES = [
    ("load_nasa", "generate_nasa_battery", "nasa", "nasa"),
    ("load_degradation", "generate_degradation", "degradation", "degradation"),
    ("load_ev_charging", "generate_ev_charging", "ev_charging", "ev_charging"),
    ("load_rul", "generate_rul_features", "rul", "rul"),
    ("load_bms_telemetry", "generate_bms_telemetry", "bms_v21", "bms_v21"),
    ("load_distributed_bms", "generate_distributed_bms", "dist_bms", "dist_bms"),
]


class TestFirstCsv:
    def test_none_when_directory_missing_or_empty(self, tmp_path):
        assert loaders._first_csv(str(tmp_path / "absent")) is None
        assert loaders._first_csv(str(tmp_path)) is None

    def test_finds_nested_csv_only(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "a" / "b" / "data.csv").write_text("x\n1\n")
        assert loaders._first_csv(str(tmp_path)).endswith("data.csv")


@pytest.mark.parametrize("fn,gen,tag,sub", CASES)
class TestEachLoader:
    def test_reads_real_csv_when_present(self, fn, gen, tag, sub, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(loaders, gen, lambda: pytest.fail("synthetic generator must not run"))
        (tmp_path / "deep").mkdir()
        (tmp_path / "deep" / "f.csv").write_text("a,b\n1,2\n3,4\n5,6\n")
        df = getattr(loaders, fn)(str(tmp_path))
        assert list(df.columns) == ["a", "b"] and len(df) == 3
        out = capsys.readouterr().out
        assert f"[{tag}] loaded real data" in out and "(3 rows)" in out

    def test_falls_back_to_its_own_generator(self, fn, gen, tag, sub, tmp_path, monkeypatch, capsys):
        sentinel = pd.DataFrame({"synthetic": [1]})
        for _, other_gen, _, _ in CASES:
            monkeypatch.setattr(loaders, other_gen, lambda g=other_gen: pytest.fail(f"wrong generator {g}"))
        monkeypatch.setattr(loaders, gen, lambda: sentinel)
        assert getattr(loaders, fn)(str(tmp_path / "missing")) is sentinel
        out = capsys.readouterr().out
        assert f"[{tag}] real data not found" in out and "using synthetic" in out

    def test_default_directory_points_at_data_kaggle_subdir(self, fn, gen, tag, sub):
        default = getattr(loaders, fn).__defaults__[0]
        parts = default.replace("\\", "/").split("/")
        assert parts[-2:] == ["kaggle", sub]
        assert parts[-3] == "data"


def test_download_commands_cover_every_dataset_directory():
    cmds = loaders.DOWNLOAD_COMMANDS
    for _, _, _, sub in CASES:
        assert f"-p data/kaggle/{sub} --unzip" in cmds
    assert cmds.count("kaggle datasets download") == 6

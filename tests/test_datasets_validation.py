"""Offline tests for the synthetic dataset generators and the BMS validation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bms import datasets
from bms.datasets import synthetic, validation


class TestMetrics:
    def test_rmse_mae_exact(self):
        p, a = np.array([1.0, 2.0, 5.0]), np.array([1.0, 4.0, 2.0])
        assert validation._rmse(p, a) == pytest.approx(np.sqrt((0 + 4 + 9) / 3))
        assert validation._mae(p, a) == pytest.approx((0 + 2 + 3) / 3)

    def test_mape_ignores_zero_actuals(self):
        p, a = np.array([5.0, 9.0, 3.0]), np.array([0.0, 10.0, 2.0])
        assert validation._mape(p, a) == pytest.approx((10.0 + 50.0) / 2)


class TestRegistryAndGenerators:
    def test_registry_names_and_load_all(self):
        assert list(synthetic.DATASET_REGISTRY) == ["nasa_battery", "degradation", "ev_charging",
                                                     "rul_features", "bms_telemetry", "distributed_bms"]

    @pytest.mark.parametrize("name", list(synthetic.DATASET_REGISTRY))
    def test_generators_are_deterministic_and_non_empty(self, name):
        fn = synthetic.DATASET_REGISTRY[name]
        a, b = fn(), fn()
        assert len(a) > 0 and not a.isna().all().any()
        pd.testing.assert_frame_equal(a, b)

    @pytest.mark.parametrize("name", ["degradation", "rul_features", "ev_charging"])
    def test_seed_changes_output(self, name):
        fn = synthetic.DATASET_REGISTRY[name]
        assert not fn(seed=1).equals(fn(seed=2))

    def test_sizes_follow_arguments(self):
        assert len(synthetic.generate_degradation(n_cycles=20)) == 20
        assert len(synthetic.generate_rul_features(n_cycles=30)) == 30
        assert synthetic.generate_ev_charging(n_sessions=3)["session_id"].nunique() == 3
        assert synthetic.generate_bms_telemetry(duration_s=10, n_cells=3)["cell_id"].nunique() == 3
        assert synthetic.generate_distributed_bms(duration_s=5, n_modules=2)["module_id"].nunique() == 2

    def test_degradation_is_monotonic_fade(self):
        df = synthetic.generate_degradation(n_cycles=100)
        assert df["cycle"].tolist() == list(range(1, 101))
        assert df["soh_pct"].iloc[0] > df["soh_pct"].iloc[-1]
        assert df["r0_mohm"].iloc[-1] > df["r0_mohm"].iloc[0]

    def test_distributed_bms_has_one_degraded_module(self):
        df = synthetic.generate_distributed_bms(duration_s=60)
        soh = df.groupby("module_id")["soh_pct"].first()
        assert (soh < 90).sum() == 1 and soh.idxmin() == 3 and soh.min() == pytest.approx(85.0)

    def test_load_all_synthetic_matches_registry(self):
        out = datasets.load_all_synthetic()
        assert list(out) == list(synthetic.DATASET_REGISTRY)
        assert all(isinstance(v, pd.DataFrame) for v in out.values())


def deg_frame(soh, cap, r0):
    n = len(soh)
    return pd.DataFrame({"cycle": [10 * (i + 1) for i in range(n)], "soh_pct": soh,
                         "capacity_ah": cap, "r0_mohm": r0})


class TestDegradationModel:
    def test_perfect_linear_fade_has_zero_error(self):
        df = deg_frame([99, 98, 97, 96], [4.95, 4.9, 4.85, 4.8], [10, 10.1, 10.2, 10.3])
        r = validation.validate_degradation_model(df)
        assert r["n_cycles"] == 40
        assert r["soh_rmse_pct"] == 0.0 and r["soh_mae_pct"] == 0.0 and r["cap_rmse_ah"] == 0.0
        assert r["fitted_deg_rate"] == pytest.approx(0.1)           # %/cycle
        assert r["eol_cycle_pred"] in (199, 200)  # int() truncation of 200.0 +/- float noise
        assert r["fitted_r0_growth"] == pytest.approx((10.3 / 10 - 1) / 40 * 100)

    def test_zero_degradation_reports_sentinel_eol(self):
        df = deg_frame([100, 100, 100], [5.0, 5.0, 5.0], [10, 10, 10])
        r = validation.validate_degradation_model(df)
        assert r["eol_cycle_pred"] == 9999 and r["fitted_deg_rate"] == 0.0

    def test_nan_rows_are_dropped(self):
        df = deg_frame([99, np.nan, 97, 96], [4.95, 4.9, 4.85, 4.8], [10, 10.1, 10.2, 10.3])
        assert validation.validate_degradation_model(df)["n_cycles"] == 40


class TestRulModel:
    def test_linear_capacity_fade(self):
        cycles = np.arange(0, 10)
        cap = 2.0 - 0.05 * cycles
        eol = 2.0 * 0.8
        rul = np.maximum(0, (cap - eol) / 0.05)
        df = pd.DataFrame({"cycle": cycles, "capacity_ah": cap, "rul": rul})
        r = validation.validate_rul_model(df)
        assert r["fitted_fade_rate"] == pytest.approx(0.05)
        assert r["eol_cycle_predicted"] in (7, 8)  # int() truncation of 7.999999... is float noise
        assert r["rul_rmse_cycles"] == pytest.approx(0.0, abs=1e-6)
        assert r["rul_mae_cycles"] == pytest.approx(0.0, abs=1e-6)
        assert r["n_cycles"] == 9

    def test_mape_only_over_positive_true_rul(self):
        cycles = np.arange(0, 10)
        cap = 2.0 - 0.05 * cycles
        rul = np.zeros(10)
        rul[:5] = [8.0, 7.0, 6.0, 5.0, 4.0]
        df = pd.DataFrame({"cycle": cycles, "capacity_ah": cap, "rul": rul})
        r = validation.validate_rul_model(df)
        assert r["rul_mape_pct"] == pytest.approx(0.0, abs=1e-6)


def dist_frame(rows):
    return pd.DataFrame(rows, columns=["module_id", "voltage_v", "soc_pct", "temp_c", "soh_pct", "alert"])


class TestDistributedBms:
    def test_degraded_module_detected(self):
        df = dist_frame([
            (1, 3.3, 80, 25, 95, "OK"), (1, 3.3, 60, 25, 95, "OK"),
            (2, 3.3, 78, 26, 98, "OK"), (2, 3.3, 58, 26, 98, "WARN"),
            (3, 3.2, 70, 30, 85, "WARN"), (3, 3.2, 30, 30, 85, "WARN"),
        ])
        r = validation.validate_distributed_bms(df)
        assert r["degraded_module_ids"] == [3]
        assert r["degraded_avg_soh_pct"] == 85.0 and r["healthy_avg_soh_pct"] == 96.5
        assert r["soh_detectable"] is True
        assert r["degraded_alert_count"] == 2 and r["healthy_alert_count"] == 1
        assert [m["module_id"] for m in r["module_summary"]] == [1, 2, 3]
        assert r["module_summary"][2]["min_soc"] == 30 and r["module_summary"][2]["alert_count"] == 2

    def test_soh_threshold_is_strict_90(self):
        df = dist_frame([(1, 3.3, 50, 25, 90, "OK"), (2, 3.3, 50, 25, 89.9, "OK"), (3, 3.3, 50, 25, 100, "OK")])
        assert validation.validate_distributed_bms(df)["degraded_module_ids"] == [2]

    def test_not_detectable_when_degraded_min_soc_is_higher(self):
        df = dist_frame([(1, 3.3, 20, 25, 95, "OK"), (2, 3.3, 60, 25, 85, "OK")])
        assert validation.validate_distributed_bms(df)["soh_detectable"] is False


class TestPhysicsBackedValidators:
    def test_soc_estimation_no_discharge_rows(self):
        df = pd.DataFrame({"battery_id": ["B1"], "step_type": ["charge"], "cycle": [1], "capacity_ah": [2.0],
                           "current_a": [1.0], "voltage_v": [3.7]})
        assert validation.validate_soc_estimation(df) == {"error": "no discharge steps found"}

    def test_voltage_model_no_discharging_rows(self):
        df = pd.DataFrame({"cell_id": [1], "status": ["IDLE"], "soc_pct": [50.0], "current_a": [0.0],
                           "voltage_v": [3.3]})
        assert validation.validate_voltage_model(df) == {"error": "no discharging rows found"}

    def test_soc_estimation_on_synthetic_is_accurate(self):
        r = validation.validate_soc_estimation(synthetic.generate_nasa_battery(n_cycles=3))
        assert r["n_samples"] > 100
        assert 0 <= r["rmse_soc"] <= r["max_err_soc"] < 0.05
        assert r["mae_soc"] <= r["rmse_soc"] + 1e-9
        assert abs(r["ekf_soc_final"] - r["true_soc_final"]) < 0.05

    def test_voltage_model_on_synthetic_is_accurate(self):
        r = validation.validate_voltage_model(synthetic.generate_bms_telemetry(duration_s=700, n_cells=2))
        assert r["n_samples"] == 600  # capped by .head(600)
        assert r["v_mae_mv"] <= r["v_rmse_mv"] <= r["v_max_err_mv"] < 50


class TestRunnerAndReport:
    def test_runner_maps_present_datasets_only(self):
        data = {"degradation": synthetic.generate_degradation(n_cycles=30)}
        assert list(validation.validate_bms_model(data)) == ["degradation_model"]
        assert validation.validate_bms_model({}) == {}
        assert validation.validate_bms_model({"ev_charging": pd.DataFrame()}) == {}

    def test_runner_key_mapping(self):
        small = {
            "nasa_battery": synthetic.generate_nasa_battery(n_cycles=2),
            "degradation": synthetic.generate_degradation(n_cycles=20),
            "bms_telemetry": synthetic.generate_bms_telemetry(duration_s=50, n_cells=2),
            "rul_features": synthetic.generate_rul_features(n_cycles=40),
            "distributed_bms": synthetic.generate_distributed_bms(duration_s=20),
        }
        assert list(validation.validate_bms_model(small)) == [
            "soc_estimation", "degradation_model", "voltage_model", "rul_model", "distributed_bms"]

    def test_report_verdicts_use_strict_thresholds(self, capsys):
        base = {"n_samples": 10, "mae_soc": 0.0, "max_err_soc": 0.0, "ekf_soc_final": 0.5, "true_soc_final": 0.5}
        validation.print_validation_report({"soc_estimation": dict(base, rmse_soc=0.0299)})
        assert "PASS" in capsys.readouterr().out
        validation.print_validation_report({"soc_estimation": dict(base, rmse_soc=0.03)})
        assert "REVIEW" in capsys.readouterr().out
        v = {"n_samples": 5, "v_mae_mv": 1.0, "v_max_err_mv": 2.0, "v_mape_pct": 0.1}
        validation.print_validation_report({"voltage_model": dict(v, v_rmse_mv=19.99)})
        assert "PASS" in capsys.readouterr().out
        validation.print_validation_report({"voltage_model": dict(v, v_rmse_mv=20.0)})
        assert "REVIEW" in capsys.readouterr().out

    def test_report_error_branches_and_framing(self, capsys):
        validation.print_validation_report({"soc_estimation": {"error": "boom"}, "voltage_model": {
            "error": "bad", "n_samples": 0}})
        out = capsys.readouterr().out
        assert out.count("ERROR:") == 2
        assert "BMS MODEL VALIDATION REPORT" in out and out.count("=" * 74) == 4  # header and footer rules

    def test_report_full_on_synthetic(self, capsys):
        small = {
            "nasa_battery": synthetic.generate_nasa_battery(n_cycles=2),
            "degradation": synthetic.generate_degradation(n_cycles=20),
            "bms_telemetry": synthetic.generate_bms_telemetry(duration_s=50, n_cells=2),
            "rul_features": synthetic.generate_rul_features(n_cycles=40),
            "distributed_bms": synthetic.generate_distributed_bms(duration_s=20),
        }
        validation.print_validation_report(validation.validate_bms_model(small))
        out = capsys.readouterr().out
        for n in ("1. EKF SOC Estimation", "2. Capacity Fade Model", "3. 2-RC Voltage Model",
                  "4. RUL Estimation", "5. Multi-Module Health Detection", "Module summary"):
            assert n in out
        assert "Degraded modules  : [3]" in out and "SOC drop detected : YES" in out

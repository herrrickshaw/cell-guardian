"""Scenario-level tests for the lifecycle simulations in ``bms.simulation`` (degradation, cold, driving, makers)."""
from __future__ import annotations

import pytest

from bms import simulation as sim
from bms.vehicles import TWO_WHEELER


class TestDegradation:
    def test_history_rows_and_values(self, capsys):
        hist = sim.simulate_degradation(TWO_WHEELER, n_cycles=50, deg_rate=0.002, r0_growth_rate=0.004,
                                        print_every=25)
        assert [h["cycle"] for h in hist] == [1, 25, 50]
        assert hist[1]["soh_capacity_pct"] == pytest.approx(95.0)
        assert hist[2]["soh_capacity_pct"] == pytest.approx(90.0)
        cell = TWO_WHEELER.cell_config
        assert hist[0]["capacity_ah"] == pytest.approx(
            cell.nominal_capacity_ah * TWO_WHEELER.n_cells_parallel * 0.998, abs=0.01)
        assert hist[2]["r0_mohm"] == pytest.approx(cell.r0 * 1000 * 1.2, abs=0.001)
        assert hist[2]["soh_resistance_pct"] == pytest.approx(100 / 1.2, abs=0.01)
        out = capsys.readouterr().out
        assert "DEGRADATION SIMULATION" in out and "EOL at SOH=" in out

    def test_last_cycle_always_recorded_when_not_multiple(self):
        hist = sim.simulate_degradation(TWO_WHEELER, n_cycles=30, print_every=25)
        assert [h["cycle"] for h in hist] == [1, 25, 30]

    def test_status_thresholds(self):
        warn = TWO_WHEELER.pack_cfg.soh_warning_pct
        crit = TWO_WHEELER.pack_cfg.soh_critical_pct
        # choose a rate so that cycle 1 is just above warn, cycle 2 below warn, cycle 3 below crit
        step = (100.0 - warn) / 1.5 / 100.0
        hist = sim.simulate_degradation(TWO_WHEELER, n_cycles=3, deg_rate=step, print_every=1)
        assert hist[0]["status"] == "OK"
        assert hist[1]["status"] in ("WARNING", "CRITICAL")
        hist = sim.simulate_degradation(TWO_WHEELER, n_cycles=2, deg_rate=(100.0 - crit + 1) / 100.0 / 1.0,
                                        print_every=1)
        assert hist[-1]["status"] == "CRITICAL"

    def test_soh_floors_at_zero(self):
        hist = sim.simulate_degradation(TWO_WHEELER, n_cycles=10, deg_rate=0.5, print_every=10)
        assert hist[-1]["soh_capacity_pct"] == 0.0 and hist[-1]["capacity_ah"] == 0.0

    @pytest.mark.xfail(
        strict=True, raises=ZeroDivisionError,
        reason="simulation.py:250 computes round(0.02/deg_rate/100) in the banner, so deg_rate=0 raises "
               "ZeroDivisionError even though the EOL line (`if deg_rate > 0 else 99999`) handles zero",
    )
    def test_zero_rate_prints_sentinel_eol(self, capsys):
        sim.simulate_degradation(TWO_WHEELER, n_cycles=3, deg_rate=0.0, print_every=1)
        out = capsys.readouterr().out
        assert "EOL at SOH=" in out and "~99,999 cycles" in out

    def test_eol_projection_line(self, capsys):
        sim.simulate_degradation(TWO_WHEELER, n_cycles=2, deg_rate=0.001, print_every=1)
        eol_pct = TWO_WHEELER.pack_cfg.soh_warning_pct
        expected = int((100.0 - eol_pct) / 0.1)
        assert f"EOL at SOH={eol_pct}% : ~{expected:,} cycles" in capsys.readouterr().out


class TestCold:
    def test_sweep_results_structure_and_faults(self, capsys):
        res = sim.simulate_cold_temperature(TWO_WHEELER, ambient_temps_c=[25.0, -30.0], max_discharge_s=20)
        assert list(res) == [25.0, -30.0]
        warm, cold = res[25.0], res[-30.0]
        assert warm["fault"] == "-" and warm["discharge_blocked"] is False
        assert warm["t_amb_c"] == 25.0 and warm["ah_out"] > 0
        assert cold["fault"] == "UNDER_TEMPERATURE"      # trips before the tick-7 power-limit sample
        assert cold["ah_out"] < warm["ah_out"]
        assert cold["t_peak_c"] == -30.0
        assert "COLD TEMPERATURE SIMULATION" in capsys.readouterr().out

    def test_defaults_use_thirty_percent_of_max_discharge(self, capsys):
        res = sim.simulate_cold_temperature(TWO_WHEELER, ambient_temps_c=[25.0], max_discharge_s=3)
        cur = TWO_WHEELER.cell_config.i_max_discharge * TWO_WHEELER.n_cells_parallel * 0.30
        assert f"I_discharge={cur:.0f} A" in capsys.readouterr().out
        assert res[25.0]["ah_out"] == pytest.approx(3 * cur / 3600.0, abs=0.06)

    def test_default_temperature_sweep(self, capsys):
        res = sim.simulate_cold_temperature(TWO_WHEELER, max_discharge_s=1)
        assert list(res) == [25.0, 5.0, -10.0, -20.0, -30.0]


class TestDrivingAndManufacturers:
    def test_driving_results_per_condition(self, capsys):
        out = sim.simulate_driving_degradation(TWO_WHEELER, simulation_years=3)
        assert list(out)[:3] == ["Gentle City Driver", "Average City Driver", "Highway Commuter"]
        for res in out.values():
            years = [y for y, _ in res["soh_by_year"]]
            assert years == [0.0, 1.0, 2.0, 3.0]
            assert res["soh_by_year"][3][1] == pytest.approx(max(0.0, 100.0 - res["annual_loss_pct"] * 3), abs=0.06)
        assert out["Aggressive Driver"]["annual_loss_pct"] > out["Gentle City Driver"]["annual_loss_pct"]
        text = capsys.readouterr().out
        assert "DRIVING CONDITIONS DEGRADATION" in text and "Legend: ~ = SOH below warning" in text

    def test_driving_print_every_years_and_calendar_term(self):
        out = sim.simulate_driving_degradation(TWO_WHEELER, simulation_years=4, print_every_years=2,
                                               calendar_aging_pct_year=2.0)
        res = next(iter(out.values()))
        assert [y for y, _ in res["soh_by_year"]] == [0.0, 2.0, 4.0]
        base = sim.simulate_driving_degradation(TWO_WHEELER, simulation_years=4, calendar_aging_pct_year=0.5)
        assert res["annual_loss_pct"] == pytest.approx(next(iter(base.values()))["annual_loss_pct"] + 1.5)

    def test_manufacturer_comparison_runs(self, capsys):
        sim.simulate_manufacturer_comparison(simulation_years=2)
        assert "MANUFACTURER BMS COMPARISON" in capsys.readouterr().out

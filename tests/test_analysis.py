"""Each analysis function is checked against hand-built signals with known answers,
then the process tools are checked against the synthetic rev-A / rev-B runs."""

import copy

import numpy as np
import pandas as pd
import pytest

from powerops import analysis as A

from .conftest import make_df, plan_test


# ----------------------------------------------------------------- helpers
def test_find_runs_and_merge_runs():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert A.find_runs(mask) == [(1, 2), (5, 5), (7, 9)]
    assert A.merge_runs(A.find_runs(mask), max_gap=1) == [(1, 2), (5, 9)]


# ----------------------------------------------------------------- summarize_run
def test_summarize_run_steps_and_missing_tests(plan):
    df = pd.concat([make_df(500, step=1), make_df(500, step=5, t0=0.5)], ignore_index=True)
    r = A.summarize_run(df, plan)
    assert [(s["step"], s["start_s"], s["end_s"], s["plan_tests"]) for s in r["test_steps"]] == [
        (1, 0.0, 0.499, ["EFF-25"]), (5, 0.5, 0.999, ["BURST-1"])]
    assert r["plan_tests_without_data"] == ["EFF-50", "EFF-75", "EFF-100", "SAG-1", "THERM-1"]
    assert r["channels"]["rack_power_kw"]["mean"] == 50.0


# ----------------------------------------------------------------- detect_threshold_violations
def test_violations_known_events_and_per_channel_merge_gap(plan):
    df = make_df(2000)
    df.loc[100:109, "dc_bus_voltage_v"] = 750.0  # below the 760 V illustrative minimum
    df.loc[[1000, 1100, 1200], "sic_temp_c"] = 126.0  # 100 ms apart: one event thanks to the 0.5 s thermal merge gap
    r = A.detect_threshold_violations(df, plan)
    bus = r["channels"]["dc_bus_voltage_v"]["events"]
    assert len(bus) == 1 and (bus[0]["type"], bus[0]["start_s"], bus[0]["n_samples"], bus[0]["peak_value"]) == ("below_min", 0.1, 10, 750.0)
    temp = r["channels"]["sic_temp_c"]["events"]
    assert len(temp) == 1 and temp[0]["n_samples"] == 3 and temp[0]["exceedance"] == 1.0
    assert r["channels"]["rack_power_kw"]["status"] == "within limits"


# ----------------------------------------------------------------- check_power_balance
def test_power_balance_consistent_on_clean_data(plan):
    r = A.check_power_balance(make_df(2000, rack=100, loss=1.6), plan)
    assert r["verdict"] == "CONSISTENT" and r["plausibility_check"]["windows_checked"] == 4


def test_power_balance_flags_gain_error_and_battery_offset(plan):
    gain = make_df(2000, rack=100, loss=1.6)
    gain["grid_power_kw"] *= 0.9  # grid sensor reads 10% low -> output > input
    r = A.check_power_balance(gain, plan)
    assert r["verdict"] == "SUSPECT" and r["plausibility_check"]["windows_flagged"] == 4
    assert r["plausibility_check"]["flagged"][0]["implied_efficiency_pct"] > 100
    offset = make_df(2000, rack=100, loss=1.6)
    offset["battery_power_kw"] = 3.0
    r = A.check_power_balance(offset, plan)
    assert r["verdict"] == "SUSPECT" and r["battery_idle_check"]["pass"] is False
    assert "sensor offset" in r["findings"][0]


def test_power_balance_on_synthetic_runs(plan, run_a, run_b, run_fault):
    assert A.check_power_balance(run_a.df, plan)["verdict"] == "CONSISTENT"
    assert A.check_power_balance(run_b.df, plan)["verdict"] == "CONSISTENT"  # sag windows and light load are skipped
    fault = A.check_power_balance(run_fault.df, plan)
    assert fault["verdict"] == "SUSPECT" and fault["battery_idle_check"]["median_battery_kw_healthy_grid"] == pytest.approx(3.0, abs=0.05)


# ----------------------------------------------------------------- evaluate_efficiency
def test_efficiency_known_value(plan):
    r = A.evaluate_efficiency(make_df(2000, step=1, rack=50, loss=1.0), plan, plan_test(plan, "EFF-25"))
    assert r["verdict"] == "PASS"
    assert r["kpis"][0]["measured"] == pytest.approx(100 * 50 / 51, abs=1e-3)
    assert r["details"]["window_start_s"] == 1.0  # first settle_s ignored
    assert r["details"]["mean_loss_kw"] == 1.0 and r["details"]["measured_load_pct"] == 25.0


def test_efficiency_fail_off_target_and_not_run(plan):
    r = A.evaluate_efficiency(make_df(2000, step=1, rack=80, loss=4.0), plan, plan_test(plan, "EFF-25"))
    assert r["verdict"] == "FAIL" and any("target" in x for x in r["limitations"])
    assert A.evaluate_efficiency(make_df(100, step=3), plan, plan_test(plan, "EFF-25"))["verdict"] == "NOT RUN"


# ----------------------------------------------------------------- analyze_load_bursts
def _burst_df(dip_v=10.0, tau=0.002):
    df = make_df(1000, step=5, rack=100)
    t = df["timestamp_s"].to_numpy()
    df.loc[500:549, "rack_power_kw"] = 180.0  # 50 ms burst
    after = (t >= 0.5) & (t < 0.55)
    df.loc[after, "dc_bus_voltage_v"] = 800 - dip_v * np.exp(-(t[after] - 0.5) / tau)
    return df


def test_burst_known_dip_and_recovery(plan):
    r = A.analyze_load_bursts(_burst_df(), plan, plan_test(plan, "BURST-1"))
    rise, fall = r["details"]["edges"]
    assert (rise["direction"], rise["time_s"], rise["delta_kw"]) == ("rise", 0.5, 80.0)
    assert rise["peak_deviation_v"] == pytest.approx(-10.0) and rise["peak_time_s"] == 0.5
    # 10 V x exp(-t / 2 ms) is inside the ±4 V band from t = 2 ms on.
    assert rise["recovery_time_s"] == pytest.approx(0.002)
    assert fall["direction"] == "fall" and fall["recovery_time_s"] == 0.0
    assert r["verdict"] == "PASS"


def test_burst_fails_on_large_dip_or_slow_recovery(plan):
    test = plan_test(plan, "BURST-1")
    assert {k["name"]: k["pass"] for k in A.analyze_load_bursts(_burst_df(dip_v=20), plan, test)["kpis"]} == {
        "max_abs_bus_deviation_v": False, "worst_recovery_time_s": True}
    slow = A.analyze_load_bursts(_burst_df(dip_v=10, tau=0.01), plan, test)
    assert {k["name"]: k["pass"] for k in slow["kpis"]} == {"max_abs_bus_deviation_v": True, "worst_recovery_time_s": False}


def test_burst_without_edges_is_inconclusive(plan):
    assert A.analyze_load_bursts(make_df(1000, step=5), plan, plan_test(plan, "BURST-1"))["verdict"] == "INCONCLUSIVE"


# ----------------------------------------------------------------- analyze_grid_sag
def _sag_df(pickup_samples=5, dip_to=785.0):
    df = make_df(1000, step=6, rack=150, loss=2.0)
    df.loc[300:499, "grid_voltage_pu"] = 0.5
    df.loc[300:499, "grid_power_kw"] = 45.0
    df.loc[300 + pickup_samples : 499, "battery_power_kw"] = 107.0
    df.loc[300 : 300 + pickup_samples - 1, "dc_bus_voltage_v"] = dip_to
    return df


def test_sag_known_pickup_and_minimum(plan):
    r = A.analyze_grid_sag(_sag_df(), plan, plan_test(plan, "SAG-1"))
    d = r["details"]
    assert (d["sag_start_s"], d["sag_end_s"], d["sag_duration_s"], d["deficit_kw"]) == (0.3, 0.499, 0.2, 107.0)
    k = {x["name"]: x["measured"] for x in r["kpis"]}
    assert k == {"min_bus_voltage_v": 785.0, "battery_pickup_time_s": 0.005}
    assert r["verdict"] == "PASS"


def test_sag_fails_on_slow_pickup_or_deep_dip(plan):
    test = plan_test(plan, "SAG-1")
    assert A.analyze_grid_sag(_sag_df(pickup_samples=30), plan, test)["verdict"] == "FAIL"
    assert A.analyze_grid_sag(_sag_df(dip_to=750.0), plan, test)["verdict"] == "FAIL"
    assert A.analyze_grid_sag(make_df(500, step=6), plan, test)["status"] == "no grid sag found in this step"


# ----------------------------------------------------------------- evaluate_thermal
def _ramp_df(start, slope):
    df = make_df(8000, step=7)
    df["sic_temp_c"] = start + slope * df["timestamp_s"]
    return df


def test_thermal_known_peak_and_slope(plan):
    r = A.evaluate_thermal(_ramp_df(100, 0.5), plan, plan_test(plan, "THERM-1"))
    k = {x["name"]: x["measured"] for x in r["kpis"]}
    assert k["peak_sic_temp_c"] == pytest.approx(104.0, abs=0.01) and k["end_slope_c_per_s"] == pytest.approx(0.5)
    assert r["verdict"] == "PASS"
    hot = A.evaluate_thermal(_ramp_df(120, 1.5), plan, plan_test(plan, "THERM-1"))
    assert hot["verdict"] == "FAIL" and not any(x["pass"] for x in hot["kpis"])


# ----------------------------------------------------------------- process tools
def test_plan_on_rev_a_passes_everything(plan, run_a):
    r = A.run_validation_plan(run_a.df, plan)
    assert r["summary"]["tests_run"] == 7 and r["summary"]["pass"] == 7 and r["summary"]["failed_tests"] == []


def test_plan_on_rev_b_fails_the_expected_tests(plan, run_b):
    r = A.run_validation_plan(run_b.df, plan)
    assert r["summary"]["failed_tests"] == ["EFF-25", "BURST-1", "THERM-1"]
    burst = next(t for t in r["tests"] if t["test_id"] == "BURST-1")
    assert burst["details"]["edge_count"] == 22


def test_plan_filters_and_missing_steps(plan, run_a):
    r = A.run_validation_plan(run_a.df, plan, test_types=["thermal"])
    assert [t["test_id"] for t in r["tests"]] == ["THERM-1"]
    partial = run_a.df[run_a.df.test_step != 6]
    sag = next(t for t in A.run_validation_plan(partial, plan)["tests"] if t["test_id"] == "SAG-1")
    assert sag["verdict"] == "NOT RUN"


def test_sensor_fault_drags_efficiency_down(plan, run_fault):
    failed = A.run_validation_plan(run_fault.df, plan)["summary"]["failed_tests"]
    assert failed == ["EFF-25", "EFF-50", "EFF-75", "EFF-100"]  # caused by the sensor, which check_power_balance flags


def test_compare_b_to_a_finds_regressions_and_new_failures(plan, run_a, run_b):
    r = A.compare_to_baseline(run_b.df, run_a.df, plan)
    assert r["summary"]["new_failures"] == ["EFF-25", "BURST-1", "THERM-1"]
    changes = {(x["test_id"], x["kpi"]): x["change"] for x in r["kpi_deltas"]}
    assert changes[("SAG-1", "battery_pickup_time_s")] == "regression"  # still PASS, but slower than baseline
    assert changes[("SAG-1", "min_bus_voltage_v")] == "no significant change"
    eff = next(x for x in r["kpi_deltas"] if x["test_id"] == "EFF-25")
    assert eff["delta"] == pytest.approx(eff["candidate"] - eff["baseline"], abs=1e-3) and eff["delta"] < -1


def test_compare_identical_runs_has_no_regressions(plan, run_a):
    r = A.compare_to_baseline(run_a.df, run_a.df, plan)
    assert r["summary"]["regressions"] == 0 and r["verdict_changes"] == []


def test_compare_respects_tolerance(plan, run_a, run_b):
    loose = copy.deepcopy(plan)
    loose["regression"]["efficiency_pct_tolerance"] = 5.0
    r = A.compare_to_baseline(run_b.df, run_a.df, loose, test_ids=["EFF-50"])
    assert [x["change"] for x in r["kpi_deltas"]] == ["no significant change"]

import json

import numpy as np
import plotly.io as pio
import pytest

from powerops.numbers_check import collect_numbers, find_unverified_numbers
from powerops.plots import PLOT_TYPES, decimate_min_max, generate_plots
from powerops.tools import TOOL_NAMES, ToolContext, build_tool_schemas, run_tool

from .conftest import make_df


# ----------------------------------------------------------------- generate_plots
@pytest.mark.parametrize("plot_type, plot_id", [("overview", "overview"), ("efficiency_curve", "efficiency_curve"),
                                                ("burst", "burst_BURST-1"), ("grid_sag", "grid_sag_SAG-1"),
                                                ("thermal", "thermal_THERM-1")])
def test_each_plot_type_builds_a_serialisable_figure(plan, run_b, plot_type, plot_id):
    summary, figures = generate_plots(run_b.df, plan, plot_type, is_synthetic=True)
    assert summary["status"] == "ok" and [f["plot_id"] for f in figures] == [plot_id]
    assert figures[0]["figure"].layout.title.text.startswith("SYNTHETIC DATA")
    pio.from_json(figures[0]["figure"].to_json())
    assert all(set(p) == {"plot_id", "description"} for p in summary["plots_created"])  # no figure data for the agent


def test_baseline_is_overlaid_when_given(plan, run_a, run_b):
    for plot_type in ("efficiency_curve", "burst", "thermal"):
        alone = generate_plots(run_b.df, plan, plot_type)[1][0]["figure"]
        both = generate_plots(run_b.df, plan, plot_type, baseline_df=run_a.df, labels=("rev-B", "rev-A"))[1][0]["figure"]
        assert len(both.data) == len(alone.data) + 1
        assert "rev-A" in {tr.name for tr in both.data}


def test_plot_without_data_explains_why(plan):
    summary, figs = generate_plots(make_df(500, step=1), plan, "burst")
    assert figs == [] and "not run" in summary["status"]
    with pytest.raises(ValueError):
        generate_plots(make_df(10), plan, "pie")


def test_min_max_decimation_keeps_spikes():
    x = np.arange(100_000, dtype=float)
    y = np.zeros(100_000)
    y[54_321], y[77_777] = 99.0, -42.0
    xd, yd = decimate_min_max(x, y, max_points=1000)
    assert len(xd) <= 1000 and 99.0 in yd and -42.0 in yd and np.all(np.diff(xd) > 0)


# ----------------------------------------------------------------- tool registry
def test_schemas_list_tools_and_plan_test_ids(plan):
    schemas = build_tool_schemas(plan)
    assert [s["function"]["name"] for s in schemas] == TOOL_NAMES
    run_plan = next(s for s in schemas if s["function"]["name"] == "run_validation_plan")
    assert run_plan["function"]["parameters"]["properties"]["test_ids"]["items"]["enum"] == [t["id"] for t in plan["tests"]]
    plots = next(s for s in schemas if s["function"]["name"] == "generate_plots")
    assert plots["function"]["parameters"]["properties"]["plot_type"]["enum"] == PLOT_TYPES
    assert all(s["function"]["parameters"]["additionalProperties"] is False for s in schemas)


def test_run_tool_executes_with_baseline_and_collects_figures(plan, run_a, run_b):
    ctx = ToolContext(run_b, plan, baseline=run_a)
    rec = run_tool("compare_to_baseline", '{"test_ids": ["EFF-25"]}', ctx)
    assert rec.ok and rec.result["baseline_run"] == run_a.label and rec.result["candidate_run"] == run_b.label
    assert run_tool("generate_plots", {"plot_type": "thermal", "test_id": None}, ctx).ok  # nulls dropped
    assert len(ctx.figures) == 1 and len(ctx.figures[0]["figure"].data) == 2


@pytest.mark.parametrize(
    "name, args, message",
    [
        ("flash_firmware", {}, "Unknown tool"),
        ("summarize_run", "{not json", "not valid JSON"),
        ("run_validation_plan", {"test_ids": ["EFF-999"]}, "Invalid arguments"),  # only plan test ids are allowed
        ("run_validation_plan", {"acceptance": {"min_efficiency_pct": 50}}, "Invalid arguments"),  # cannot loosen criteria
        ("generate_plots", {}, "Invalid arguments"),
        ("compare_to_baseline", {}, "No baseline run is loaded"),
    ],
)
def test_run_tool_rejects_bad_calls_without_raising(plan, run_b, name, args, message):
    rec = run_tool(name, args, ToolContext(run_b, plan))
    assert not rec.ok and message in rec.result["error"]


def test_run_tool_turns_exceptions_into_errors(plan, run_b):
    broken = {**plan, "power_balance": {}}
    rec = run_tool("check_power_balance", {}, ToolContext(run_b, broken))
    assert not rec.ok and "check_power_balance failed" in rec.result["error"]


# ----------------------------------------------------------------- numbers check
def test_collect_numbers_walks_nested_structures():
    assert sorted(collect_numbers({"a": 1, "b": [2.5, {"c": "3 V"}], "d": True, "e": None})) == [1.0, 2.5, 3.0]


def test_numbers_matching_rounding_sign_and_units():
    ref = {"dev": -18.668, "recovery_time_s": 0.011, "count": 3, "limit": 16.0}
    text = "Bus dipped 18.67 V (−18.668 V exactly), recovered in 11 ms; 3 failures; illustrative limit 16 V."
    assert find_unverified_numbers(text, ref) == []


def test_fabricated_numbers_are_flagged():
    assert find_unverified_numbers("Peak 127.17 °C; MTBF 50000 h; efficiency 99.1%.", {"peak": 127.17}) == ["50000", "99.1"]

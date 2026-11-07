"""The tool registry: what the LLM is allowed to call, and how we run it.

build_tool_schemas(plan) describes each tool and its arguments as JSON Schema
and is sent to the OpenAI API. Test IDs are listed from the plan, so the model
can only ask for tests that exist. The model replies with a tool name and JSON
arguments; run_tool() validates them and calls the matching Python function.

The model can NOT pass data, change limits or acceptance criteria, or call
anything not listed here (no hardware, files or network).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import jsonschema

from . import analysis
from .config import TEST_TYPES
from .data_io import MEASUREMENT_COLUMNS, Dataset
from .plots import PLOT_TYPES, generate_plots

TOOL_NAMES = ["summarize_run", "run_validation_plan", "compare_to_baseline", "check_power_balance",
              "detect_threshold_violations", "generate_plots"]


def build_tool_schemas(plan: dict) -> list[dict]:
    test_ids = [t["id"] for t in plan["tests"]]
    test_id_list = {"type": "array", "items": {"type": "string", "enum": test_ids}, "minItems": 1}

    def fn(name, description, properties=None, required=None):
        params = {"type": "object", "properties": properties or {}, "additionalProperties": False}
        if required:
            params["required"] = required
        return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}

    return [
        fn("summarize_run", "Run duration and sample rate, min/max/mean per channel, which test steps are in the log and "
           "which plan tests have no data."),
        fn("run_validation_plan", "Run the validation plan's tests and return a PASS/FAIL scorecard with measured KPIs vs "
           "illustrative acceptance criteria. Test types: efficiency, load_burst (800 V bus deviation/recovery on GPU load "
           "edges), grid_sag (ride-through and battery pickup), thermal (SiC peak temperature, stabilisation). "
           "Use for 'generate the validation report' or questions about any specific test.",
           {"test_ids": {**test_id_list, "description": "Only these tests. Omit for all."},
            "test_types": {"type": "array", "items": {"type": "string", "enum": sorted(TEST_TYPES)}, "minItems": 1,
                           "description": "Only tests of these types. Omit for all."}}),
        fn("compare_to_baseline", "Run the same plan on the loaded baseline run and the candidate run; return per-KPI "
           "deltas, regressions beyond tolerance, and PASS/FAIL verdict changes. Only works if a baseline run is loaded.",
           {"test_ids": {**test_id_list, "description": "Only these tests. Omit for all."}}),
        fn("check_power_balance", "Data-integrity check of the power sensors (battery-port idle offset; implied efficiency "
           "plausibility). Run this before trusting efficiency results."),
        fn("detect_threshold_violations", "Every excursion beyond the illustrative channel limits (start/end, duration, peak).",
           {"channels": {"type": "array", "items": {"type": "string", "enum": MEASUREMENT_COLUMNS}, "minItems": 1,
                         "description": "Channels to check. Omit for all channels with limits."}}),
        fn("generate_plots", "Create a plot for the engineer to see in the UI: overview (all channels), efficiency_curve, "
           "burst (worst GPU load edge), grid_sag, thermal. Baseline is overlaid automatically when loaded.",
           {"plot_type": {"type": "string", "enum": PLOT_TYPES},
            "test_id": {"type": "string", "enum": test_ids, "description": "Which test to plot (burst/grid_sag/thermal)."}},
           required=["plot_type"]),
    ]


@dataclass
class ToolContext:
    """Everything a tool needs. Created by the server, never by the LLM."""

    dataset: Dataset
    plan: dict
    baseline: Dataset | None = None
    figures: list[dict] = field(default_factory=list)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result: dict

    @property
    def ok(self) -> bool:
        return "error" not in self.result


def _dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    df, plan = ctx.dataset.df, ctx.plan
    if name == "summarize_run":
        return analysis.summarize_run(df, plan)
    if name == "run_validation_plan":
        result = analysis.run_validation_plan(df, plan, **args)
        result["run"] = ctx.dataset.label
        return result
    if name == "compare_to_baseline":
        if ctx.baseline is None:
            return {"error": "No baseline run is loaded. Load a baseline (e.g. the previous revision) to compare against."}
        result = analysis.compare_to_baseline(df, ctx.baseline.df, plan, **args)
        result["candidate_run"], result["baseline_run"] = ctx.dataset.label, ctx.baseline.label
        return result
    if name == "check_power_balance":
        return analysis.check_power_balance(df, plan)
    if name == "detect_threshold_violations":
        return analysis.detect_threshold_violations(df, plan, **args)
    if name == "generate_plots":
        baseline_df = ctx.baseline.df if ctx.baseline is not None else None
        labels = (ctx.dataset.short_label, ctx.baseline.short_label if ctx.baseline else "baseline")
        summary, figures = generate_plots(df, plan, is_synthetic=ctx.dataset.is_synthetic,
                                          baseline_df=baseline_df, labels=labels, **args)
        ctx.figures.extend(figures)
        return summary
    raise KeyError(name)  # unreachable: names are checked before dispatch


def run_tool(name: str, arguments: dict | str | None, ctx: ToolContext) -> ToolCallRecord:
    """Validate and execute one tool call. Never raises: problems become {"error": ...}."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolCallRecord(name, {"_raw": arguments}, {"error": f"Arguments were not valid JSON: {exc}"})
    if not isinstance(arguments, dict):
        return ToolCallRecord(name, {"_raw": arguments}, {"error": "Arguments must be a JSON object."})
    arguments = {k: v for k, v in arguments.items() if v is not None}

    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in build_tool_schemas(ctx.plan)}
    if name not in schemas:
        return ToolCallRecord(name, arguments, {"error": f"Unknown tool '{name}'. Available tools: {TOOL_NAMES}"})
    try:
        jsonschema.validate(arguments, schemas[name])
    except jsonschema.ValidationError as exc:
        return ToolCallRecord(name, arguments, {"error": f"Invalid arguments for {name}: {exc.message}"})
    try:
        result = _dispatch(name, arguments, ctx)
    except Exception as exc:  # a bug or odd data should not crash the whole conversation
        result = {"error": f"{name} failed: {type(exc).__name__}: {exc}"}
    return ToolCallRecord(name, arguments, result)

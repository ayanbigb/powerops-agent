"""Load the validation plan (tests + acceptance criteria + channel limits) from JSON.

The example file (config/validation_plan.example.json) contains ILLUSTRATIVE
numbers only. They are not real product specifications. Point
POWEROPS_PLAN_PATH at your own file to use a different plan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN_PATH = PROJECT_ROOT / "config" / "validation_plan.example.json"

MEASUREMENT_CHANNELS = [
    "grid_voltage_pu",
    "grid_power_kw",
    "battery_power_kw",
    "rack_power_kw",
    "dc_bus_voltage_v",
    "sic_temp_c",
]

# Every test type and the acceptance-criteria keys it needs (all positive numbers).
TEST_TYPES = {
    "efficiency": ["target_load_pct", "min_efficiency_pct"],
    "load_burst": ["min_burst_kw", "edge_window_s", "max_deviation_v", "recovery_band_v", "max_recovery_s"],
    "grid_sag": ["sag_threshold_pu", "pickup_fraction", "min_bus_voltage_v", "max_pickup_s"],
    "thermal": ["max_temp_c", "max_end_slope_c_per_s"],
}

REQUIRED_SECTIONS = {
    "violation_detection": ["merge_gap_s"],
    "steady_state": ["settle_s"],
    "power_balance": ["window_s", "min_power_fraction", "min_plausible_efficiency_pct",
                      "max_plausible_efficiency_pct", "battery_idle_tolerance_kw", "healthy_grid_min_pu"],
    "regression": ["efficiency_pct_tolerance", "voltage_v_tolerance", "time_s_tolerance",
                   "temp_c_tolerance", "slope_c_per_s_tolerance"],
}


class ConfigError(ValueError):
    """Raised when the plan file is missing or invalid."""


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_plan(plan: dict) -> None:
    """Raise ConfigError with a readable message if the plan is unusable."""
    problems: list[str] = []

    if not _is_number(plan.get("rated_power_kw")) or plan["rated_power_kw"] <= 0:
        problems.append("rated_power_kw must be a positive number")

    channels = plan.get("channels")
    if not isinstance(channels, dict):
        problems.append("'channels' section is missing or not an object")
        channels = {}
    for name, spec in channels.items():
        if name not in MEASUREMENT_CHANNELS:
            problems.append(f"unknown channel '{name}' (expected one of {MEASUREMENT_CHANNELS})")
            continue
        if not isinstance(spec, dict):
            problems.append(f"channels.{name} must be an object")
            continue
        for key in ("min", "max", "nominal", "merge_gap_s"):
            if key in spec and not _is_number(spec[key]):
                problems.append(f"channels.{name}.{key} must be a number")
        if _is_number(spec.get("min")) and _is_number(spec.get("max")) and spec["min"] >= spec["max"]:
            problems.append(f"channels.{name}: min must be less than max")

    for section, keys in REQUIRED_SECTIONS.items():
        block = plan.get(section)
        if not isinstance(block, dict):
            problems.append(f"'{section}' section is missing")
            continue
        for key in keys:
            value = block.get(key)
            if not _is_number(value) or value <= 0:
                problems.append(f"{section}.{key} must be a positive number (got {value!r})")

    tests = plan.get("tests")
    if not isinstance(tests, list) or not tests:
        problems.append("'tests' must be a non-empty list")
        tests = []
    seen: set[str] = set()
    for i, test in enumerate(tests):
        tid = test.get("id") if isinstance(test, dict) else None
        label = tid or f"tests[{i}]"
        if not isinstance(tid, str) or not tid:
            problems.append(f"{label}: every test needs a string 'id'")
        elif tid in seen:
            problems.append(f"duplicate test id '{tid}'")
        else:
            seen.add(tid)
        ttype = test.get("type") if isinstance(test, dict) else None
        if ttype not in TEST_TYPES:
            problems.append(f"{label}: unknown type {ttype!r} (choose from {sorted(TEST_TYPES)})")
            continue
        if not isinstance(test.get("step"), int) or isinstance(test.get("step"), bool) or test["step"] < 0:
            problems.append(f"{label}: 'step' must be a non-negative integer (the test_step value in the log)")
        for key in TEST_TYPES[ttype]:
            if not _is_number(test.get(key)) or test[key] <= 0:
                problems.append(f"{label}.{key} must be a positive number (got {test.get(key)!r})")

    if problems:
        raise ConfigError("Invalid validation plan: " + "; ".join(problems))


def load_plan(path: str | Path | None = None) -> dict:
    """Read, validate and return the validation plan as a plain dict."""
    path = Path(path or os.getenv("POWEROPS_PLAN_PATH") or DEFAULT_PLAN_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path  # relative paths are relative to the project folder
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Plan file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Plan file {path} is not valid JSON: {exc}") from exc
    if not isinstance(plan, dict):
        raise ConfigError(f"Plan file {path} must contain a JSON object")
    validate_plan(plan)
    plan.setdefault("profile_name", path.stem)
    try:
        plan["_source_path"] = str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:  # file lives outside the project
        plan["_source_path"] = str(path)
    return plan

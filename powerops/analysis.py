"""Deterministic analysis tools for SST validation runs. The ONLY source of measured values.

Building blocks (each tested on its own):
    summarize_run               run timing, per-channel statistics, test steps found
    detect_threshold_violations excursions beyond illustrative channel limits
    check_power_balance         data-integrity checks on the power sensors
    evaluate_efficiency         one efficiency test (steady-state energy ratio)
    analyze_load_bursts         800 V bus deviation and recovery at every GPU load edge
    analyze_grid_sag            ride-through: bus minimum and battery-port pickup time
    evaluate_thermal            SiC peak temperature and whether it stabilised

Process tools (what the agent uses to automate a validation report):
    run_validation_plan         run every test in the plan -> PASS/FAIL scorecard
    compare_to_baseline         same plan on two runs -> per-KPI deltas and regressions

Every result is a JSON-serialisable dict with measured values, "method" and "limitations".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data_io import MEASUREMENT_COLUMNS, STEP_COLUMN, TIME_COLUMN, UNITS

TIME_DECIMALS = 4
VALUE_DECIMALS = 4


# --------------------------------------------------------------------------- helpers
def _r(x, nd: int = VALUE_DECIMALS):
    """Round a number for reporting; NaN/None become None (JSON null)."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def _t(x):
    return _r(x, TIME_DECIMALS)


def median_dt(df: pd.DataFrame) -> float:
    return float(np.median(np.diff(df[TIME_COLUMN].to_numpy())))


def seconds_to_samples(seconds: float, dt: float, minimum: int = 1) -> int:
    return max(minimum, int(round(seconds / dt)))


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (first_index, last_index) for every consecutive run of True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist()))


def merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Join runs separated by at most `max_gap` samples."""
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def step_rows(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Rows of one test step, re-indexed from 0."""
    return df[df[STEP_COLUMN] == step].reset_index(drop=True)


def _plan_profile(plan: dict) -> str:
    return f"{plan.get('profile_name', 'unnamed plan')} (illustrative values, not a product specification)"


def _kpi(name: str, measured, limit: float, comparison: str, unit: str, tolerance_key: str) -> dict:
    """One acceptance check. comparison 'min' = measured must be >= limit; 'max' = must be <= limit."""
    if measured is None:
        passed = False
    else:
        passed = measured >= limit if comparison == "min" else measured <= limit
    return {"name": name, "measured": measured, "limit": limit, "comparison": comparison, "unit": unit,
            "pass": bool(passed), "tolerance_key": tolerance_key}


def _test_result(test: dict, status: str, kpis: list[dict], details: dict, method: str, limitations: list[str]) -> dict:
    if status != "ok":
        verdict = "NOT RUN" if status.startswith("not run") else "INCONCLUSIVE"
    else:
        verdict = "PASS" if all(k["pass"] for k in kpis) else "FAIL"
    return {"test_id": test["id"], "type": test["type"], "step": test["step"], "description": test.get("description", ""),
            "status": status, "verdict": verdict, "kpis": kpis, "details": details,
            "method": method, "limitations": limitations}


def _not_run(test: dict, method: str) -> dict:
    return _test_result(test, f"not run: test_step {test['step']} not found in this log", [], {}, method, [])


def _port_powers(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Total power into and out of the SST. Battery: + = discharging (input), - = charging (output)."""
    batt = rows["battery_power_kw"]
    p_in = rows["grid_power_kw"] + batt.clip(lower=0)
    p_out = rows["rack_power_kw"] + (-batt).clip(lower=0)
    return p_in, p_out


def _detect_edges(p: pd.Series, w: int, min_step: float) -> list[dict]:
    """Abrupt changes in `p`: compare the median of w samples before vs from each sample.
    Edge time = first sample crossing halfway between the old and new level."""
    if p.notna().sum() < 2 * w:
        return []
    min_periods = max(2, w // 2)
    before = p.rolling(w, min_periods=min_periods).median().shift(1).to_numpy()
    after = p.iloc[::-1].rolling(w, min_periods=min_periods).median().iloc[::-1].to_numpy()
    delta = after - before
    candidate = np.abs(np.nan_to_num(delta)) >= min_step
    values = p.to_numpy(dtype=float)
    edges: list[dict] = []
    for a, b in find_runs(candidate):
        i_best = a + int(np.argmax(np.abs(delta[a : b + 1])))
        pre, post = before[i_best], after[i_best]
        mid = (pre + post) / 2.0
        lo, hi = max(0, a - w), min(len(values), b + w + 1)
        seg = values[lo:hi]
        crossed = np.flatnonzero(seg >= mid) if post > pre else np.flatnonzero(seg <= mid)
        k = lo + int(crossed[0]) if crossed.size else i_best
        if edges and k - edges[-1]["index"] < w:
            if abs(post - pre) <= abs(edges[-1]["delta"]):
                continue
            edges.pop()
        edges.append({"index": int(k), "pre": float(pre), "post": float(post), "delta": float(post - pre)})
    return edges


# --------------------------------------------------------------------------- summarize_run
def summarize_run(df: pd.DataFrame, plan: dict) -> dict:
    t = df[TIME_COLUMN].to_numpy()
    dt = np.diff(t)
    med = float(np.median(dt))
    max_gap = float(dt.max())
    channels = {}
    for col in MEASUREMENT_COLUMNS:
        s = df[col]
        n_valid = int(s.notna().sum())
        entry = {"unit": UNITS[col], "valid_samples": n_valid, "missing_samples": int(len(s) - n_valid),
                 "missing_percent": _r(100.0 * (len(s) - n_valid) / len(s), 2)}
        if n_valid == 0:
            entry["status"] = "no valid data"
        else:
            i_min, i_max = s.idxmin(), s.idxmax()
            entry.update(status="ok", min=_r(s[i_min]), min_time_s=_t(t[i_min]), max=_r(s[i_max]),
                         max_time_s=_t(t[i_max]), mean=_r(s.mean()), std=_r(s.std(ddof=0)))
        channels[col] = entry

    tests_by_step: dict[int, list[str]] = {}
    for test in plan.get("tests", []):
        tests_by_step.setdefault(test["step"], []).append(test["id"])
    steps = []
    for step, rows in df.dropna(subset=[STEP_COLUMN]).groupby(STEP_COLUMN):
        steps.append({"step": int(step), "start_s": _t(rows[TIME_COLUMN].min()), "end_s": _t(rows[TIME_COLUMN].max()),
                      "n_samples": int(len(rows)), "mean_rack_power_kw": _r(rows["rack_power_kw"].mean(), 2),
                      "plan_tests": tests_by_step.get(int(step), [])})
    found = {s["step"] for s in steps}
    rack = df["rack_power_kw"]
    ok = rack.notna().to_numpy()
    return {
        "tool": "summarize_run",
        "run": {"n_samples": int(len(df)), "start_s": _t(t[0]), "end_s": _t(t[-1]), "duration_s": _t(t[-1] - t[0]),
                "median_sample_interval_s": _r(med, 6), "approx_sample_rate_hz": _r(1.0 / med, 2),
                "max_gap_s": _r(max_gap, 6), "irregular_sampling": bool(max_gap > 1.5 * med)},
        "channels": channels,
        "test_steps": steps,
        "plan_tests_without_data": [t_["id"] for t_ in plan.get("tests", []) if t_["step"] not in found],
        "rack_energy_kwh": _r(np.trapezoid(rack.to_numpy()[ok], t[ok]) / 3600.0, 4) if ok.sum() >= 2 else None,
        "method": "pandas statistics over valid samples; test steps grouped by the test_step column; "
                  "rack energy = trapezoidal integral of rack_power_kw over time.",
        "limitations": ["Extremes are single samples and may include noise.", "Missing samples are skipped, not interpolated."],
    }


# --------------------------------------------------------------------------- detect_threshold_violations
def detect_threshold_violations(df: pd.DataFrame, plan: dict, channels: list[str] | None = None,
                                max_events_per_channel: int = 20) -> dict:
    channels = channels or [c for c in MEASUREMENT_COLUMNS if c in plan.get("channels", {})]
    t = df[TIME_COLUMN].to_numpy()
    steps = df[STEP_COLUMN].to_numpy()
    dt = median_dt(df)
    merge_gap_s = plan["violation_detection"]["merge_gap_s"]
    out, total = {}, 0
    for col in channels:
        lim = plan.get("channels", {}).get(col, {})
        # Slow signals (temperature) may set their own, longer merge gap so noise at the limit is one event.
        merge_gap = int(round(lim.get("merge_gap_s", merge_gap_s) / dt))
        values = df[col].to_numpy(dtype=float)
        entry: dict = {"unit": UNITS[col], "limits_used": {k: lim[k] for k in ("min", "max") if k in lim}}
        if not entry["limits_used"]:
            out[col] = {**entry, "status": "no limits configured", "event_count": 0, "events": []}
            continue
        if np.isnan(values).all():
            out[col] = {**entry, "status": "no valid data", "event_count": 0, "events": []}
            continue
        events = []
        checks = [("above_max", lim["max"], values > lim["max"])] if "max" in lim else []
        checks += [("below_min", lim["min"], values < lim["min"])] if "min" in lim else []
        for kind, limit, mask in checks:
            for a, b in merge_runs(find_runs(mask), max_gap=merge_gap):
                seg = values[a : b + 1]
                k = int(np.nanargmax(seg)) if kind == "above_max" else int(np.nanargmin(seg))
                events.append({"type": kind, "limit": limit, "start_s": _t(t[a]), "end_s": _t(t[b]),
                               "test_step": None if np.isnan(steps[a]) else int(steps[a]),
                               "n_samples": int(mask[a : b + 1].sum()), "duration_s": _t((b - a + 1) * dt),
                               "peak_value": _r(seg[k]), "peak_time_s": _t(t[a + k]),
                               "exceedance": _r(seg[k] - limit if kind == "above_max" else limit - seg[k])})
        events.sort(key=lambda e: e["start_s"])
        total += len(events)
        out[col] = {**entry, "status": "violations found" if events else "within limits", "event_count": len(events),
                    "worst_exceedance": _r(max((e["exceedance"] for e in events), default=None)),
                    "events": events[:max_events_per_channel]}
        if len(events) > max_events_per_channel:
            out[col]["events_truncated"] = True
    return {
        "tool": "detect_threshold_violations",
        "plan_profile": _plan_profile(plan),
        "parameters": {"merge_gap_s": merge_gap_s},
        "total_events": total,
        "channels_with_violations": [c for c, e in out.items() if e.get("event_count")],
        "channels": out,
        "method": "Each sample compared with the channel's illustrative min/max (strict inequality); consecutive "
                  f"violating samples form one event; events separated by <= {merge_gap_s} s (or the channel's own "
                  "merge_gap_s) are merged.",
        "limitations": ["Limits are illustrative, not a product specification.",
                        "Time resolution is one sample; missing samples never count as violations."],
    }


# --------------------------------------------------------------------------- check_power_balance
def check_power_balance(df: pd.DataFrame, plan: dict) -> dict:
    """Are the power sensors telling a physically consistent story?

    1) Battery-idle check: while the grid is healthy the battery port should carry ~0 kW.
       A steady non-zero reading points to a sensor offset.
    2) Plausibility check: in windows of window_s, implied efficiency = energy out / energy in
       must lie in a plausible band. Outside it (or above 100 %) the power sensors disagree.
    """
    cfg = plan["power_balance"]
    rated = plan["rated_power_kw"]
    dt = median_dt(df)
    w = seconds_to_samples(cfg["window_s"], dt, minimum=5)
    t = df[TIME_COLUMN].to_numpy()
    p_in, p_out = _port_powers(df)
    valid = (p_in.notna() & p_out.notna()).to_numpy()
    pin, pout = p_in.to_numpy(), p_out.to_numpy()
    steps = df[STEP_COLUMN].to_numpy()
    # Around grid events the bus capacitors briefly supply or absorb energy, so the energy ratio is not a
    # sensor check there. Skip windows within 100 ms of any unhealthy-grid sample.
    unhealthy = (df["grid_voltage_pu"] < cfg["healthy_grid_min_pu"]).astype(float)
    near_event = unhealthy.rolling(2 * seconds_to_samples(0.1, dt) + 1, center=True, min_periods=1).max().to_numpy() > 0

    windows, flagged, low_power, grid_event = 0, [], 0, 0
    for a in range(0, len(df) - w + 1, w):
        sl = slice(a, a + w)
        ok = valid[sl]
        if ok.sum() < w // 2:
            continue
        if near_event[sl].any():
            grid_event += 1
            continue
        mean_in = float(np.mean(pin[sl][ok]))
        if mean_in < cfg["min_power_fraction"] * rated:
            low_power += 1
            continue
        windows += 1
        eff = 100.0 * float(np.sum(pout[sl][ok])) / float(np.sum(pin[sl][ok]))
        if not cfg["min_plausible_efficiency_pct"] <= eff <= cfg["max_plausible_efficiency_pct"]:
            flagged.append({"start_s": _t(t[a]), "end_s": _t(t[a + w - 1]),
                            "test_step": None if np.isnan(steps[a]) else int(steps[a]),
                            "implied_efficiency_pct": _r(eff, 3), "mean_input_kw": _r(mean_in, 2),
                            "mean_output_kw": _r(float(np.mean(pout[sl][ok])), 2)})

    healthy = (df["grid_voltage_pu"] >= cfg["healthy_grid_min_pu"]).to_numpy()
    idle_batt = df.loc[healthy, "battery_power_kw"].dropna()
    idle_median = _r(idle_batt.median(), 3) if len(idle_batt) else None
    idle_ok = idle_median is not None and abs(idle_median) <= cfg["battery_idle_tolerance_kw"]

    findings = []
    if idle_median is not None and not idle_ok:
        findings.append(f"Battery port reads a median {idle_median} kW while the grid is healthy "
                        f"(tolerance ±{cfg['battery_idle_tolerance_kw']} kW): likely a battery power-sensor offset. "
                        "Efficiency results that include battery power are unreliable.")
    if flagged:
        findings.append(f"{len(flagged)} window(s) have an implied efficiency outside "
                        f"{cfg['min_plausible_efficiency_pct']}-{cfg['max_plausible_efficiency_pct']} %: "
                        "power sensors disagree (check calibration, offsets, or sign conventions).")
    return {
        "tool": "check_power_balance",
        "verdict": "SUSPECT" if findings else "CONSISTENT",
        "findings": findings,
        "battery_idle_check": {"median_battery_kw_healthy_grid": idle_median,
                               "tolerance_kw": cfg["battery_idle_tolerance_kw"], "pass": bool(idle_ok)},
        "plausibility_check": {"window_s": cfg["window_s"], "windows_checked": windows,
                               "windows_skipped_low_power": low_power, "windows_skipped_grid_event": grid_event,
                               "windows_flagged": len(flagged),
                               "flagged": flagged[:20]},
        "method": "Battery idle: median battery_power_kw over samples with grid_voltage_pu >= "
                  f"{cfg['healthy_grid_min_pu']}. Plausibility: implied efficiency = sum(output power) / sum(input power) "
                  f"per {cfg['window_s']} s window (input = grid + battery discharge; output = rack + battery charge); "
                  f"windows below {cfg['min_power_fraction'] * 100:.0f}% of rated input power, or within 100 ms of a "
                  "grid event, are skipped.",
        "limitations": ["Can show that sensors disagree, not which one is wrong.",
                        "Energy stored in the DC-bus capacitors makes very short windows unreliable."],
    }


# --------------------------------------------------------------------------- evaluate_efficiency
def evaluate_efficiency(df: pd.DataFrame, plan: dict, test: dict) -> dict:
    settle = plan["steady_state"]["settle_s"]
    method = (f"Steady-state window = the test step minus its first {settle} s. Efficiency = sum(output power) / "
              "sum(input power) over the window (energy ratio), input = grid + battery discharge, "
              "output = rack + battery charge.")
    rows = step_rows(df, test["step"])
    if rows.empty:
        return _not_run(test, method)
    steady = rows[rows[TIME_COLUMN] >= rows[TIME_COLUMN].min() + settle]
    p_in, p_out = _port_powers(steady)
    ok = p_in.notna() & p_out.notna()
    limitations = ["Assumes power sensors are calibrated; run check_power_balance to confirm.",
                   "Accuracy is limited by power-measurement uncertainty, which this log does not state."]
    if ok.sum() < 10:
        return _test_result(test, "insufficient data in steady-state window", [], {}, method, limitations)
    eff = 100.0 * float(p_out[ok].sum()) / float(p_in[ok].sum())
    load_pct = 100.0 * float(steady["rack_power_kw"].mean()) / plan["rated_power_kw"]
    details = {"window_start_s": _t(steady[TIME_COLUMN].min()), "window_end_s": _t(steady[TIME_COLUMN].max()),
               "mean_input_kw": _r(p_in[ok].mean(), 3), "mean_output_kw": _r(p_out[ok].mean(), 3),
               "mean_loss_kw": _r((p_in[ok] - p_out[ok]).mean(), 3), "measured_load_pct": _r(load_pct, 2),
               "target_load_pct": test["target_load_pct"]}
    if abs(load_pct - test["target_load_pct"]) > 5:
        limitations.append(f"Measured load {load_pct:.1f}% is more than 5 points from the {test['target_load_pct']}% target.")
    kpis = [_kpi("efficiency_pct", _r(eff, 3), test["min_efficiency_pct"], "min", "%", "efficiency_pct_tolerance")]
    return _test_result(test, "ok", kpis, details, method, limitations)


# --------------------------------------------------------------------------- analyze_load_bursts
def analyze_load_bursts(df: pd.DataFrame, plan: dict, test: dict) -> dict:
    band = test["recovery_band_v"]
    method = (f"Load edges = abrupt rack_power_kw changes >= {test['min_burst_kw']} kW (median of {test['edge_window_s']} s "
              "before vs after). Per edge: baseline = median bus voltage over 10 ms before; response window runs to the "
              "next edge (max 100 ms); peak deviation = sample furthest from baseline; recovery = time until the bus "
              f"stays within ±{band} V of the window's final value (median of its last 10 ms).")
    rows = step_rows(df, test["step"])
    if rows.empty:
        return _not_run(test, method)
    limitations = ["Peak deviation uses raw samples (includes noise); peaks between samples are not captured.",
                   "Recovery is judged against the final value inside each window, not the nominal 800 V."]
    t = rows[TIME_COLUMN].to_numpy()
    v = rows["dc_bus_voltage_v"].to_numpy(dtype=float)
    dt = median_dt(rows)
    w = seconds_to_samples(test["edge_window_s"], dt, minimum=3)
    edges = _detect_edges(rows["rack_power_kw"], w, test["min_burst_kw"])
    if not edges:
        return _test_result(test, "no load edges found in this step", [], {"edge_count": 0}, method, limitations)
    n_base, n_final, n_max = (seconds_to_samples(s, dt, 3) for s in (0.01, 0.01, 0.1))
    results = []
    for i, e in enumerate(edges):
        k = e["index"]
        prev_k = edges[i - 1]["index"] if i else 0
        next_k = edges[i + 1]["index"] if i + 1 < len(edges) else len(v)
        base = v[max(prev_k, k - n_base) : k]
        win_end = min(next_k, k + n_max)
        win, tw = v[k:win_end], t[k:win_end]
        entry = {"edge_number": i + 1, "time_s": _t(t[k]), "direction": "rise" if e["delta"] > 0 else "fall",
                 "delta_kw": _r(e["delta"], 2)}
        if np.isfinite(base).sum() < 3 or np.isfinite(win[-n_final:]).sum() < 3 or len(win) < 2 * n_final:
            results.append({**entry, "status": "insufficient data"})
            continue
        baseline = float(np.nanmedian(base))
        final = float(np.nanmedian(win[-n_final:]))
        j = int(np.nanargmax(np.abs(win - baseline)))
        outside = np.flatnonzero(~(np.abs(win - final) <= band))
        if outside.size == 0:
            recovery = 0.0
        elif outside[-1] >= len(win) - n_final:
            recovery = None
        else:
            recovery = float(tw[outside[-1] + 1] - t[k])
        results.append({**entry, "status": "ok", "baseline_v": _r(baseline, 3), "peak_deviation_v": _r(win[j] - baseline, 3),
                        "peak_time_s": _t(tw[j]), "final_v": _r(final, 3), "recovered": recovery is not None,
                        "recovery_time_s": _t(recovery) if recovery is not None else None})
    good = [r for r in results if r["status"] == "ok"]
    if not good:
        return _test_result(test, "insufficient data around every edge", [], {"edges": results}, method, limitations)
    worst_dev = max(good, key=lambda r: abs(r["peak_deviation_v"]))
    unrecovered = [r["edge_number"] for r in good if not r["recovered"]]
    worst_rec = None if unrecovered else max(r["recovery_time_s"] for r in good)
    kpis = [_kpi("max_abs_bus_deviation_v", _r(abs(worst_dev["peak_deviation_v"]), 3), test["max_deviation_v"], "max", "V",
                 "voltage_v_tolerance"),
            _kpi("worst_recovery_time_s", worst_rec, test["max_recovery_s"], "max", "s", "time_s_tolerance")]
    details = {"edge_count": len(results), "worst_deviation_edge": worst_dev["edge_number"],
               "worst_deviation_time_s": worst_dev["time_s"], "edges_not_recovered": unrecovered, "edges": results[:30]}
    return _test_result(test, "ok", kpis, details, method, limitations)


# --------------------------------------------------------------------------- analyze_grid_sag
def analyze_grid_sag(df: pd.DataFrame, plan: dict, test: dict) -> dict:
    method = (f"Sag = samples with grid_voltage_pu < {test['sag_threshold_pu']}. Deficit = median grid power in the 50 ms "
              "before the sag minus the median grid power during the sag. Battery pickup time = time from sag start "
              f"until battery_power_kw first reaches {test['pickup_fraction']:.0%} of that deficit. Bus minimum is taken "
              "from sag start until 100 ms after it ends.")
    rows = step_rows(df, test["step"])
    if rows.empty:
        return _not_run(test, method)
    limitations = ["Only the first sag in the step is analysed.",
                   "Pickup time resolution is one sample; the deficit estimate assumes grid power is steady before the sag."]
    t = rows[TIME_COLUMN].to_numpy()
    dt = median_dt(rows)
    pu = rows["grid_voltage_pu"].to_numpy(dtype=float)
    sags = merge_runs(find_runs(pu < test["sag_threshold_pu"]), max_gap=seconds_to_samples(0.005, dt))
    if not sags:
        return _test_result(test, "no grid sag found in this step", [], {}, method, limitations)
    a, b = sags[0]
    grid = rows["grid_power_kw"].to_numpy(dtype=float)
    batt = rows["battery_power_kw"].to_numpy(dtype=float)
    bus = rows["dc_bus_voltage_v"].to_numpy(dtype=float)
    pre_grid = float(np.nanmedian(grid[max(0, a - seconds_to_samples(0.05, dt)) : a]))
    during = grid[min(b, a + seconds_to_samples(0.02, dt)) : b + 1]
    during_grid = float(np.nanmedian(during))
    deficit = pre_grid - during_grid
    threshold = test["pickup_fraction"] * deficit
    post = min(len(t), b + 1 + seconds_to_samples(0.1, dt))
    reached = np.flatnonzero(batt[a:post] >= threshold)
    pickup = float(t[a + reached[0]] - t[a]) if reached.size and deficit > 0 else None
    seg = bus[a:post]
    j_min, j_max = int(np.nanargmin(seg)), int(np.nanargmax(seg))
    details = {"sag_start_s": _t(t[a]), "sag_end_s": _t(t[b]), "sag_duration_s": _t((b - a + 1) * dt),
               "min_grid_voltage_pu": _r(np.nanmin(pu[a : b + 1])), "pre_sag_grid_kw": _r(pre_grid, 2),
               "grid_kw_during_sag": _r(during_grid, 2), "deficit_kw": _r(deficit, 2),
               "peak_battery_kw": _r(np.nanmax(batt[a:post]), 2), "min_bus_voltage_time_s": _t(t[a + j_min]),
               "max_bus_voltage_v": _r(seg[j_max], 3), "max_bus_voltage_time_s": _t(t[a + j_max])}
    kpis = [_kpi("min_bus_voltage_v", _r(seg[j_min], 3), test["min_bus_voltage_v"], "min", "V", "voltage_v_tolerance"),
            _kpi("battery_pickup_time_s", _t(pickup) if pickup is not None else None, test["max_pickup_s"], "max", "s",
                 "time_s_tolerance")]
    return _test_result(test, "ok", kpis, details, method, limitations)


# --------------------------------------------------------------------------- evaluate_thermal
def evaluate_thermal(df: pd.DataFrame, plan: dict, test: dict) -> dict:
    method = ("Peak = maximum sic_temp_c in the step. End slope = least-squares straight-line fit over the last 1 s; a "
              "slope near zero means the temperature has stabilised.")
    rows = step_rows(df, test["step"])
    if rows.empty:
        return _not_run(test, method)
    limitations = ["Single sensor; true junction temperature may be higher than the measured module temperature.",
                   "A short soak may end before thermal equilibrium; the end slope shows how far from it."]
    temp = rows["sic_temp_c"]
    t = rows[TIME_COLUMN]
    if temp.notna().sum() < 10:
        return _test_result(test, "insufficient temperature data", [], {}, method, limitations)
    i_max = temp.idxmax()
    last = rows[t >= t.max() - 1.0].dropna(subset=["sic_temp_c"])
    slope = float(np.polyfit(last[TIME_COLUMN], last["sic_temp_c"], 1)[0]) if len(last) >= 10 else None
    first = rows[t <= t.min() + 0.5]["sic_temp_c"]
    details = {"start_temp_c": _r(first.median(), 2), "peak_time_s": _t(t[i_max]),
               "end_temp_c": _r(last["sic_temp_c"].median(), 2) if len(last) else None,
               "soak_duration_s": _t(t.max() - t.min())}
    kpis = [_kpi("peak_sic_temp_c", _r(temp[i_max], 2), test["max_temp_c"], "max", "°C", "temp_c_tolerance"),
            _kpi("end_slope_c_per_s", _r(slope, 3) if slope is not None else None, test["max_end_slope_c_per_s"], "max",
                 "°C/s", "slope_c_per_s_tolerance")]
    return _test_result(test, "ok", kpis, details, method, limitations)


EVALUATORS = {
    "efficiency": evaluate_efficiency,
    "load_burst": analyze_load_bursts,
    "grid_sag": analyze_grid_sag,
    "thermal": evaluate_thermal,
}


# --------------------------------------------------------------------------- process tools
def run_validation_plan(df: pd.DataFrame, plan: dict, test_ids: list[str] | None = None,
                        test_types: list[str] | None = None) -> dict:
    """Run the plan's tests and return a PASS/FAIL scorecard plus per-test details."""
    tests = [t for t in plan["tests"]
             if (not test_ids or t["id"] in test_ids) and (not test_types or t["type"] in test_types)]
    results = [EVALUATORS[t["type"]](df, plan, t) for t in tests]
    scorecard = []
    for r in results:
        if not r["kpis"]:
            scorecard.append({"test_id": r["test_id"], "verdict": r["verdict"], "kpi": r["status"], "measured": None,
                              "limit": None, "unit": ""})
        for k in r["kpis"]:
            scorecard.append({"test_id": r["test_id"], "verdict": r["verdict"], "kpi": k["name"], "measured": k["measured"],
                              "limit": f"{'>=' if k['comparison'] == 'min' else '<='} {k['limit']}", "unit": k["unit"],
                              "kpi_pass": k["pass"]})
    counts = {v: sum(1 for r in results if r["verdict"] == v) for v in ("PASS", "FAIL", "NOT RUN", "INCONCLUSIVE")}
    unknown = sorted(set(test_ids or []) - {t["id"] for t in plan["tests"]})
    return {
        "tool": "run_validation_plan",
        "plan_profile": _plan_profile(plan),
        "summary": {"tests_run": len(results), **{k.lower().replace(" ", "_"): v for k, v in counts.items()},
                    "failed_tests": [r["test_id"] for r in results if r["verdict"] == "FAIL"],
                    **({"unknown_test_ids": unknown} if unknown else {})},
        "scorecard": scorecard,
        "tests": results,
        "method": "Each plan test is evaluated by its type-specific analysis (see each test's 'method'); a test PASSES "
                  "only if every KPI meets its illustrative acceptance criterion.",
        "limitations": ["Acceptance criteria are illustrative; PASS/FAIL is a comparison with those values only.",
                        "A FAIL is a prompt for engineering review, not a root-cause diagnosis."],
    }


def compare_to_baseline(df: pd.DataFrame, baseline_df: pd.DataFrame, plan: dict,
                        test_ids: list[str] | None = None) -> dict:
    """Run the same plan on the candidate and the baseline and report KPI deltas."""
    cand = run_validation_plan(df, plan, test_ids)
    base = run_validation_plan(baseline_df, plan, test_ids)
    base_by_id = {r["test_id"]: r for r in base["tests"]}
    rows, verdict_changes = [], []
    for r in cand["tests"]:
        b = base_by_id.get(r["test_id"])
        if b is None:
            continue
        if b["verdict"] != r["verdict"]:
            verdict_changes.append({"test_id": r["test_id"], "baseline": b["verdict"], "candidate": r["verdict"]})
        b_kpis = {k["name"]: k for k in b["kpis"]}
        for k in r["kpis"]:
            bk = b_kpis.get(k["name"])
            if bk is None:
                continue
            tol = plan["regression"][k["tolerance_key"]]
            if k["measured"] is None or bk["measured"] is None:
                delta, change = None, ("regression" if k["measured"] is None and bk["measured"] is not None else "unknown")
            else:
                delta = k["measured"] - bk["measured"]
                worse = delta < 0 if k["comparison"] == "min" else delta > 0
                change = ("regression" if worse else "improvement") if abs(delta) > tol else "no significant change"
            rows.append({"test_id": r["test_id"], "kpi": k["name"], "unit": k["unit"], "baseline": bk["measured"],
                         "candidate": k["measured"], "delta": _r(delta, 4) if delta is not None else None,
                         "tolerance": tol, "change": change, "baseline_verdict": b["verdict"],
                         "candidate_verdict": r["verdict"]})
    regressions = [x for x in rows if x["change"] == "regression"]
    return {
        "tool": "compare_to_baseline",
        "summary": {"kpis_compared": len(rows), "regressions": len(regressions),
                    "improvements": sum(1 for x in rows if x["change"] == "improvement"),
                    "regressed_tests": sorted({x["test_id"] for x in regressions}),
                    "new_failures": [v["test_id"] for v in verdict_changes if v["candidate"] == "FAIL"]},
        "verdict_changes": verdict_changes,
        "kpi_deltas": rows,
        "method": "The same validation plan is run on both logs. delta = candidate - baseline. A KPI is a regression if "
                  "it moved in the worse direction by more than its tolerance in the plan's 'regression' section.",
        "limitations": ["Assumes both runs used the same test sequence and comparable bench conditions.",
                        "One run per revision: run-to-run variation is not measured, so small deltas may be noise."],
    }

"""generate_plots: Plotly figures built from raw samples + deterministic tool results.

Plot types:
    overview          every channel vs time, test-step markers, illustrative limits, violation peaks
    efficiency_curve  efficiency vs load for the candidate (and baseline), with acceptance criteria
    burst             800 V bus response to the worst GPU load edge (baseline overlaid if loaded)
    grid_sag          grid voltage, grid vs battery power, and bus voltage through the sag
    thermal           SiC temperature during the soak vs its limit (baseline overlaid if loaded)

Plots never compute new measurements for the agent; it only receives a short description.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analysis
from .data_io import MEASUREMENT_COLUMNS, STEP_COLUMN, TIME_COLUMN, UNITS

PLOT_TYPES = ["overview", "efficiency_curve", "burst", "grid_sag", "thermal"]

# Light chart surface. Slot 1 (blue) = candidate / main series; slot 2 (orange, dashed) = baseline or
# second series. Red is reserved for limits and violations. Palette validated with the dataviz checker.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
SERIES_2 = "#eb6834"
SERIES_BAND = "rgba(42, 120, 214, 0.14)"
CRITICAL = "#d03b3b"
CRITICAL_FILL = "rgba(208, 59, 59, 0.14)"
GOOD = "#0ca30c"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
LABELS = {
    "grid_voltage_pu": "Grid voltage",
    "grid_power_kw": "Grid-port power",
    "battery_power_kw": "Battery-port power",
    "rack_power_kw": "Rack (800 V) power",
    "dc_bus_voltage_v": "800 V DC bus",
    "sic_temp_c": "SiC module temperature",
}
MAX_POINTS_PER_TRACE = 6000


def decimate_min_max(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS_PER_TRACE):
    """Shrink a long trace for display while keeping every spike (min and max of each bucket)."""
    n = len(x)
    if n <= max_points:
        return x, y
    buckets = max_points // 2
    size = int(np.ceil(n / buckets))
    # Lay the samples out as a (buckets x size) table, padding the end with NaN, then find each row's min and max.
    table = np.concatenate([np.asarray(y, dtype=float), np.full(buckets * size - n, np.nan)]).reshape(buckets, size)
    missing = np.isnan(table)
    i_min = np.where(missing, np.inf, table).argmin(axis=1)  # an all-NaN row returns 0 -> keeps the gap visible
    i_max = np.where(missing, -np.inf, table).argmax(axis=1)
    start = np.arange(buckets) * size
    idx = np.unique(np.concatenate([start + i_min, start + i_max]))
    idx = idx[idx < n]
    return x[idx], y[idx]


def _style(fig: go.Figure, title: str, height: int, subtitle: str = "", legend: bool = False,
           right_margin: int = 24) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK), x=0.01, xanchor="left",
                   subtitle=dict(text=subtitle, font=dict(size=12, color=INK_SECONDARY))),
        height=height, margin=dict(l=64, r=right_margin, t=96 if subtitle else 70, b=110 if legend else 50),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=INK_SECONDARY),
        # Legend under the plot so it never collides with titles or subplot titles.
        showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, xanchor="left"),
        hovermode="x unified", hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_annotations(font=dict(family=FONT, color=INK_SECONDARY))
    return fig


def _title(text: str, is_synthetic: bool) -> str:
    return f"{'SYNTHETIC DATA · ' if is_synthetic else ''}{text}"


def _verdict_mark(verdict: str) -> str:
    color = GOOD if verdict == "PASS" else CRITICAL
    icon = "✓" if verdict == "PASS" else "✕"
    return f"<span style='color:{color}'>{icon}</span> {verdict}"


def _find_test(plan: dict, test_type: str, test_id: str | None) -> dict | None:
    for t in plan["tests"]:
        if t["type"] == test_type and (test_id is None or t["id"] == test_id):
            return t
    return None


# --------------------------------------------------------------------------- overview
def overview_figure(df: pd.DataFrame, plan: dict, is_synthetic: bool) -> go.Figure:
    """Row 1 is a thin 'test sequence' strip (one band per test step); rows 2-7 are the channels."""
    violations = analysis.detect_threshold_violations(df, plan)
    t = df[TIME_COLUMN].to_numpy()
    dt = analysis.median_dt(df)
    n_rows = len(MEASUREMENT_COLUMNS) + 1
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.04] + [0.16] * len(MEASUREMENT_COLUMNS),
                        subplot_titles=["Test sequence"] + [f"{LABELS[c]} ({UNITS[c]})" for c in MEASUREMENT_COLUMNS])
    # Shapes and labels are collected in lists and added once at the end: Plotly's per-row helpers
    # (add_hline / add_vrect / add_vline with row=...) are slow on a many-panel figure.
    shapes, notes = [], []

    tests_by_step: dict[int, list[str]] = {}
    for test in plan["tests"]:
        tests_by_step.setdefault(test["step"], []).append(test["id"])
    for i, (step, rows) in enumerate(df.dropna(subset=[STEP_COLUMN]).groupby(STEP_COLUMN)):
        x0, x1 = float(rows[TIME_COLUMN].min()), float(rows[TIME_COLUMN].max())
        shapes.append(dict(type="rect", xref="x", yref="y domain", x0=x0, x1=x1, y0=0, y1=1, line=dict(color=SURFACE, width=2),
                           fillcolor="#e1e0d9" if i % 2 else "#f0efec"))
        shapes.append(dict(type="line", xref="x", yref="paper", x0=x0, x1=x0, y0=0, y1=1,
                           line=dict(color=MUTED, width=1, dash="dot"), layer="below"))
        label = "/".join(tests_by_step.get(int(step), []))
        if label:  # steps without a plan test (e.g. warm-up) stay unlabeled
            notes.append(dict(xref="x", yref="y domain", x=(x0 + x1) / 2, y=0.5, showarrow=False, text=label,
                              font=dict(size=10, color=INK)))
    fig.update_yaxes(visible=False, row=1, col=1)

    for row, col in enumerate(MEASUREMENT_COLUMNS, start=2):
        unit = UNITS[col]
        ax = str(row)  # subplot axes are named x, x2, x3 ... / y, y2, y3 ...
        x, y = decimate_min_max(t, df[col].to_numpy(dtype=float))
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line=dict(color=SERIES, width=1.5), name=LABELS[col],
                                   hovertemplate=f"%{{y:.3f}} {unit}<extra>{LABELS[col]}</extra>"), row=row, col=1)
        lim = plan.get("channels", {}).get(col, {})
        for key, anchor in (("max", "bottom"), ("min", "top")):
            if key in lim:
                shapes.append(dict(type="line", xref=f"x{ax} domain", yref=f"y{ax}", x0=0, x1=1, y0=lim[key], y1=lim[key],
                                   line=dict(color=CRITICAL, width=1, dash="dash")))
                notes.append(dict(xref=f"x{ax} domain", yref=f"y{ax}", x=0.005, y=lim[key], xanchor="left", yanchor=anchor,
                                  showarrow=False, text=f"illustrative {key} {lim[key]} {unit}",
                                  font=dict(size=11, color=INK_SECONDARY)))
        events = violations["channels"].get(col, {}).get("events", [])
        for e in events[:50]:
            shapes.append(dict(type="rect", xref=f"x{ax}", yref=f"y{ax} domain", x0=e["start_s"] - dt / 2,
                               x1=e["end_s"] + dt / 2, y0=0, y1=1, fillcolor=CRITICAL_FILL, line_width=0, layer="below"))
        if events:
            fig.add_trace(go.Scatter(x=[e["peak_time_s"] for e in events], y=[e["peak_value"] for e in events], mode="markers",
                                     marker=dict(symbol="x", size=10, color=CRITICAL), name="Violation peak",
                                     hovertemplate=f"✕ peak %{{y:.3f}} {unit}<extra>violation</extra>"), row=row, col=1)
    fig.update_layout(shapes=shapes, annotations=list(fig.layout.annotations) + notes)
    fig.update_xaxes(title_text="timestamp_s (s)", row=n_rows, col=1)
    subtitle = (f"{violations['total_events']} limit-violation event(s) · dashed red = illustrative limit · "
                "✕ = violation peak · dotted = test-step start")
    return _style(fig, _title("Validation run overview", is_synthetic), height=1220, subtitle=subtitle)


# --------------------------------------------------------------------------- efficiency curve
def efficiency_figure(df, plan, is_synthetic, baseline_df=None, labels=("candidate", "baseline")):
    def points(frame):
        res = analysis.run_validation_plan(frame, plan, test_types=["efficiency"])
        return [(r["details"]["measured_load_pct"], r["kpis"][0]["measured"], r) for r in res["tests"] if r["kpis"]]

    cand = points(df)
    if not cand:
        return None, "No efficiency tests could be evaluated in this log."
    fig = go.Figure()
    series = [(cand, labels[0], SERIES, "solid")]
    if baseline_df is not None:
        series.append((points(baseline_df), labels[1], SERIES_2, "dash"))
    for pts, name, color, dash in series:
        if not pts:
            continue
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=name,
                                 line=dict(color=color, width=2, dash=dash), marker=dict(size=9, color=color,
                                 line=dict(width=2, color=SURFACE)),
                                 customdata=[p[2]["test_id"] for p in pts],
                                 hovertemplate="%{customdata}: %{y:.3f}% at %{x:.1f}% load<extra>" + name + "</extra>"))
        fig.add_annotation(x=pts[-1][0], y=pts[-1][1], text=name, showarrow=False, xanchor="left", xshift=10,
                           font=dict(size=11, color=INK_SECONDARY))
    limits = [(p[2]["details"]["target_load_pct"], p[2]["kpis"][0]["limit"], p[2]["test_id"], p[2]["verdict"]) for p in cand]
    fig.add_trace(go.Scatter(x=[l[0] for l in limits], y=[l[1] for l in limits], mode="markers", name="Illustrative minimum",
                             marker=dict(symbol="line-ew", size=26, color=CRITICAL, line=dict(width=2, color=CRITICAL)),
                             customdata=[[l[2], l[3]] for l in limits],
                             hovertemplate="%{customdata[0]} minimum %{y}% · candidate %{customdata[1]}<extra></extra>"))
    fails = [l[2] for l in limits if l[3] == "FAIL"]
    fig.update_xaxes(title_text="Measured load (% of rated)", range=[15, 112])
    fig.update_yaxes(title_text="Efficiency (%)", ticksuffix="%")
    subtitle = ("red bars = illustrative acceptance minimum per load point · "
                + (f"failing: {', '.join(fails)}" if fails else "all load points meet their minimum"))
    return _style(fig, _title("Efficiency vs load", is_synthetic), height=500, subtitle=subtitle, legend=True,
                  right_margin=150), "ok"


# --------------------------------------------------------------------------- burst
def burst_figure(df, plan, test, is_synthetic, baseline_df=None, labels=("candidate", "baseline")):
    res = analysis.analyze_load_bursts(df, plan, test)
    if res["status"] != "ok":
        return None, f"{test['id']}: {res['status']}"
    edge = next(e for e in res["details"]["edges"] if e["edge_number"] == res["details"]["worst_deviation_edge"])
    t0, t1 = edge["time_s"] - 0.02, edge["time_s"] + 0.06
    band = test["recovery_band_v"]

    def window(frame):
        sel = (frame[TIME_COLUMN] >= t0) & (frame[TIME_COLUMN] <= t1)
        return frame.loc[sel]

    w = window(df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3],
                        subplot_titles=["800 V DC bus (V)", "Rack power (kW)"])
    fig.add_shape(type="rect", x0=edge["time_s"], x1=t1, y0=edge["final_v"] - band, y1=edge["final_v"] + band,
                  fillcolor=SERIES_BAND, line_width=0, layer="below", row=1, col=1)
    if baseline_df is not None:
        b = window(baseline_df)
        fig.add_trace(go.Scatter(x=b[TIME_COLUMN], y=b["dc_bus_voltage_v"], mode="lines", name=labels[1],
                                 line=dict(color=SERIES_2, width=1.5, dash="dash"),
                                 hovertemplate="%{y:.2f} V<extra>" + labels[1] + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=w[TIME_COLUMN], y=w["dc_bus_voltage_v"], mode="lines", name=labels[0],
                             line=dict(color=SERIES, width=2), hovertemplate="%{y:.2f} V<extra>" + labels[0] + "</extra>"),
                  row=1, col=1)
    peak_v = edge["baseline_v"] + edge["peak_deviation_v"]
    fig.add_trace(go.Scatter(x=[edge["peak_time_s"]], y=[peak_v], mode="markers", name="Peak",
                             marker=dict(size=11, color=INK, line=dict(width=2, color=SURFACE)),
                             hovertemplate=f"peak deviation {edge['peak_deviation_v']:+.2f} V<extra></extra>"), row=1, col=1)
    fig.add_annotation(x=edge["peak_time_s"], y=peak_v, text=f"peak {edge['peak_deviation_v']:+.2f} V", showarrow=True,
                       ax=-70, ay=0, xanchor="right", arrowcolor=MUTED, font=dict(size=11, color=INK), row=1, col=1)
    if edge.get("recovery_time_s") is not None:
        fig.add_vline(x=edge["time_s"] + edge["recovery_time_s"], line=dict(color=INK_SECONDARY, width=1), row=1, col=1,
                      annotation_text=f"recovered +{edge['recovery_time_s']} s", annotation_position="top right",
                      annotation_font=dict(size=11, color=INK_SECONDARY))
    fig.add_vline(x=edge["time_s"], line=dict(color=MUTED, width=1, dash="dot"), row="all", col=1)
    fig.add_trace(go.Scatter(x=w[TIME_COLUMN], y=w["rack_power_kw"], mode="lines", name="Rack power", showlegend=False,
                             line=dict(color=SERIES, width=1.5), hovertemplate="%{y:.1f} kW<extra>Rack power</extra>"),
                  row=2, col=1)
    fig.update_xaxes(title_text="timestamp_s (s)", row=2, col=1)
    title = _title(f"{test['id']} worst GPU load edge (#{edge['edge_number']}, {edge['delta_kw']:+.1f} kW) · "
                   f"{_verdict_mark(res['verdict'])}", is_synthetic)
    k = {x["name"]: x for x in res["kpis"]}
    subtitle = (f"max |deviation| {k['max_abs_bus_deviation_v']['measured']} V (limit {test['max_deviation_v']} V) · "
                f"worst recovery {k['worst_recovery_time_s']['measured']} s (limit {test['max_recovery_s']} s) · "
                f"shaded = ±{band} V recovery band")
    return _style(fig, title, height=560, subtitle=subtitle, legend=baseline_df is not None), "ok"


# --------------------------------------------------------------------------- grid sag
def sag_figure(df, plan, test, is_synthetic):
    res = analysis.analyze_grid_sag(df, plan, test)
    if res["status"] != "ok":
        return None, f"{test['id']}: {res['status']}"
    d = res["details"]
    t0, t1 = d["sag_start_s"] - 0.05, d["sag_end_s"] + 0.15
    w = df[(df[TIME_COLUMN] >= t0) & (df[TIME_COLUMN] <= t1)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Grid voltage (pu)", "Port power (kW)", "800 V DC bus (V)"])
    fig.add_trace(go.Scatter(x=w[TIME_COLUMN], y=w["grid_voltage_pu"], mode="lines", name="Grid voltage", showlegend=False,
                             line=dict(color=SERIES, width=1.5), hovertemplate="%{y:.3f} pu<extra></extra>"), row=1, col=1)
    for col, color, dash, name in (("grid_power_kw", SERIES, "solid", "Grid port"), ("battery_power_kw", SERIES_2, "dash", "Battery port")):
        fig.add_trace(go.Scatter(x=w[TIME_COLUMN], y=w[col], mode="lines", name=name, line=dict(color=color, width=2, dash=dash),
                                 hovertemplate="%{y:.1f} kW<extra>" + name + "</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=w[TIME_COLUMN], y=w["dc_bus_voltage_v"], mode="lines", name="Bus voltage", showlegend=False,
                             line=dict(color=SERIES, width=2), hovertemplate="%{y:.2f} V<extra></extra>"), row=3, col=1)
    fig.add_hline(y=test["min_bus_voltage_v"], line=dict(color=CRITICAL, width=1, dash="dash"), row=3, col=1,
                  annotation_text=f"illustrative min {test['min_bus_voltage_v']} V", annotation_position="bottom left",
                  annotation_font=dict(size=11, color=INK_SECONDARY))
    fig.add_vrect(x0=d["sag_start_s"], x1=d["sag_end_s"], fillcolor="rgba(137, 135, 129, 0.10)", line_width=0, row="all", col=1)
    k = {x["name"]: x for x in res["kpis"]}
    pickup = k["battery_pickup_time_s"]["measured"]
    if pickup is not None:
        fig.add_vline(x=d["sag_start_s"] + pickup, line=dict(color=INK_SECONDARY, width=1), row=2, col=1,
                      annotation_text=f"battery pickup +{pickup} s", annotation_position="top right",
                      annotation_font=dict(size=11, color=INK_SECONDARY))
    fig.update_xaxes(title_text="timestamp_s (s)", row=3, col=1)
    title = _title(f"{test['id']} grid-sag ride-through · {_verdict_mark(res['verdict'])}", is_synthetic)
    subtitle = (f"bus minimum {k['min_bus_voltage_v']['measured']} V (limit {test['min_bus_voltage_v']} V) · battery pickup "
                f"{pickup} s (limit {test['max_pickup_s']} s) · grey = sag")
    return _style(fig, title, height=640, subtitle=subtitle, legend=True), "ok"


# --------------------------------------------------------------------------- thermal
def thermal_figure(df, plan, test, is_synthetic, baseline_df=None, labels=("candidate", "baseline")):
    res = analysis.evaluate_thermal(df, plan, test)
    if res["status"] != "ok":
        return None, f"{test['id']}: {res['status']}"
    fig = go.Figure()
    frames = [(df, labels[0], SERIES, "solid")] + ([(baseline_df, labels[1], SERIES_2, "dash")] if baseline_df is not None else [])
    for frame, name, color, dash in frames:
        rows = analysis.step_rows(frame, test["step"])
        if rows.empty:
            continue
        x, y = decimate_min_max(rows[TIME_COLUMN].to_numpy(), rows["sic_temp_c"].to_numpy(dtype=float))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=2, dash=dash),
                                 hovertemplate="%{y:.2f} °C<extra>" + name + "</extra>"))
        fig.add_annotation(x=x[-1], y=y[-1], text=name, showarrow=False, xanchor="right", yshift=12,
                           font=dict(size=11, color=INK_SECONDARY))
    fig.add_hline(y=test["max_temp_c"], line=dict(color=CRITICAL, width=1, dash="dash"),
                  annotation_text=f"illustrative max {test['max_temp_c']} °C", annotation_position="top left",
                  annotation_font=dict(size=11, color=INK_SECONDARY))
    fig.update_xaxes(title_text="timestamp_s (s)")
    fig.update_yaxes(title_text="SiC module temperature (°C)")
    k = {x["name"]: x for x in res["kpis"]}
    title = _title(f"{test['id']} thermal soak · {_verdict_mark(res['verdict'])}", is_synthetic)
    subtitle = (f"peak {k['peak_sic_temp_c']['measured']} °C (limit {test['max_temp_c']} °C) · end slope "
                f"{k['end_slope_c_per_s']['measured']} °C/s (limit {test['max_end_slope_c_per_s']} °C/s)")
    return _style(fig, title, height=460, subtitle=subtitle, legend=baseline_df is not None), "ok"


# --------------------------------------------------------------------------- entry point
def generate_plots(df: pd.DataFrame, plan: dict, plot_type: str = "overview", test_id: str | None = None,
                   is_synthetic: bool = False, baseline_df: pd.DataFrame | None = None,
                   labels: tuple[str, str] = ("candidate", "baseline")) -> tuple[dict, list[dict]]:
    """Return (summary for the agent, list of {plot_id, title, description, figure})."""
    if plot_type not in PLOT_TYPES:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Choose from {PLOT_TYPES}.")
    fig, message, plot_id, description = None, "ok", plot_type, ""
    if plot_type == "overview":
        fig = overview_figure(df, plan, is_synthetic)
        description = "All six channels vs time with test-step markers, illustrative limits and violation peaks."
    elif plot_type == "efficiency_curve":
        fig, message = efficiency_figure(df, plan, is_synthetic, baseline_df, labels)
        description = "Efficiency vs load for the candidate" + (" and baseline" if baseline_df is not None else "") + \
                      ", with each load point's illustrative minimum."
    else:
        test_type = {"burst": "load_burst", "grid_sag": "grid_sag", "thermal": "thermal"}[plot_type]
        test = _find_test(plan, test_type, test_id)
        if test is None:
            message = f"No {test_type} test{' ' + test_id if test_id else ''} in the plan."
        else:
            plot_id = f"{plot_type}_{test['id']}"
            if plot_type == "burst":
                fig, message = burst_figure(df, plan, test, is_synthetic, baseline_df, labels)
                description = f"Bus-voltage response to the worst load edge in {test['id']}."
            elif plot_type == "grid_sag":
                fig, message = sag_figure(df, plan, test, is_synthetic)
                description = f"Grid voltage, grid vs battery power and bus voltage through the {test['id']} sag."
            else:
                fig, message = thermal_figure(df, plan, test, is_synthetic, baseline_df, labels)
                description = f"SiC temperature during {test['id']} vs its illustrative limit."
    figures = [] if fig is None else [{"plot_id": plot_id, "title": fig.layout.title.text, "description": description,
                                       "figure": fig}]
    summary = {
        "tool": "generate_plots",
        "plot_type": plot_type,
        "status": message,
        "plots_created": [{k: f[k] for k in ("plot_id", "description")} for f in figures],
        "note": "Figures are displayed to the engineer in the UI. You cannot see them; only describe values you got "
                "from an analysis tool.",
        "method": f"Plotly figures of raw samples (min/max-decimated above {MAX_POINTS_PER_TRACE} points per trace) "
                  "overlaid with analysis-tool results.",
        "limitations": ["Plots are visual aids; the measured values are the analysis-tool outputs."],
    }
    return summary, figures

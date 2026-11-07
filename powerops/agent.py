"""The PowerOps validation agent: turn a bench run (and optionally a baseline run)
into a validation scorecard, a regression check, and a draft report.

Two interchangeable "brains" decide WHICH tools to run:

* LLM mode  - the OpenAI model reads the question, requests tool calls, reads the
              JSON results, and writes a short engineering summary.
* Demo mode - a deterministic keyword planner picks tools and a fixed template
              writes the answer. No API key or network; same input -> same output.

In BOTH modes the numbers come only from the Python tools. The model never sees
raw samples, and numbers_check flags any number in its prose that no tool produced.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .data_io import Dataset
from .numbers_check import find_unverified_numbers
from .report import build_report
from .tools import ToolCallRecord, ToolContext, build_tool_schemas, run_tool

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
MAX_TOOL_ROUNDS = 8
MAX_TOOL_RESULT_CHARS = 24_000

SYSTEM_PROMPT = """You are PowerOps Agent, a read-only assistant for power-electronics validation engineers. \
You analyse bench logs from a multi-port solid-state transformer (SST) that feeds an 800 V DC bus (grid port, \
battery port, rack/load port). Your main job is to turn a validation run into a scorecard against the validation \
plan, check for regressions against a baseline run, and draft a report for engineering review. \
This is an independent student prototype on synthetic data, not an official product of any company.

Rules you must follow:
1. You learn about the data ONLY by calling the provided tools. You never see raw samples.
2. Every number you write must be copied from a tool result (rounding for readability is fine). Never calculate \
new measurements yourself: no arithmetic, averaging, differencing or unit conversion of measured values. If a \
quantity is not available from a tool, say so.
3. For a validation report: call run_validation_plan, check_power_balance, and compare_to_baseline if a baseline \
is loaded; add generate_plots for the failing or regressed tests. Call the smallest set of tools that answers \
narrower questions.
4. If check_power_balance is SUSPECT, say clearly that efficiency results may be unreliable.
5. Acceptance criteria and limits are ILLUSTRATIVE, not a real product specification. Say "illustrative".
6. You cannot control hardware, send commands, or change settings, limits or the plan. Never decide whether a \
revision is ready to release, ship or certify: say that the scorecard is input to an engineering review and sign-off.
7. Answer in concise Markdown: the verdict summary first, then failures and regressions with values, units and \
test IDs, then one line on the most relevant method or limitation."""

_CONTROL_VERBS = (r"(turn|switch|shut|trip|disable|enable|reset|reboot|restart|energi[sz]e|de-?energi[sz]e|"
                  r"set(?!\s+up)|change|adjust|increase|decrease|raise|lower|send|write|issue|push|flash|command)")
_CONTROL_TARGETS = (r"(converter|inverter|sst|transformer|supply|breaker|relay|contactor|hardware|device|unit|bench|"
                    r"output|load|setpoint|set point|limit|duty|gain|voltage|current|fan|register|firmware|power|pwm)")
CONTROL_PATTERNS = [
    rf"^\s*(please\s+|can you\s+|could you\s+|would you\s+|go ahead and\s+|now\s+)?{_CONTROL_VERBS}\b.*\b{_CONTROL_TARGETS}",
    r"\b(modbus|can bus|write (to )?(the )?registers?|send (a |the )?commands?)\b",
]
SAFETY_PATTERNS = [
    r"\bis (it|this|the \w+) safe\b",
    r"\bsafe to (operate|run|use|ship|deploy|energi[sz]e|touch|continue|release)\b",
    r"\b(ready|ok|okay|good|cleared) (to|for) (release|ship|deploy|production)\b",
    r"\bshould (i|we) (ship|deploy|operate|release|energi[sz]e|continue|keep running)\b",
    r"\b(certif|qualif|approve|sign[- ]?off)\w*\b",
]
CONTROL_REFUSAL = (
    "I can't do that. PowerOps Agent is read-only analysis software: it has no connection to any hardware "
    "and cannot send commands, change setpoints, limits or the validation plan, or operate equipment. I can "
    "analyse the loaded run instead - for example: *\"Generate the validation report\"* or *\"Did this revision "
    "regress against the baseline?\"*"
)
SAFETY_NOTICE = (
    "This prototype does not decide whether a revision is safe, ready to release, ship or certify. The scorecard "
    "compares measurements with illustrative criteria and is input to an engineering review; the decision and "
    "sign-off belong to qualified engineers using approved procedures."
)


def classify_request(question: str) -> str:
    """Return 'control', 'safety_judgement' or 'analysis'."""
    q = question.lower()
    if any(re.search(p, q) for p in CONTROL_PATTERNS):
        return "control"
    if any(re.search(p, q) for p in SAFETY_PATTERNS):
        return "safety_judgement"
    return "analysis"


# --------------------------------------------------------------------------- demo planner
TOPICS = {
    "efficiency": [r"\befficien", r"\bloss(es)?\b"],
    "load_burst": [r"\bburst", r"\bgpu\b", r"\btransient", r"\bdips?\b", r"\brecover", r"\bload (edge|step)", r"\bovershoot"],
    "grid_sag": [r"\bsag", r"\bride[- ]?through", r"\bgrid (event|fault|dip)", r"\bbattery\b", r"\bpick ?up", r"\bbackup"],
    "thermal": [r"\btherm", r"\btemp", r"\bsic\b", r"\bhot\b", r"\bheat"],
}
COMPARE_PATTERNS = [r"\bregress", r"\bcompar", r"\bbaseline", r"\bprevious\b", r"\brev[- ]?a\b", r"\bversus\b", r"\bvs\.?\b",
                    r"\bworse\b", r"\bbetter\b", r"\bchanged?\b"]
INTEGRITY_PATTERNS = [r"\bsensors?\b", r"\bintegrity", r"\bpower balance", r"\bcalibrat", r"\btrust", r"\boffset"]
VIOLATION_PATTERNS = [r"\bviolat", r"\bexceed", r"\bout of (spec|limits?)", r"\blimits?\b"]
SUMMARY_PATTERNS = [r"\bsummar", r"\boverview\b", r"\bdescribe\b", r"\bwhat('s| is) in\b", r"\btest steps?\b"]
FULL_PATTERNS = [r"\breport\b", r"\bvalidation\b", r"\bscorecard\b", r"\bfull\b", r"\ball (the )?tests\b", r"\bpass", r"\bfail",
                 r"\bready\b", r"\brelease\b", r"\bship\b", r"\bsign[- ]?off\b"]
PLOT_FOR_TOPIC = {"efficiency": "efficiency_curve", "load_burst": "burst", "grid_sag": "grid_sag", "thermal": "thermal"}


def _has(q: str, patterns: list[str]) -> bool:
    return any(re.search(p, q) for p in patterns)


def plan_demo_tools(question: str, has_baseline: bool) -> list[tuple[str, dict]]:
    """Choose tools with simple, transparent keyword rules (no LLM)."""
    q = question.lower()
    topics = sorted(t for t, pats in TOPICS.items() if _has(q, pats))
    compare, integrity = _has(q, COMPARE_PATTERNS), _has(q, INTEGRITY_PATTERNS)
    violations, summary = _has(q, VIOLATION_PATTERNS), _has(q, SUMMARY_PATTERNS)
    full = _has(q, FULL_PATTERNS) or not (topics or compare or integrity or violations or summary)

    if full:
        plan = [("run_validation_plan", {}), ("check_power_balance", {})]
        if has_baseline:
            plan.append(("compare_to_baseline", {}))
        plan += [("generate_plots", {"plot_type": p}) for p in ("efficiency_curve", "burst", "grid_sag", "thermal")]
        return plan

    plan: list[tuple[str, dict]] = []
    if summary:
        plan.append(("summarize_run", {}))
    if integrity or "efficiency" in topics:
        plan.append(("check_power_balance", {}))
    if topics:
        plan.append(("run_validation_plan", {"test_types": topics}))
    elif compare and not has_baseline:  # nothing to compare against: the scorecard is the next best answer
        plan.append(("run_validation_plan", {}))
    if compare and has_baseline:
        plan.append(("compare_to_baseline", {}))
    if violations:
        plan.append(("detect_threshold_violations", {}))
    if summary or violations:
        plan.append(("generate_plots", {"plot_type": "overview"}))
    plan += [("generate_plots", {"plot_type": PLOT_FOR_TOPIC[t]}) for t in topics]
    if compare and not topics:
        plan.append(("generate_plots", {"plot_type": "efficiency_curve"}))
    return plan


def _kpi_text(k: dict) -> str:
    op = ">=" if k["comparison"] == "min" else "<="
    value = "not measured" if k["measured"] is None else f"{k['measured']} {k['unit']}"
    return f"{k['name']} {value} (illustrative limit {op} {k['limit']} {k['unit']})"


def compose_demo_answer(records: list[ToolCallRecord], notes: list[str] | None = None) -> str:
    """Write the answer from tool results with a fixed template (no LLM)."""
    lines = ["*Deterministic demo mode: tools were chosen by keyword rules and this text was filled in from their "
             "outputs by a template. No LLM was used.*", ""]
    plots: list[str] = []
    suspect = False
    ran_efficiency = False
    for rec in records:
        r = rec.result
        if not rec.ok:
            lines += [f"**{rec.name}** could not run: {r['error']}", ""]
            continue
        if rec.name == "run_validation_plan":
            s = r["summary"]
            lines.append(f"**Validation scorecard for {r.get('run', 'this run')}: {s['pass']} of {s['tests_run']} tests PASS**"
                         + (f" · FAIL: {', '.join(s['failed_tests'])}" if s["failed_tests"] else ""))
            for t in r["tests"]:
                ran_efficiency |= t["type"] == "efficiency"
                detail = "; ".join(_kpi_text(k) for k in t["kpis"]) or t["status"]
                lines.append(f"- {t['test_id']} ({t['description']}): **{t['verdict']}** - {detail}")
            lines.append("")
        elif rec.name == "compare_to_baseline":
            s = r["summary"]
            lines.append(f"**Regression check: {r['candidate_run']} vs {r['baseline_run']}** - {s['regressions']} KPI "
                         f"regression(s) across {len(s['regressed_tests'])} test(s); new failures: "
                         f"{', '.join(s['new_failures']) or 'none'}")
            for x in r["kpi_deltas"]:
                if x["change"] == "regression":
                    delta = "n/a" if x["delta"] is None else f"{x['delta']:+}"
                    lines.append(f"- {x['test_id']} {x['kpi']}: {x['baseline']} to {x['candidate']} {x['unit']} "
                                 f"(change {delta}, tolerance {x['tolerance']}) - verdict {x['baseline_verdict']} to "
                                 f"{x['candidate_verdict']}")
            if not s["regressions"]:
                lines.append("- No KPI moved in the worse direction by more than its tolerance.")
            lines.append("")
        elif rec.name == "check_power_balance":
            suspect = r["verdict"] == "SUSPECT"
            lines.append(f"**Data integrity (power sensors): {r['verdict']}**")
            lines += [f"- {f}" for f in r["findings"]] or ["- Battery-port idle reading and implied efficiency are plausible."]
            lines.append("")
        elif rec.name == "summarize_run":
            run = r["run"]
            lines.append("**Run summary**")
            lines.append(f"- {run['n_samples']} samples over {run['duration_s']} s (about {run['approx_sample_rate_hz']} Hz).")
            for s in r["test_steps"]:
                lines.append(f"- Step {s['step']} ({'/'.join(s['plan_tests']) or 'no plan test'}): {s['start_s']} to "
                             f"{s['end_s']} s, mean rack power {s['mean_rack_power_kw']} kW.")
            if r["plan_tests_without_data"]:
                lines.append(f"- Plan tests with no data in this log: {', '.join(r['plan_tests_without_data'])}.")
            lines.append("")
        elif rec.name == "detect_threshold_violations":
            lines.append(f"**Illustrative-limit violations: {r['total_events']} event(s)**")
            for col, c in r["channels"].items():
                if not c.get("events"):
                    lines.append(f"- {col}: {c['status']}.")
                for e in c.get("events", []):
                    word = "above max" if e["type"] == "above_max" else "below min"
                    lines.append(f"- {col} {word} {e['limit']} {c['unit']}: {e['start_s']} to {e['end_s']} s, peak "
                                 f"{e['peak_value']} {c['unit']} at {e['peak_time_s']} s.")
            lines.append("")
        elif rec.name == "generate_plots":
            plots += [p["plot_id"] for p in r["plots_created"]] or [f"none ({r['status']})"]
    if suspect and ran_efficiency:
        lines += ["⚠ The power-sensor integrity check is SUSPECT, so the efficiency verdicts above may be caused by a "
                  "measurement problem rather than the hardware. Resolve the sensor issue before acting on them.", ""]
    lines += notes or []
    if plots:
        lines.append(f"**Plots shown:** {', '.join(plots)}")
    lines.append("")
    lines.append("*Draft for engineering review. Release decisions and sign-off are made by qualified engineers, not by this tool.*")
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- result object
@dataclass
class AgentResult:
    question: str
    answer: str
    mode: str  # "demo", "llm", "demo (fallback)" or "refused"
    model: str | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    figures: list[dict] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    report_markdown: str = ""


# --------------------------------------------------------------------------- LLM mode
def _llm_user_message(question: str, ctx: ToolContext) -> str:
    context = {
        "candidate_run": ctx.dataset.info(),
        "baseline_run": ctx.baseline.info() if ctx.baseline else "none loaded",
        "validation_plan": {"profile": ctx.plan.get("profile_name"),
                            "tests": [{"id": t["id"], "type": t["type"], "description": t.get("description", "")}
                                      for t in ctx.plan["tests"]]},
    }
    return f"Context (metadata only, no raw samples):\n{json.dumps(context, indent=1)}\n\nEngineer's question: {question}"


def _tool_message_content(result: dict) -> str:
    text = json.dumps(result, default=str, separators=(",", ":"))
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + '..."[TRUNCATED: result too long; ask for fewer tests]"'
    return text


def _run_llm(question: str, ctx: ToolContext, client, model: str, max_rounds: int):
    """The tool-calling loop. Returns (answer, records, notices)."""
    records: list[ToolCallRecord] = []
    notices: list[str] = []
    tools = build_tool_schemas(ctx.plan)
    extra = {"temperature": float(os.environ["OPENAI_TEMPERATURE"])} if os.getenv("OPENAI_TEMPERATURE") else {}
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": _llm_user_message(question, ctx)}]
    for _ in range(max_rounds):
        response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", **extra)
        message = response.choices[0].message
        if not message.tool_calls:
            return (message.content or "").strip(), records, notices
        # 1) Remember what the model asked for...
        #    Provider-specific extras (e.g. Gemini's thought_signature on each tool call) must be returned
        #    unchanged or the next request is rejected, so they are copied through.
        messages.append({"role": "assistant", "content": message.content, "tool_calls": [
            {**(getattr(tc, "model_extra", None) or {}),
             "id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in message.tool_calls]})
        # 2) ...run each requested tool in Python, and send the JSON result back.
        for tc in message.tool_calls:
            record = run_tool(tc.function.name, tc.function.arguments, ctx)
            records.append(record)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": _tool_message_content(record.result)})
    notices.append(f"The model asked for tools more than {max_rounds} times; it was asked to answer with what it had.")
    response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="none", **extra)
    return (response.choices[0].message.content or "").strip(), records, notices


def _make_openai_client():
    from openai import OpenAI  # imported lazily so demo mode works without API configuration

    # OPENAI_BASE_URL lets any OpenAI-compatible endpoint be used (e.g. Gemini's).
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL") or None,
                  timeout=60.0, max_retries=2)


# --------------------------------------------------------------------------- entry point
def run_agent(
    question: str,
    dataset: Dataset,
    plan: dict,
    baseline: Dataset | None = None,
    mode: str = "auto",
    client=None,
    model: str | None = None,
    max_rounds: int = MAX_TOOL_ROUNDS,
    generated_at: datetime | None = None,
) -> AgentResult:
    """Answer one question about a candidate run (optionally compared with a baseline run).

    mode: "demo" (no API), "llm" (OpenAI), or "auto" (LLM if a key/client is available, else demo).
    client: optional OpenAI-compatible client, used by tests to fake the API.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")
    if mode not in {"auto", "demo", "llm"}:
        raise ValueError(f"Unknown mode '{mode}'. Use 'auto', 'demo' or 'llm'.")
    model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    generated_at = generated_at or datetime.now(timezone.utc)
    ctx = ToolContext(dataset=dataset, plan=plan, baseline=baseline)
    notices: list[str] = []

    kind = classify_request(question)
    if kind == "control":
        result = AgentResult(question, CONTROL_REFUSAL, "refused", None,
                             notices=["Request asked for a control or configuration action; no tools were run."])
        result.report_markdown = build_report(result, dataset, baseline, plan, generated_at)
        return result
    if kind == "safety_judgement":
        notices.append(SAFETY_NOTICE)

    has_key = bool(os.getenv("OPENAI_API_KEY"))
    use_llm = mode == "llm" or (mode == "auto" and (client is not None or has_key))
    if use_llm and client is None and not has_key:
        notices.append("OPENAI_API_KEY is not set, so the deterministic demo mode was used instead.")
        use_llm = False

    answer, records, used_mode, used_model = "", [], "demo", None
    if use_llm:
        try:
            client = client or _make_openai_client()
            answer, records, llm_notices = _run_llm(question, ctx, client, model, max_rounds)
            notices += llm_notices
            used_mode, used_model = "llm", model
            if not answer:
                notices.append("The model returned an empty answer; see the measured values below.")
        except Exception as exc:  # API down, bad key, rate limit, timeout, malformed response...
            log.warning("LLM call failed: %s", exc, exc_info=True)
            notices.append(f"LLM API error ({type(exc).__name__}: {str(exc)[:300]}). "
                           "Fell back to the deterministic demo mode.")
            ctx.figures.clear()
            used_mode, use_llm = "demo (fallback)", False

    if not use_llm:
        planned = plan_demo_tools(question, has_baseline=baseline is not None)
        notes = []
        if baseline is None and _has(question.lower(), COMPARE_PATTERNS):
            notes.append("No baseline run is loaded, so no regression comparison was made. Load the previous revision "
                         "as the baseline to compare.")
        records = [run_tool(name, args, ctx) for name, args in planned]
        answer = compose_demo_answer(records, notes)

    reference = [[r.result for r in records], dataset.info(), baseline.info() if baseline else {}, plan, question]
    result = AgentResult(question=question, answer=answer, mode=used_mode, model=used_model, tool_calls=records,
                         figures=list(ctx.figures), notices=notices,
                         unverified_numbers=find_unverified_numbers(answer, reference))
    result.report_markdown = build_report(result, dataset, baseline, plan, generated_at)
    return result

"""Streamlit UI for PowerOps Agent.

This file only talks to the FastAPI backend over HTTP; all analysis happens
there. Start the backend first, then run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DEFAULT_API_URL = os.getenv("POWEROPS_API_URL", "http://127.0.0.1:8000")
MODES = {"Auto (LLM if key is set)": "auto", "Demo - no API key needed": "demo", "LLM": "llm"}
SLOTS = {"candidate": "Run under test", "baseline": "Baseline run (optional)"}
EXAMPLE_QUESTIONS = [
    "Generate the validation report for this run.",
    "Did this revision regress against the baseline?",
    "Does the 800 V bus recover fast enough from GPU load bursts?",
    "Did the battery port pick up the load during the grid sag?",
    "Can we trust the power-sensor data in this log?",
    "Is this revision ready to release?",
]

st.set_page_config(page_title="PowerOps Agent", layout="wide")
state = st.session_state
for key, default in (("candidate", None), ("baseline", None), ("messages", []), ("overview", None),
                     ("csv", {}), ("pending_question", None), ("auto_random_done", False)):
    state.setdefault(key, default)


# --------------------------------------------------------------------------- backend helper
def api(method: str, path: str, **kwargs):
    """Call the backend. Returns (json, None) on success or (None, error message)."""
    url = f"{state.get('api_url', DEFAULT_API_URL).rstrip('/')}{path}"
    try:
        response = httpx.request(method, url, timeout=180.0, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"Cannot reach the backend at {url} ({type(exc).__name__}). Is `uvicorn api.main:app` running?"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        if isinstance(detail, dict):
            detail = detail.get("message", "") + "\n\n" + "\n".join(f"- {e}" for e in detail.get("errors", []))
        elif isinstance(detail, list):  # FastAPI request-validation errors
            detail = "\n".join(f"- {d.get('loc', [''])[-1]}: {d.get('msg')}" for d in detail)
        return None, f"Backend returned {response.status_code}: {detail}"
    if "text/csv" in response.headers.get("content-type", ""):
        return response.content, None
    return response.json(), None


def to_figures(figures: list[dict]) -> list[tuple[str, object]]:
    return [(f["plot_id"], pio.from_json(f["figure_json"])) for f in figures]


def set_slot(slot: str, payload: dict) -> None:
    """Store a loaded dataset in a slot; refresh the overview plot and cached CSV."""
    state[slot] = payload
    state.messages = []
    state.csv[slot], _ = api("GET", f"/datasets/{payload['dataset_id']}/csv")
    if slot == "candidate":
        plots, _ = api("GET", f"/datasets/{payload['dataset_id']}/plots/overview")
        state.overview = to_figures(plots["plots"])[0][1] if plots and plots["plots"] else None


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    state.api_url = st.text_input("Backend URL", value=state.get("api_url", DEFAULT_API_URL))
    health, health_err = api("GET", "/health")
    if health_err:
        st.error(health_err)
        st.code("uvicorn api.main:app --reload --port 8000", language="bash")
    else:
        st.success("Backend connected")
        st.caption(f"LLM API key configured: **{'yes' if health['llm_configured'] else 'no'}** · model `{health['model']}`")
    mode_label = st.radio("Agent mode", list(MODES), help="Demo mode is deterministic and never calls an API.")
    if health and not health["llm_configured"] and MODES[mode_label] != "demo":
        st.caption("No OPENAI_API_KEY on the backend, so requests will use demo mode.")
    with st.expander("Validation plan (illustrative)"):
        plan, plan_err = api("GET", "/plan") if not health_err else (None, health_err)
        if plan:
            st.caption(plan.get("_notice", ""))
            st.dataframe(pd.DataFrame([{"test": t["id"], "type": t["type"], "step": t["step"],
                                        "description": t.get("description", "")} for t in plan["tests"]]),
                         hide_index=True)
        else:
            st.write(plan_err)
    auto_random = st.toggle("New random simulated run on each page load", value=True,
                            help="Refreshing the page starts a new session and loads a different random unit, "
                                 "unless you upload your own data.")
    if st.button("Clear chat", width="stretch"):
        state.messages = []

# --------------------------------------------------------------------------- header
st.title("PowerOps Agent")
st.caption("Draft SST validation reports and revision regression checks from **synthetic** 800 V DC bench logs.")
st.warning(
    "Independent student prototype - not an official product of any company. Synthetic data only. Acceptance "
    "criteria are illustrative, not product specifications. Read-only: no hardware control, and no release or "
    "safety decisions.",
)
if health_err:
    st.stop()

# A new browser session (e.g. a page refresh) gets a fresh random unit, unless the user turned that off.
if auto_random and state.candidate is None and not state.auto_random_done:
    state.auto_random_done = True
    payload, err = api("POST", "/datasets/random")
    if err:
        st.error(err)
    else:
        set_slot("candidate", payload)

tab_data, tab_chat = st.tabs(["1 · Load runs", "2 · Validate with the agent"])

# --------------------------------------------------------------------------- data tab
with tab_data:
    st.subheader("Quick start")
    if st.button("Load demo pair: rev-A baseline + rev-B candidate", type="primary"):
        pair, err = api("POST", "/datasets/demo-pair")
        if err:
            st.error(err)
        else:
            set_slot("baseline", pair["baseline"])
            set_slot("candidate", pair["candidate"])
    if st.button("New random run", help="A unit with randomly drawn behaviour; may pass, fail, or have a sensor fault."):
        payload, err = api("POST", "/datasets/random")
        if err:
            st.error(err)
        else:
            set_slot("candidate", payload)
    st.caption("All runs are synthetic and follow the same 8-step validation sequence. A random run is a new unit "
               "every time: its parameters are hidden from the agent, so it has to find any failures itself.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Generate a synthetic run")
        with st.form("synthetic"):
            revision = st.selectbox("Revision behaviour", ["A", "B", "C"],
                                    format_func=lambda r: {"A": "rev-A (baseline: passes)", "B": "rev-B (candidate: regressions)",
                                                           "C": "rev-C (fixes rev-B, slow battery pickup)"}[r])
            seed = st.number_input("Random seed", value=42, step=1)
            fault = st.checkbox("Inject a battery power-sensor fault (+3 kW offset)")
            slot = st.radio("Load as", list(SLOTS), format_func=SLOTS.get, horizontal=True, key="gen_slot")
            if st.form_submit_button("Generate synthetic run"):
                payload, err = api("POST", "/datasets/synthetic",
                                   json={"revision": revision, "seed": int(seed), "battery_sensor_fault": fault})
                if err:
                    st.error(err)
                else:
                    set_slot(slot, payload)
    with right:
        st.subheader("Upload a CSV")
        st.caption("Required columns: `timestamp_s, test_step, grid_voltage_pu, grid_power_kw, battery_power_kw, "
                   "rack_power_kw, dc_bus_voltage_v, sic_temp_c`. Lines starting with `#` are comments; "
                   "`# key: value` lines become run metadata.")
        upload = st.file_uploader("CSV bench log", type=["csv"])
        up_slot = st.radio("Load as", list(SLOTS), format_func=SLOTS.get, horizontal=True, key="up_slot")
        if st.button("Validate & load", disabled=upload is None):
            payload, err = api("POST", "/datasets/upload", files={"file": (upload.name, upload.getvalue(), "text/csv")})
            if err:
                st.error(err)
            else:
                set_slot(up_slot, payload)

    st.divider()
    cols = st.columns(2, gap="large")
    for col, slot in zip(cols, SLOTS):
        ds = state[slot]
        with col:
            st.markdown(f"**{SLOTS[slot]}**")
            if not ds:
                st.caption("Nothing loaded.")
                continue
            st.markdown(f"`{ds['label']}`  \n{ds['name']}")
            st.caption("SYNTHETIC DATA - not real measurements." if ds["is_synthetic"]
                       else "User-supplied file. Provenance is not verified by this tool.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Samples", f"{ds['n_samples']:,}")
            c2.metric("Duration", f"{ds['duration_s']} s")
            c3.metric("Test steps", len(ds["test_steps"]))
            for w in ds["warnings"]:
                st.warning(w)
            if state.csv.get(slot):
                st.download_button("Download CSV", state.csv[slot], file_name=ds["name"], mime="text/csv", key=f"csv_{slot}")
    if state.candidate:
        with st.expander("First 10 rows of the run under test"):
            st.dataframe(pd.DataFrame(state.candidate["preview"]), hide_index=True)
        if state.overview is not None:
            st.plotly_chart(state.overview, width="stretch", theme=None, key="overview_chart")
        st.success("Runs loaded. Open **2 · Validate with the agent**.")


# --------------------------------------------------------------------------- chat tab
def render_assistant(i: int, msg: dict) -> None:
    p = msg["payload"]
    mode = p["mode"] + (f" · {p['model']}" if p.get("model") else "")
    st.caption(f"Mode: **{mode}** · {len(p['tool_calls'])} tool call(s)")
    for note in p["notices"]:
        st.warning(note)
    st.markdown(p["answer"])
    if p["unverified_numbers"]:
        st.error("⚠ Numbers in this answer that do not match any tool output: "
                 + ", ".join(f"`{n}`" for n in p["unverified_numbers"]) + ". Treat them with caution.")
    elif p["tool_calls"]:
        st.caption("✓ Every number in this answer matches a tool output.")
    for j, (plot_id, fig) in enumerate(msg["figures"]):
        st.plotly_chart(fig, width="stretch", theme=None, key=f"fig_{i}_{j}_{plot_id}")
    if p["tool_calls"]:
        with st.expander(f"Tool calls ({len(p['tool_calls'])}): exact arguments and JSON results"):
            for k, call in enumerate(p["tool_calls"], start=1):
                st.markdown(f"**{k}. `{call['name']}`** · arguments `{call['arguments']}` · {'ok' if call['ok'] else 'error'}")
                st.json(call["result"], expanded=False)
    col_a, col_b = st.columns([1, 3])
    col_a.download_button("Download draft report (.md)", p["report_markdown"],
                          file_name=f"sst_validation_report_{i // 2 + 1}.md", mime="text/markdown", key=f"dl_{i}")
    with col_b.expander("Preview report"):
        st.markdown(p["report_markdown"])


with tab_chat:
    if not state.candidate:
        st.info("Load a run under test in **1 · Load runs** first (the demo pair is quickest).")
    else:
        base = state.baseline
        st.caption(f"Run under test: `{state.candidate['label']}` · baseline: "
                   f"`{base['label'] if base else 'none'}` · mode: {mode_label}")
        st.caption("Try an example:")
        with st.container(horizontal=True, wrap=True, gap="small"):
            for q in EXAMPLE_QUESTIONS:
                if st.button(q, width="content"):
                    state.pending_question = q

        for i, msg in enumerate(state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_assistant(i, msg)

        typed = st.chat_input("Ask about efficiency, GPU bursts, grid sag, thermal, regressions or sensor integrity...")
        question = typed or state.pending_question
        if question:
            state.pending_question = None
            state.messages.append({"role": "user", "content": question})
            with st.spinner("Agent is running the validation tools..."):
                payload, err = api("POST", "/chat", json={
                    "dataset_id": state.candidate["dataset_id"],
                    "baseline_dataset_id": base["dataset_id"] if base else None,
                    "question": question, "mode": MODES[mode_label]})
            if err:
                state.messages.pop()
                st.error(err)
            else:
                state.messages.append({"role": "assistant", "payload": payload, "figures": to_figures(payload["figures"])})
                st.rerun()

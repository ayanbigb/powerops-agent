"""Agent tests. The OpenAI client is replaced by a scripted fake, so the
tool-calling loop is exercised exactly as in production but offline."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from powerops.agent import classify_request, plan_demo_tools, run_agent


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def reply(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class FakeClient:
    """Mimics client.chat.completions.create(). Each script step gets the messages and returns a reply."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(kwargs["messages"]) if callable(step) else step


def tool_results(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


# ----------------------------------------------------------------- classification + demo planner
@pytest.mark.parametrize(
    "question, kind",
    [
        ("Flash the new firmware to the unit", "control"),
        ("Increase the current limit to 70 A", "control"),
        ("Can you shut down the SST?", "control"),
        ("Please write register 40001 over Modbus", "control"),
        ("Did the battery pick up the load during the sag?", "analysis"),
        ("How did the bus voltage change after each burst?", "analysis"),
        ("Is rev B ready to release?", "safety_judgement"),
        ("Can I sign off on this run?", "safety_judgement"),
    ],
)
def test_classify_request(question, kind):
    assert classify_request(question) == kind


def test_demo_planner_maps_questions_to_tools():
    names = lambda q, b=True: [n for n, _ in plan_demo_tools(q, b)]
    assert names("Generate the validation report")[:3] == ["run_validation_plan", "check_power_balance", "compare_to_baseline"]
    assert "compare_to_baseline" not in names("Generate the validation report", b=False)
    assert plan_demo_tools("How does the bus handle GPU bursts?", True) == [
        ("run_validation_plan", {"test_types": ["load_burst"]}), ("generate_plots", {"plot_type": "burst"})]
    assert names("What is the efficiency at light load?") == ["check_power_balance", "run_validation_plan", "generate_plots"]
    assert names("Did it regress vs the baseline?") == ["compare_to_baseline", "generate_plots"]
    assert names("Did it regress?", b=False) == ["run_validation_plan", "generate_plots"]
    assert names("Can we trust the power sensors?") == ["check_power_balance"]


# ----------------------------------------------------------------- demo mode
def test_demo_report_is_deterministic_verified_and_finds_the_regressions(plan, run_a, run_b):
    a = run_agent("Generate the validation report", run_b, plan, baseline=run_a, mode="demo")
    b = run_agent("Generate the validation report", run_b, plan, baseline=run_a, mode="demo")
    assert a.mode == "demo" and a.answer == b.answer and a.unverified_numbers == []
    assert "4 of 7 tests PASS" in a.answer and "new failures: EFF-25, BURST-1, THERM-1" in a.answer
    assert [f["plot_id"] for f in a.figures] == ["efficiency_curve", "burst_BURST-1", "grid_sag_SAG-1", "thermal_THERM-1"]


def test_demo_warns_when_sensor_integrity_is_suspect(plan, run_fault):
    r = run_agent("Generate the validation report", run_fault, plan, mode="demo")
    assert "Data integrity (power sensors): SUSPECT" in r.answer
    assert "efficiency verdicts above may be caused by a measurement problem" in r.answer


def test_missing_baseline_is_explained(plan, run_b):
    r = run_agent("Did this revision regress?", run_b, plan, mode="demo")
    assert "No baseline run is loaded" in r.answer and r.tool_calls[0].name == "run_validation_plan"


def test_control_requests_are_refused_without_running_tools(plan, run_b):
    r = run_agent("Flash the new firmware to the unit", run_b, plan, mode="demo")
    assert r.mode == "refused" and r.tool_calls == [] and "read-only" in r.answer
    assert "# SST Validation Report (DRAFT)" in r.report_markdown


def test_release_questions_get_analysis_plus_a_sign_off_notice(plan, run_a, run_b):
    r = run_agent("Is rev B ready to release?", run_b, plan, baseline=run_a, mode="demo")
    assert any("does not decide whether a revision is safe, ready to release" in n for n in r.notices)
    assert r.tool_calls and "Release decisions and sign-off are made by qualified engineers" in r.answer


def test_bad_inputs_and_missing_key(plan, run_b):
    with pytest.raises(ValueError):
        run_agent("  ", run_b, plan)
    with pytest.raises(ValueError):
        run_agent("hi", run_b, plan, mode="turbo")
    r = run_agent("Any limit violations?", run_b, plan, mode="llm")
    assert r.mode == "demo" and any("OPENAI_API_KEY is not set" in n for n in r.notices)


# ----------------------------------------------------------------- LLM mode (fake client)
def test_llm_tool_calling_loop_and_number_check(plan, run_a, run_b):
    def first(messages):
        return reply(tool_calls=[tool_call("c1", "run_validation_plan", json.dumps({"test_ids": ["BURST-1"]})),
                                 tool_call("c2", "compare_to_baseline", json.dumps({"test_ids": ["BURST-1"]}))])

    def second(messages):
        burst = tool_results(messages)[0]["tests"][0]
        dev = next(k for k in burst["kpis"] if k["name"] == "max_abs_bus_deviation_v")["measured"]
        return reply(content=f"- BURST-1 FAILS: the 800 V bus deviated {dev} V on a GPU load edge "
                             f"(illustrative limit 16.0 V).\n- Estimated field failure rate: 0.37%.")

    client = FakeClient([first, second])
    r = run_agent("Why did BURST-1 fail?", run_b, plan, baseline=run_a, mode="llm", client=client, model="fake-model")
    assert r.mode == "llm" and r.model == "fake-model"
    assert [c.name for c in r.tool_calls] == ["run_validation_plan", "compare_to_baseline"]
    assert r.unverified_numbers == ["0.37"]  # the invented figure is caught; the real deviation passes
    first_req, second_req = client.requests
    assert [t["function"]["name"] for t in first_req["tools"]][1] == "run_validation_plan"
    assert '"baseline_run"' in first_req["messages"][1]["content"]  # the model is told a baseline exists
    assert [m["tool_call_id"] for m in second_req["messages"] if m["role"] == "tool"] == ["c1", "c2"]
    # The model never receives raw samples: a mid-run bus voltage reading is nowhere in the conversation.
    raw = f"{run_b.df['dc_bus_voltage_v'].iloc[12345]}"
    assert raw not in json.dumps(client.requests)


def test_llm_bad_tool_calls_are_reported_back(plan, run_b):
    def first(messages):
        return reply(tool_calls=[tool_call("a", "run_validation_plan", '{"test_ids": ["EFF-999"]}'),
                                 tool_call("b", "summarize_run", "{oops"),
                                 tool_call("c", "flash_firmware", "{}")])

    def second(messages):
        assert len([r for r in tool_results(messages) if "error" in r]) == 3
        return reply(content="I could not run the requested analysis.")

    r = run_agent("Check it", run_b, plan, mode="llm", client=FakeClient([first, second]))
    assert r.mode == "llm" and [c.ok for c in r.tool_calls] == [False, False, False]


@pytest.mark.parametrize("error", [
    openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
    openai.RateLimitError("rate limited", response=httpx.Response(429, request=httpx.Request("POST", "https://x")), body=None),
    AttributeError("malformed response"),
])
def test_api_errors_fall_back_to_demo_mode(plan, run_b, error):
    r = run_agent("Generate the validation report", run_b, plan, mode="llm", client=FakeClient([error]))
    assert r.mode == "demo (fallback)" and any("Fell back" in n for n in r.notices)
    assert r.tool_calls and r.unverified_numbers == []


def test_tool_round_limit_forces_a_final_answer(plan, run_b):
    loop = lambda messages: reply(tool_calls=[tool_call(f"id{len(messages)}", "summarize_run", "{}")])
    client = FakeClient([loop, loop, reply(content="Done.")])
    r = run_agent("Summarize", run_b, plan, mode="llm", client=client, max_rounds=2)
    assert r.answer == "Done." and len(r.tool_calls) == 2 and client.requests[-1]["tool_choice"] == "none"

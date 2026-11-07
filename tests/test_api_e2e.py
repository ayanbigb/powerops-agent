"""End-to-end tests through the FastAPI backend, in demo mode (no API key)."""

import plotly.io as pio
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from powerops.synthetic import SyntheticConfig, synthetic_csv_bytes


@pytest.fixture
def client():
    return TestClient(api_main.app)


def upload(client, cfg, name):
    res = client.post("/datasets/upload", files={"file": (name, synthetic_csv_bytes(cfg), "text/csv")})
    assert res.status_code == 200, res.text
    return res.json()


def test_end_to_end_validation_report_with_regression_check(client):
    # 1) Upload a baseline (rev-A) and a candidate (rev-B), exactly as the UI does.
    base = upload(client, SyntheticConfig(revision="A"), "revA.csv")
    cand = upload(client, SyntheticConfig(revision="B", seed=43), "revB.csv")
    assert cand["is_synthetic"] and cand["label"] == "SST-DEMO-001 rev-B (candidate)" and cand["test_steps"] == list(range(8))

    # 2) Ask for the validation report in deterministic demo mode.
    res = client.post("/chat", json={"dataset_id": cand["dataset_id"], "baseline_dataset_id": base["dataset_id"],
                                     "question": "Generate the validation report", "mode": "demo"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["mode"] == "demo" and body["unverified_numbers"] == []
    calls = {c["name"]: c["result"] for c in body["tool_calls"]}
    assert {"run_validation_plan", "check_power_balance", "compare_to_baseline", "generate_plots"} <= set(calls)

    # 3) The agent reached the right engineering conclusions.
    assert calls["run_validation_plan"]["summary"]["failed_tests"] == ["EFF-25", "BURST-1", "THERM-1"]
    assert calls["compare_to_baseline"]["summary"]["new_failures"] == ["EFF-25", "BURST-1", "THERM-1"]
    assert calls["check_power_balance"]["verdict"] == "CONSISTENT"

    # 4) Plots arrive as Plotly JSON the UI can render.
    assert len(body["figures"]) == 4
    for f in body["figures"]:
        pio.from_json(f["figure_json"])

    # 5) The draft report has every section, ending in a blank sign-off block.
    report = body["report_markdown"]
    for heading in ["# SST Validation Report (DRAFT)", "## Summary", "## Number check", "## Validation scorecard",
                    "## Regression vs baseline", "## Data integrity (power sensors)", "## Methods", "## Limitations",
                    "## Tool-call trace", "## Engineering review and sign-off"]:
        assert heading in report, heading
    assert "| BURST-1 | ✕ FAIL | max_abs_bus_deviation_v |" in report
    assert "| Reviewed by | | | |" in report and "SYNTHETIC" in report


def test_demo_pair_and_sensor_fault_endpoints(client):
    pair = client.post("/datasets/demo-pair").json()
    assert pair["baseline"]["label"].endswith("rev-A (baseline)") and pair["candidate"]["label"].endswith("rev-B (candidate)")
    fault = client.post("/datasets/synthetic", json={"revision": "A", "battery_sensor_fault": True}).json()
    body = client.post("/chat", json={"dataset_id": fault["dataset_id"], "question": "Can we trust the sensors?",
                                      "mode": "demo"}).json()
    assert body["tool_calls"][0]["result"]["verdict"] == "SUSPECT"


def test_plot_endpoint_with_baseline(client):
    pair = client.post("/datasets/demo-pair").json()
    cid, bid = pair["candidate"]["dataset_id"], pair["baseline"]["dataset_id"]
    res = client.get(f"/datasets/{cid}/plots/efficiency_curve", params={"baseline_id": bid}).json()
    assert res["status"] == "ok" and res["plots"][0]["plot_id"] == "efficiency_curve"
    assert client.get(f"/datasets/{cid}/plots/pie").status_code == 404


@pytest.mark.parametrize("content, fragment", [
    (b"", "File is empty"),
    (b"time,volts\n1,2\n", "Missing required column"),
    (b"timestamp_s,test_step,grid_voltage_pu,grid_power_kw,battery_power_kw,rack_power_kw,dc_bus_voltage_v,sic_temp_c\n",
     "no data rows"),
])
def test_bad_uploads_return_readable_422(client, content, fragment):
    res = client.post("/datasets/upload", files={"file": ("bad.csv", content, "text/csv")})
    assert res.status_code == 422 and fragment in " ".join(res.json()["detail"]["errors"])


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 100)
    assert client.post("/datasets/upload", files={"file": ("big.csv", b"x" * 500, "text/csv")}).status_code == 413


def test_request_errors_and_refusal(client):
    ds = client.post("/datasets/synthetic", json={}).json()
    did = ds["dataset_id"]
    assert client.post("/chat", json={"dataset_id": "missing", "question": "hi"}).status_code == 404
    assert client.post("/chat", json={"dataset_id": did, "baseline_dataset_id": "nope", "question": "hi"}).status_code == 404
    assert client.post("/chat", json={"dataset_id": did, "baseline_dataset_id": did, "question": "hi"}).status_code == 422
    assert client.post("/chat", json={"dataset_id": did, "question": ""}).status_code == 422
    assert client.post("/datasets/synthetic", json={"revision": "Z"}).status_code == 422
    refused = client.post("/chat", json={"dataset_id": did, "question": "Flash the new firmware to the unit"}).json()
    assert refused["mode"] == "refused" and refused["tool_calls"] == []


def test_health_plan_and_broken_config(client, monkeypatch, tmp_path):
    assert client.get("/health").json()["llm_configured"] is False
    assert "ILLUSTRATIVE" in client.get("/plan").json()["_notice"]
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    monkeypatch.setenv("POWEROPS_PLAN_PATH", str(bad))
    res = client.get("/plan")
    assert res.status_code == 500 and "not valid JSON" in res.json()["detail"]


def test_random_dataset_endpoint(client):
    a = client.post("/datasets/random").json()
    b = client.post("/datasets/random").json()
    assert a["is_synthetic"] and a["metadata"]["revision"].startswith("random-sim")
    assert a["name"] != b["name"]  # a new unit each time
    seeded = [client.post("/datasets/random", json={"seed": 5}).json() for _ in range(2)]
    assert seeded[0]["preview"] == seeded[1]["preview"] and "seed5" in seeded[0]["name"]
    report = client.post("/chat", json={"dataset_id": a["dataset_id"], "question": "Generate the validation report",
                                        "mode": "demo"}).json()
    assert report["mode"] == "demo" and report["unverified_numbers"] == []

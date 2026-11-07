"""UI smoke tests: run the real Streamlit script with HTTP calls routed into the FastAPI app."""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture
def backend(monkeypatch):
    client = TestClient(app)

    def fake_request(method, url, timeout=None, **kwargs):
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        return client.request(method, path, **kwargs)

    monkeypatch.setattr(httpx, "request", fake_request)


def test_ui_demo_pair_then_validation_report(backend):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and any("Backend connected" in s.value for s in at.success)
    # A new session starts with a fresh random simulated unit already loaded.
    assert at.session_state["candidate"]["metadata"]["revision"].startswith("random-sim")

    at.radio[0].set_value("Demo - no API key needed").run()
    next(b for b in at.button if b.label.startswith("Load demo pair")).click().run()
    assert not at.exception
    assert at.session_state["candidate"]["label"].endswith("rev-B (candidate)")
    assert at.session_state["baseline"]["label"].endswith("rev-A (baseline)")
    assert at.session_state["overview"] is not None

    at.chat_input[0].set_value("Generate the validation report for this run.").run()
    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    payload = messages[1]["payload"]
    assert payload["mode"] == "demo"
    assert "compare_to_baseline" in [c["name"] for c in payload["tool_calls"]]
    assert any("Every number in this answer matches" in c.value for c in at.caption)


def test_ui_reports_backend_down_without_crashing(monkeypatch):
    def down(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "request", down)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and any("Cannot reach the backend" in e.value for e in at.error)

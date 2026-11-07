import numpy as np
import pandas as pd
import pytest

from powerops.config import load_plan
from powerops.data_io import load_csv
from powerops.synthetic import SyntheticConfig, synthetic_csv_bytes


@pytest.fixture(scope="session")
def plan():
    return load_plan()


@pytest.fixture(scope="session")
def run_a():
    return load_csv(synthetic_csv_bytes(SyntheticConfig(revision="A", seed=42)), name="SYNTHETIC_revA.csv")


@pytest.fixture(scope="session")
def run_b():
    return load_csv(synthetic_csv_bytes(SyntheticConfig(revision="B", seed=43)), name="SYNTHETIC_revB.csv")


@pytest.fixture(scope="session")
def run_fault():
    cfg = SyntheticConfig(revision="A", seed=44, battery_sensor_offset_kw=3.0)
    return load_csv(synthetic_csv_bytes(cfg), name="SYNTHETIC_revA_fault.csv")


@pytest.fixture(autouse=True)
def no_openai_key(monkeypatch):
    """Tests must never call the real API, even if the developer has a key exported."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
    monkeypatch.delenv("POWEROPS_PLAN_PATH", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    # The app modules call load_dotenv(); stop them re-reading the developer's real key from .env.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


def make_df(n=1000, dt=0.001, step=1, rack=50.0, loss=1.0, bus=800.0, temp=60.0, t0=0.0) -> pd.DataFrame:
    """A flat, noise-free run in one test step that tests modify to create known features."""
    return pd.DataFrame({
        "timestamp_s": t0 + np.arange(n) * dt,
        "test_step": np.full(n, step),
        "grid_voltage_pu": np.full(n, 1.0),
        "grid_power_kw": np.full(n, rack + loss),
        "battery_power_kw": np.zeros(n),
        "rack_power_kw": np.full(n, rack),
        "dc_bus_voltage_v": np.full(n, bus),
        "sic_temp_c": np.full(n, temp),
    })


def csv_of(df: pd.DataFrame, header: str = "") -> bytes:
    return (header + df.to_csv(index=False)).encode()


def plan_test(plan: dict, test_id: str) -> dict:
    return next(t for t in plan["tests"] if t["id"] == test_id)

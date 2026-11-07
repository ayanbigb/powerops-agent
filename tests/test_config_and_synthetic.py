import json

import pytest

from powerops.config import ConfigError, load_plan
from powerops.synthetic import SEQUENCE, SyntheticConfig, generate_synthetic_run, synthetic_csv_bytes


def _clean(plan):
    return json.loads(json.dumps({k: v for k, v in plan.items() if not k.startswith("_")}))


# ----------------------------------------------------------------- plan config
def test_example_plan_loads_and_is_labelled_illustrative(plan):
    assert "ILLUSTRATIVE" in plan["_notice"] and "not a real spec" in plan["profile_name"]
    assert [t["id"] for t in plan["tests"]] == ["EFF-25", "EFF-50", "EFF-75", "EFF-100", "BURST-1", "SAG-1", "THERM-1"]
    assert plan["_source_path"] == "config/validation_plan.example.json"


def test_env_var_selects_plan_file(tmp_path, monkeypatch, plan):
    custom = _clean(plan)
    custom["profile_name"] = "my bench plan"
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(custom))
    monkeypatch.setenv("POWEROPS_PLAN_PATH", str(path))
    assert load_plan()["profile_name"] == "my bench plan"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["channels"]["dc_bus_voltage_v"].update(min=900.0), "min must be less than max"),
        (lambda c: c["channels"].update(bogus={"max": 1}), "unknown channel"),
        (lambda c: c["tests"][0].update(type="vibration"), "unknown type"),
        (lambda c: c["tests"][1].update(id="EFF-25"), "duplicate test id"),
        (lambda c: c["tests"][4].pop("max_recovery_s"), "max_recovery_s must be a positive number"),
        (lambda c: c["tests"][0].update(step="one"), "'step' must be a non-negative integer"),
        (lambda c: c.pop("regression"), "'regression' section is missing"),
        (lambda c: c.update(rated_power_kw=0), "rated_power_kw"),
    ],
)
def test_invalid_plans_are_rejected(tmp_path, plan, mutate, message):
    cfg = _clean(plan)
    mutate(cfg)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ConfigError, match=message):
        load_plan(path)


def test_missing_and_malformed_plan_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_plan(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_plan(bad)


# ----------------------------------------------------------------- synthetic generator
def test_generator_is_deterministic_and_follows_the_sequence():
    a = generate_synthetic_run(SyntheticConfig(seed=7))
    assert a.equals(generate_synthetic_run(SyntheticConfig(seed=7)))
    assert not a.equals(generate_synthetic_run(SyntheticConfig(seed=8)))
    assert len(a) == int(sum(d for _, _, d, _ in SEQUENCE) * 1000)
    assert sorted(a["test_step"].unique()) == [s for s, *_ in SEQUENCE]


def test_csv_is_clearly_labelled_synthetic():
    first = synthetic_csv_bytes(SyntheticConfig(revision="B")).decode().splitlines()[:4]
    assert first[0].startswith("# SYNTHETIC DATA") and "NOT real measurements" in first[0]
    assert first[2] == "# unit_id: SST-DEMO-001" and first[3] == "# revision: rev-B (candidate)"


def test_revision_b_loses_more_power_than_a():
    a = generate_synthetic_run(SyntheticConfig(revision="A", noise_kw=0.0))
    b = generate_synthetic_run(SyntheticConfig(revision="B", noise_kw=0.0))
    step1 = a["test_step"] == 1
    loss_a = (a.grid_power_kw - a.rack_power_kw)[step1].mean()
    loss_b = (b.grid_power_kw - b.rack_power_kw)[step1].mean()
    assert loss_a == pytest.approx(0.8 + 8e-5 * 50**2, abs=1e-3)
    assert loss_b > loss_a + 0.5


def test_sensor_fault_offsets_the_battery_reading():
    df = generate_synthetic_run(SyntheticConfig(battery_sensor_offset_kw=3.0, noise_kw=0.0))
    assert df.loc[df.test_step == 1, "battery_power_kw"].eq(3.0).all()


def test_grid_sag_is_where_the_plan_expects_it():
    df = generate_synthetic_run(SyntheticConfig(noise_pu=0.0))
    sag = df[df.grid_voltage_pu < 0.9]
    assert set(sag.test_step) == {6} and len(sag) == 200


@pytest.mark.parametrize("cfg", [SyntheticConfig(revision="Z"), SyntheticConfig(sample_rate_hz=10)])
def test_invalid_generator_settings_raise(cfg):
    with pytest.raises(ValueError):
        generate_synthetic_run(cfg)


# ----------------------------------------------------------------- random simulated units
def test_random_config_is_reproducible_with_a_seed_and_new_without():
    from powerops.synthetic import random_config

    assert random_config(7) == random_config(7)
    assert random_config(7).custom_revision != random_config(8).custom_revision
    a, b = random_config(), random_config()
    assert a.seed != b.seed  # no seed -> a new unit every call (collision chance 1e-6)


def test_random_units_vary_between_pass_and_fail(plan):
    from powerops import analysis
    from powerops.data_io import load_csv
    from powerops.synthetic import random_config

    failed = [analysis.run_validation_plan(load_csv(synthetic_csv_bytes(random_config(s))).df, plan)["summary"]["failed_tests"]
              for s in range(12)]
    assert any(not f for f in failed) and any(f for f in failed)


def test_random_parameters_are_hidden_from_run_metadata():
    from powerops.data_io import load_csv
    from powerops.synthetic import random_config

    cfg = random_config(123)
    ds = load_csv(synthetic_csv_bytes(cfg))
    assert ds.label == "SST-SIM-000123 random-sim #123"
    assert set(ds.metadata) == {"unit_id", "revision", "test_sequence"}  # no device parameters, no fault flag
    assert str(cfg.custom_revision.fixed_loss_kw) not in str(ds.info())

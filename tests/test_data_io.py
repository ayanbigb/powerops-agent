import numpy as np
import pytest

from powerops.data_io import REQUIRED_COLUMNS, DataValidationError, load_csv

from .conftest import csv_of, make_df

HEADER = ",".join(REQUIRED_COLUMNS) + "\n"
ROW = "{t},1,1.0,51,0,50,800,60\n"


def rows(*times) -> str:
    return "".join(ROW.format(t=t) for t in times)


def test_synthetic_run_loads_with_metadata(run_b):
    assert run_b.is_synthetic
    assert len(run_b.df) == 26_000
    assert list(run_b.df.columns) == REQUIRED_COLUMNS
    assert run_b.metadata["unit_id"] == "SST-DEMO-001"
    assert run_b.label == "SST-DEMO-001 rev-B (candidate)"
    assert run_b.info()["test_steps"] == list(range(8))
    assert "generator_config" not in run_b.metadata  # too long to be metadata


def test_plain_csv_is_not_synthetic_and_uses_file_name_as_label():
    ds = load_csv(csv_of(make_df(10)), name="bench.csv")
    assert not ds.is_synthetic and ds.warnings == [] and ds.label == "bench.csv"


def test_metadata_comments_are_parsed():
    ds = load_csv(csv_of(make_df(10), header="# unit_id: SN-7\n# revision: fw 2.1\n"))
    assert ds.metadata == {"unit_id": "SN-7", "revision": "fw 2.1"} and ds.label == "SN-7 fw 2.1"


def test_load_from_path(tmp_path):
    path = tmp_path / "run.csv"
    path.write_bytes(csv_of(make_df(10)))
    assert load_csv(path).name == "run.csv"


@pytest.mark.parametrize(
    "content, message",
    [
        (b"", "empty"),
        (b"# only a comment\n", "no header"),
        (HEADER.encode(), "no data rows"),
        (b"a,b\n1,2\n", "Missing required column"),
        ((HEADER + rows(0) + "0.001,1,1,51,0,50,800,60,EXTRA,FIELDS\n").encode(), "Malformed CSV"),
        (b"\x89PNG\r\n\x1a\n\x00\x00\xff\xfe", "not UTF-8"),
        ((HEADER + rows(0)).encode(), "at least 2 samples"),
        ((HEADER + rows("x", "y")).encode(), "No rows have a valid timestamp_s"),
    ],
)
def test_fatal_problems_raise_clear_errors(content, message):
    with pytest.raises(DataValidationError) as err:
        load_csv(content)
    assert message.lower() in str(err.value).lower()


def test_missing_test_step_column_is_named():
    content = HEADER.replace("test_step,", "").encode() + b"0,1,51,0,50,800,60\n"
    with pytest.raises(DataValidationError, match="test_step"):
        load_csv(content)


def test_semicolon_delimited_file_gets_a_hint():
    with pytest.raises(DataValidationError, match="delimiter"):
        load_csv(HEADER.replace(",", ";").encode() + b"0;1;1;51;0;50;800;60\n")


def test_non_numeric_missing_unsorted_and_duplicate_rows():
    content = HEADER + rows(0.002) + "0,1,1.0,51,0,50,oops,60\n" + "0.001,1,1.0,51,0,50,800,\n" + rows(0.001)
    ds = load_csv(content.encode())
    assert ds.df["timestamp_s"].tolist() == [0.0, 0.001, 0.002]
    text = " ".join(ds.warnings)
    for fragment in ("non-numeric", "sorted", "duplicate", "dc_bus_voltage_v: 1 missing"):
        assert fragment in text


def test_bad_test_step_values_are_warned():
    df = make_df(20)
    df.loc[0:4, "test_step"] = np.nan
    df.loc[5, "test_step"] = 1.5
    ds = load_csv(csv_of(df))
    assert ds.df["test_step"].isna().sum() == 6
    assert any("no step" in w for w in ds.warnings) and any("non-integer" in w for w in ds.warnings)


def test_channel_without_data_is_a_warning():
    df = make_df(20)
    df["sic_temp_c"] = np.nan
    assert any("sic_temp_c: no valid measurements" in w for w in load_csv(csv_of(df)).warnings)

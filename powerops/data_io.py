"""Load a bench validation log and check that it matches the expected schema.

One row per sample. Required columns:
    timestamp_s       time since start of run (s)
    test_step         which step of the validation sequence was running (integer, from the test sequencer)
    grid_voltage_pu   AC grid-port voltage, per-unit of nominal
    grid_power_kw     power drawn from the grid port
    battery_power_kw  battery-port power: + discharging into the DC bus, - charging
    rack_power_kw     power delivered to the 800 V DC rack/load port
    dc_bus_voltage_v  800 V DC bus voltage
    sic_temp_c        SiC power-module temperature

Lines starting with '#' are comments. Comment lines of the form '# key: value'
(for example '# revision: rev-B') are kept as run metadata.

Fatal problems raise DataValidationError; non-fatal ones become warnings.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLUMN = "timestamp_s"
STEP_COLUMN = "test_step"
MEASUREMENT_COLUMNS = [
    "grid_voltage_pu",
    "grid_power_kw",
    "battery_power_kw",
    "rack_power_kw",
    "dc_bus_voltage_v",
    "sic_temp_c",
]
REQUIRED_COLUMNS = [TIME_COLUMN, STEP_COLUMN, *MEASUREMENT_COLUMNS]
UNITS = {
    "timestamp_s": "s",
    "test_step": "",
    "grid_voltage_pu": "pu",
    "grid_power_kw": "kW",
    "battery_power_kw": "kW",
    "rack_power_kw": "kW",
    "dc_bus_voltage_v": "V",
    "sic_temp_c": "°C",
}

SYNTHETIC_MARKER = "SYNTHETIC"
MAX_ROWS = 2_000_000
_META_RE = re.compile(r"^#\s*([a-z_]{2,40})\s*:\s*(.{1,200})$")


class DataValidationError(ValueError):
    """The file cannot be analysed. `errors` lists every reason found."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class Dataset:
    """A validated test log plus facts about where it came from."""

    df: pd.DataFrame
    name: str
    is_synthetic: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Short human name, e.g. 'SST-DEMO-001 rev-B (candidate)'."""
        parts = [self.metadata.get("unit_id"), self.metadata.get("revision")]
        return " ".join(p for p in parts if p) or self.name

    @property
    def short_label(self) -> str:
        """Revision name for chart labels, e.g. 'rev-B (candidate)'."""
        return self.metadata.get("revision") or self.label

    def info(self) -> dict:
        """Small JSON-friendly description (no raw samples)."""
        t = self.df[TIME_COLUMN].to_numpy()
        steps = sorted(int(s) for s in self.df[STEP_COLUMN].dropna().unique())
        return {
            "name": self.name,
            "label": self.label,
            "is_synthetic": self.is_synthetic,
            "metadata": dict(self.metadata),
            "n_samples": int(len(self.df)),
            "start_s": round(float(t[0]), 6),
            "end_s": round(float(t[-1]), 6),
            "duration_s": round(float(t[-1] - t[0]), 6),
            "median_sample_interval_s": round(float(np.median(np.diff(t))), 6),
            "test_steps": steps,
            "missing_values": {c: int(self.df[c].isna().sum()) for c in MEASUREMENT_COLUMNS},
            "warnings": list(self.warnings),
        }


def _read_text(source: bytes | str | Path) -> str:
    """bytes = file contents (e.g. an upload); str or Path = a file path."""
    if isinstance(source, (str, Path)):
        source = Path(source).read_bytes()
    try:
        return source.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataValidationError(
            ["File is not UTF-8 text. Please upload a plain-text .csv file (not .xlsx or a binary file)."]
        ) from exc


def _header_info(text: str) -> tuple[bool, dict]:
    synthetic, meta = False, {}
    for line in text.splitlines()[:30]:
        if not line.startswith("#"):
            continue
        if SYNTHETIC_MARKER in line.upper():
            synthetic = True
        m = _META_RE.match(line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip()
    return synthetic, meta


def load_csv(source: bytes | str | Path, name: str | None = None) -> Dataset:
    """Parse and validate a CSV validation log. `source` is raw bytes or a file path."""
    if name is None:
        name = Path(source).name if isinstance(source, (str, Path)) else "uploaded.csv"
    text = _read_text(source)
    if not text.strip():
        raise DataValidationError(["File is empty."])

    try:
        raw = pd.read_csv(io.StringIO(text), comment="#", skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(["File has no header row or data (only comments or blank lines)."]) from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError([f"Malformed CSV: {exc}".strip()]) from exc

    errors: list[str] = []
    warnings: list[str] = []

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if len(set(raw.columns)) != len(raw.columns):
        errors.append(f"Duplicate column names: {list(raw.columns)}")

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        msg = f"Missing required column(s): {missing}. Found: {list(raw.columns)}."
        if len(raw.columns) == 1 and any(sep in raw.columns[0] for sep in (";", "\t", "|")):
            msg += " The file may use a delimiter other than ',' - please save it as comma-separated."
        errors.append(msg)
    if errors:
        raise DataValidationError(errors)

    extra = [c for c in raw.columns if c not in REQUIRED_COLUMNS]
    if extra:
        warnings.append(f"Ignored extra column(s): {extra}.")
    if len(raw) == 0:
        raise DataValidationError(["File has a header but no data rows."])
    if len(raw) > MAX_ROWS:
        raise DataValidationError([f"File has {len(raw):,} rows; this prototype accepts at most {MAX_ROWS:,}."])

    df = pd.DataFrame()
    for col in REQUIRED_COLUMNS:
        original = raw[col]
        if pd.api.types.is_numeric_dtype(original):  # pandas already parsed it: nothing non-numeric to report
            numeric, non_numeric = original.astype(float), 0
        else:
            blank = original.isna() | (original.astype(str).str.strip() == "")
            numeric = pd.to_numeric(original, errors="coerce").astype(float)
            non_numeric = int((numeric.isna() & ~blank).sum())
        if non_numeric:
            warnings.append(f"{col}: {non_numeric} non-numeric value(s) treated as missing.")
        non_finite = int(np.isinf(numeric).sum())
        if non_finite:
            warnings.append(f"{col}: {non_finite} infinite value(s) treated as missing.")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
        df[col] = numeric

    bad_time = int(df[TIME_COLUMN].isna().sum())
    if bad_time:
        warnings.append(f"Dropped {bad_time} row(s) with a missing or invalid timestamp_s.")
        df = df[df[TIME_COLUMN].notna()]
    if len(df) == 0:
        raise DataValidationError(["No rows have a valid timestamp_s."])

    if not df[TIME_COLUMN].is_monotonic_increasing:
        warnings.append("timestamp_s was not in increasing order; rows were sorted by time.")
        df = df.sort_values(TIME_COLUMN, kind="stable")
    dupes = int(df[TIME_COLUMN].duplicated().sum())
    if dupes:
        warnings.append(f"Dropped {dupes} row(s) with a duplicate timestamp_s (kept the first).")
        df = df[~df[TIME_COLUMN].duplicated()]
    if len(df) < 2:
        raise DataValidationError(["Need at least 2 samples with valid, distinct timestamps to analyse a run."])

    df = df.reset_index(drop=True)
    steps = df[STEP_COLUMN]
    if steps.isna().all():
        warnings.append("test_step has no valid values - plan tests cannot be located in this log.")
    else:
        if steps.isna().any():
            warnings.append(f"test_step: {int(steps.isna().sum())} row(s) have no step and belong to no test.")
        fractional = steps.dropna() % 1 != 0
        if fractional.any():
            warnings.append(f"test_step: {int(fractional.sum())} non-integer value(s) treated as missing.")
            df.loc[steps.notna() & (steps % 1 != 0), STEP_COLUMN] = np.nan
    for col in MEASUREMENT_COLUMNS:
        n_missing = int(df[col].isna().sum())
        if n_missing == len(df):
            warnings.append(f"{col}: no valid measurements - analyses that need it will be skipped.")
        elif n_missing:
            warnings.append(f"{col}: {n_missing} missing sample(s) ({100 * n_missing / len(df):.2f}% of rows).")

    synthetic, metadata = _header_info(text)
    return Dataset(df=df, name=name, is_synthetic=synthetic, warnings=warnings, metadata=metadata)

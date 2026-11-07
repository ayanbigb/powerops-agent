"""FastAPI backend for PowerOps Agent.

Run from the project folder:
    uvicorn api.main:app --reload --port 8000
Interactive API docs then live at http://127.0.0.1:8000/docs

Datasets are kept in memory only (a dict). Restarting the server forgets them.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from powerops.agent import DEFAULT_MODEL, AgentResult, run_agent
from powerops.config import ConfigError, load_plan
from powerops.data_io import DataValidationError, Dataset, load_csv
from powerops.plots import PLOT_TYPES, generate_plots
from powerops.synthetic import SyntheticConfig, random_config, synthetic_csv_bytes

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # OPENAI_API_KEY etc., if a .env file exists

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_DATASETS = 20

app = FastAPI(
    title="PowerOps Agent (student prototype)",
    description="Draft SST validation reports and revision regression checks from SYNTHETIC bench logs. "
                "Not an official product; read-only; no hardware control.",
    version="0.2.0",
)


class DatasetStore:
    """Tiny in-memory store. Oldest datasets are dropped after MAX_DATASETS."""

    def __init__(self, max_items: int):
        self._items: OrderedDict[str, tuple[Dataset, bytes]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_items

    def add(self, dataset: Dataset, csv_bytes: bytes) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._items[dataset_id] = (dataset, csv_bytes)
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        return dataset_id

    def get(self, dataset_id: str) -> tuple[Dataset, bytes]:
        with self._lock:
            item = self._items.get(dataset_id)
        if item is None:
            raise HTTPException(404, detail=f"Unknown dataset_id '{dataset_id}'. Upload or generate a dataset first "
                                            "(the server may have restarted).")
        return item

    def __len__(self) -> int:
        return len(self._items)


store = DatasetStore(MAX_DATASETS)


def json_safe(obj):
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def get_plan() -> dict:
    try:
        return load_plan()  # re-read every request, so edits to the JSON apply immediately
    except ConfigError as exc:
        raise HTTPException(500, detail=str(exc)) from exc


def dataset_payload(dataset_id: str, ds: Dataset) -> dict:
    preview = json.loads(ds.df.head(10).to_json(orient="records"))
    return json_safe({"dataset_id": dataset_id, **ds.info(), "preview": preview})


def figures_payload(figures: list[dict]) -> list[dict]:
    return [{"plot_id": f["plot_id"], "title": f["title"], "description": f["description"],
             "figure_json": f["figure"].to_json()} for f in figures]


def agent_payload(result: AgentResult) -> dict:
    return json_safe({
        "question": result.question, "answer": result.answer, "mode": result.mode, "model": result.model,
        "notices": result.notices, "unverified_numbers": result.unverified_numbers,
        "tool_calls": [{"name": r.name, "arguments": r.arguments, "ok": r.ok, "result": r.result} for r in result.tool_calls],
        "figures": figures_payload(result.figures), "report_markdown": result.report_markdown,
    })


def _synthetic(revision: str, seed: int, sensor_fault: bool) -> tuple[str, Dataset]:
    cfg = SyntheticConfig(revision=revision, seed=seed, battery_sensor_offset_kw=3.0 if sensor_fault else 0.0)
    csv_bytes = synthetic_csv_bytes(cfg)
    name = f"SYNTHETIC_SST_rev{revision}_seed{seed}{'_sensorfault' if sensor_fault else ''}.csv"
    ds = load_csv(csv_bytes, name=name)
    return store.add(ds, csv_bytes), ds


# --------------------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {"message": "PowerOps Agent API (student prototype). Interactive docs: /docs"}


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": bool(os.getenv("OPENAI_API_KEY")),
            "model": os.getenv("OPENAI_MODEL") or DEFAULT_MODEL, "datasets_loaded": len(store)}


@app.get("/plan")
def plan():
    return get_plan()


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        ds = load_csv(content, name=file.filename or "uploaded.csv")
    except DataValidationError as exc:
        raise HTTPException(422, detail={"message": "The CSV failed validation.", "errors": exc.errors}) from exc
    return dataset_payload(store.add(ds, content), ds)


class SyntheticRequest(BaseModel):
    revision: Literal["A", "B", "C"] = Field(
        "A", description="A = baseline, B = candidate with regressions, C = fixes B but slow battery pickup")
    seed: int = 42
    battery_sensor_fault: bool = Field(False, description="Add a +3 kW offset to the battery-port power reading")


@app.post("/datasets/synthetic")
def create_synthetic_dataset(req: SyntheticRequest):
    dataset_id, ds = _synthetic(req.revision, req.seed, req.battery_sensor_fault)
    return dataset_payload(dataset_id, ds)


class RandomRequest(BaseModel):
    seed: int | None = Field(None, ge=0, description="Omit for a brand-new random unit; give a seed to reproduce one")


@app.post("/datasets/random")
def create_random_dataset(req: RandomRequest | None = None):
    """A simulated run of a randomly behaving unit (new device parameters, sometimes a hidden sensor fault)."""
    cfg = random_config(req.seed if req else None)
    csv_bytes = synthetic_csv_bytes(cfg)
    # The file name gives nothing away about faults: the agent has to find them with its tools.
    ds = load_csv(csv_bytes, name=f"SYNTHETIC_SST_random_seed{cfg.seed}.csv")
    return dataset_payload(store.add(ds, csv_bytes), ds)


@app.post("/datasets/demo-pair")
def create_demo_pair():
    """One click for the demo: rev-A as the baseline and rev-B as the candidate under test."""
    base_id, base = _synthetic("A", 42, False)
    cand_id, cand = _synthetic("B", 43, False)
    return {"baseline": dataset_payload(base_id, base), "candidate": dataset_payload(cand_id, cand)}


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    ds, _ = store.get(dataset_id)
    return dataset_payload(dataset_id, ds)


@app.get("/datasets/{dataset_id}/csv")
def download_dataset_csv(dataset_id: str):
    ds, csv_bytes = store.get(dataset_id)
    return Response(content=csv_bytes, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{ds.name}"'})


@app.get("/datasets/{dataset_id}/plots/{plot_type}")
def get_plot(dataset_id: str, plot_type: str, test_id: str | None = None, baseline_id: str | None = Query(None)):
    if plot_type not in PLOT_TYPES:
        raise HTTPException(404, detail=f"Unknown plot type '{plot_type}'. Choose from {PLOT_TYPES}.")
    ds, _ = store.get(dataset_id)
    base = store.get(baseline_id)[0] if baseline_id else None
    summary, figures = generate_plots(ds.df, get_plan(), plot_type, test_id, ds.is_synthetic,
                                      baseline_df=base.df if base else None,
                                      labels=(ds.short_label, base.short_label if base else "baseline"))
    return {"status": summary["status"], "plots": figures_payload(figures)}


class ChatRequest(BaseModel):
    dataset_id: str = Field(description="The run under test (candidate)")
    baseline_dataset_id: str | None = Field(None, description="Optional baseline run to compare against")
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["auto", "demo", "llm"] = "auto"


@app.post("/chat")
def chat(req: ChatRequest):
    ds, _ = store.get(req.dataset_id)
    base = store.get(req.baseline_dataset_id)[0] if req.baseline_dataset_id else None
    if base is not None and req.baseline_dataset_id == req.dataset_id:
        raise HTTPException(422, detail="The baseline and the run under test are the same dataset.")
    try:
        result = run_agent(req.question, ds, get_plan(), baseline=base, mode=req.mode)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    return agent_payload(result)

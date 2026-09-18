"""
Thin FastAPI layer over the RiskModule core (pipeline/core.py). This is the
"reference implementation" referred to in README.md -- a modular monolith
with the same layer boundaries as the article's proposed microservices
architecture (Section 5), not yet split into separate deployable services.

Endpoints:
    POST /datasets            upload a CSV matching the unified `focus` schema
    GET  /datasets            list uploaded datasets
    GET  /modules             list available risk-modeling modules
    POST /runs                start a model run against an uploaded dataset
    GET  /runs/{run_id}       poll run status / summary
    GET  /runs/{run_id}/geojson   cluster map for a finished run

State (datasets, runs) lives in process memory -- there is no database yet
(PostGIS is still an open item in README.md). Restarting the server drops
everything; this is fine for local development and the pilot walkthrough,
not for production.
"""
from __future__ import annotations

import io
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline.core import RunContext
from pipeline.modules.space_time_scan import SpaceTimeScanModule

MODULES = {"space_time_scan": SpaceTimeScanModule()}

REQUIRED_COLUMNS = {
    "focus_id", "region_oblast", "district_raion", "settlement_name",
    "latitude", "longitude", "event_date", "year", "species",
    "animal_count", "confirmation_status", "data_source", "notes",
}
NUMERIC_COLUMNS = ("latitude", "longitude", "year", "animal_count")

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Biosafety Risk Pipeline (reference implementation)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_datasets: dict[str, dict[str, Any]] = {}
_runs: dict[str, dict[str, Any]] = {}


@app.get("/")
async def root():
    return {"service": "biosafety-risk-pipeline", "modules": list(MODULES)}


@app.post("/datasets")
async def ingest_dataset(disease: str = Form(...), file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"could not parse CSV: {exc}") from exc

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(422, f"missing required columns: {sorted(missing)}")

    if "reservoir_category" not in df.columns:
        df["reservoir_category"] = None
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["disease"] = disease

    dataset_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{dataset_id}.csv"
    df.to_csv(out_path, index=False)

    _datasets[dataset_id] = {
        "dataset_id": dataset_id,
        "disease": disease,
        "filename": file.filename,
        "row_count": len(df),
        "missing_coordinates": int(df["latitude"].isna().sum()),
        "missing_year": int(df["year"].isna().sum()),
        "path": str(out_path),
    }
    return _datasets[dataset_id]


@app.get("/datasets")
async def list_datasets():
    return list(_datasets.values())


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    dataset = _datasets.get(dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    return dataset


@app.get("/modules")
async def list_modules():
    return [
        {"name": m.name, "input_kind": m.input_kind, "produces": m.produces}
        for m in MODULES.values()
    ]


def _execute_run(run_id: str, dataset_id: str, module_name: str, params: dict) -> None:
    _runs[run_id]["status"] = "running"
    started = time.time()
    try:
        events = pd.read_csv(_datasets[dataset_id]["path"])
        module = MODULES[module_name]
        ctx = RunContext(disease=_datasets[dataset_id]["disease"], events=events, params=params)
        warnings = module.validate(ctx)
        result = module.run(ctx)
        result.summary["elapsed_seconds"] = round(time.time() - started, 2)
        _runs[run_id].update(
            {
                "status": "done",
                "summary": {**result.summary, "warnings": warnings},
                "clusters": result.tables["clusters"].to_dict(orient="records"),
                "geojson": result.geojson,
            }
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced to the client via /runs/{id}
        _runs[run_id].update({"status": "failed", "error": str(exc)})


@app.post("/runs")
async def create_run(
    dataset_id: str = Form(...),
    module: str = Form("space_time_scan"),
    params: str = Form("{}"),
):
    if dataset_id not in _datasets:
        raise HTTPException(404, "dataset not found")
    if module not in MODULES:
        raise HTTPException(422, f"unknown module {module!r}; available: {list(MODULES)}")
    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"params must be valid JSON: {exc}") from exc

    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "module": module,
        "params": params_dict,
        "status": "queued",
    }
    thread = threading.Thread(
        target=_execute_run, args=(run_id, dataset_id, module, params_dict), daemon=True
    )
    thread.start()
    return _runs[run_id]


@app.get("/runs")
async def list_runs():
    return [{k: v for k, v in r.items() if k not in ("clusters", "geojson")} for r in _runs.values()]


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@app.get("/runs/{run_id}/geojson")
async def get_run_geojson(run_id: str):
    run = _runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run["status"] != "done":
        raise HTTPException(409, f"run status is {run['status']!r}, not 'done'")
    return JSONResponse(run["geojson"])

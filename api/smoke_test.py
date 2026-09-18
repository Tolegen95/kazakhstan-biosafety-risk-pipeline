"""
End-to-end smoke test for the API, exercised against real normalized data
(no live server needed -- FastAPI's TestClient drives the app in-process).

    python3 api/smoke_test.py
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
DATA = Path(__file__).resolve().parents[1] / "data" / "normalized"


def main() -> None:
    print(client.get("/").json())

    with open(DATA / "rabies_normalized.csv", "rb") as fh:
        r = client.post(
            "/datasets",
            data={"disease": "rabies"},
            files={"file": ("rabies_normalized.csv", fh, "text/csv")},
        )
    r.raise_for_status()
    dataset = r.json()
    print("ingested dataset:", dataset["dataset_id"], dataset["row_count"], "rows")

    r = client.post(
        "/runs",
        data={
            "dataset_id": dataset["dataset_id"],
            "module": "space_time_scan",
            "params": '{"n_permutations": 199}',
        },
    )
    r.raise_for_status()
    run = r.json()
    run_id = run["run_id"]
    print("started run:", run_id)

    for _ in range(60):
        r = client.get(f"/runs/{run_id}")
        run = r.json()
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert run["status"] == "done", run
    print("summary:", run["summary"])
    print("clusters returned:", len(run["clusters"]))
    print("top cluster:", run["clusters"][0] if run["clusters"] else None)

    geo = client.get(f"/runs/{run_id}/geojson")
    geo.raise_for_status()
    print("geojson features:", len(geo.json()["features"]))
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()

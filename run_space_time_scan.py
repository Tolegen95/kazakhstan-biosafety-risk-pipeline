"""
Run the space-time permutation scan module on real, normalized registry
data and write cluster tables / GeoJSON / run summaries.

Usage:
    python run_space_time_scan.py --disease anthrax
    python run_space_time_scan.py --disease rabies
    python run_space_time_scan.py --disease anthrax --sensitivity
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pipeline.core import RunContext
from pipeline.modules.space_time_scan import SpaceTimeScanModule

HERE = Path(__file__).resolve().parent
NORMALIZED_DIR = HERE / "data" / "normalized"
RESULTS_DIR = NORMALIZED_DIR / "results"

# article Section 4.4 sensitivity grid
SPATIAL_GRID = [0.05, 0.10, 0.20, 0.30, 0.50]
TEMPORAL_GRID = [0.05, 0.10]


def load_events(disease: str) -> pd.DataFrame:
    path = NORMALIZED_DIR / f"{disease}_normalized.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run etl/normalize.py first")
    df = pd.read_csv(path)
    # pandas' default string-dtype inference can leave numeric columns as
    # `str` when the CSV was written with an empty cell for missing values;
    # coerce explicitly rather than rely on inference.
    for col in ("latitude", "longitude", "year", "animal_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_once(disease: str, events: pd.DataFrame, params: dict) -> tuple[dict, pd.DataFrame, dict]:
    module = SpaceTimeScanModule()
    ctx = RunContext(disease=disease, events=events, params=params)
    warnings = module.validate(ctx)
    t0 = time.time()
    result = module.run(ctx)
    elapsed = time.time() - t0
    result.summary["elapsed_seconds"] = round(elapsed, 2)
    result.summary["warnings"] = warnings
    return result.summary, result.tables["clusters"], result.geojson


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True, choices=["anthrax", "rabies", "fmd"])
    ap.add_argument("--max-spatial-fraction", type=float, default=0.5)
    ap.add_argument("--max-temporal-fraction", type=float, default=0.5)
    ap.add_argument("--n-permutations", type=int, default=199)
    ap.add_argument("--sensitivity", action="store_true", help="run the article's 5x2 sensitivity grid instead")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / args.disease
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.disease)

    if not args.sensitivity:
        params = {
            "max_spatial_fraction": args.max_spatial_fraction,
            "max_temporal_fraction": args.max_temporal_fraction,
            "n_permutations": args.n_permutations,
        }
        summary, clusters, geojson = run_once(args.disease, events, params)

        clusters.to_csv(out_dir / "clusters.csv", index=False)
        (out_dir / "clusters.geojson").write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
        print(f"\n{len(clusters)} clusters reported -> {out_dir}")
        if len(clusters):
            print(clusters.to_string(index=False))
        return

    rows = []
    for sf in SPATIAL_GRID:
        for tf in TEMPORAL_GRID:
            params = {"max_spatial_fraction": sf, "max_temporal_fraction": tf, "n_permutations": args.n_permutations}
            summary, clusters, _ = run_once(args.disease, events, params)
            rows.append(
                {
                    "max_spatial_fraction": sf,
                    "max_temporal_fraction": tf,
                    "n_candidate_zones": summary["n_candidate_zones"],
                    "n_candidate_windows": summary["n_candidate_windows"],
                    "observed_max_llr": summary["observed_max_llr"],
                    "observed_max_llr_p_value": summary["observed_max_llr_p_value"],
                    "n_significant_clusters_p<0.05": summary["n_significant_clusters_p<0.05"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                }
            )
            print(f"  sf={sf} tf={tf}: {rows[-1]}")

    sens = pd.DataFrame(rows)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    print(f"\nsensitivity grid -> {out_dir / 'sensitivity.csv'}")
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()

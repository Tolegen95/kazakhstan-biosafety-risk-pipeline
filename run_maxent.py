"""
Run the MaxEnt module on real anthrax presence data with open covariate
rasters (WorldClim). No institutional or colleague data needed.

Usage:
    python3 run_maxent.py --disease anthrax --covariate-dir /path/to/worldclim
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

from pipeline.core import RunContext
from pipeline.covariates import load_covariate_stack, load_covariate_stack_from_file
from pipeline.modules.maxent import MaxEntModule

HERE = Path(__file__).resolve().parent
NORMALIZED_DIR = HERE / "data" / "normalized"
RESULTS_DIR = NORMALIZED_DIR / "results"

# generous bounding box around Kazakhstan (matches etl/normalize.py's plausibility check)
KZ_BOUNDS = (45.0, 39.0, 88.5, 56.5)  # (min_lon, min_lat, max_lon, max_lat)


def build_covariate_paths(covariate_dir: str) -> dict[str, str]:
    paths = {}
    elev = glob.glob(os.path.join(covariate_dir, "*elev*.tif"))
    if elev:
        paths["elevation"] = elev[0]
    for bio_path in sorted(glob.glob(os.path.join(covariate_dir, "*bio_*.tif"))):
        name = os.path.basename(bio_path).split("_")[-1].replace(".tif", "")  # bio_12.tif -> 12
        paths[f"bio{name}"] = bio_path
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="anthrax", choices=["anthrax", "rabies"])
    ap.add_argument("--covariate-dir", help="directory of global WorldClim GeoTIFFs (downloaded fresh)")
    ap.add_argument("--covariate-file", default="data/covariates/kazakhstan_bioclim.tif",
                     help="pre-clipped multi-band GeoTIFF (default: the one tracked in this repo)")
    ap.add_argument("--n-iterations", type=int, default=25)
    ap.add_argument("--background-size", type=int, default=5000)
    ap.add_argument("--reservoir-category", choices=["livestock", "companion", "wildlife"], default=None)
    args = ap.parse_args()

    events = pd.read_csv(NORMALIZED_DIR / f"{args.disease}_normalized.csv")
    print(f"{len(events)} raw events loaded")
    if args.reservoir_category:
        events = events[events["reservoir_category"] == args.reservoir_category]
        print(f"{len(events)} events after filtering to reservoir_category={args.reservoir_category!r}")

    if args.covariate_dir:
        paths = build_covariate_paths(args.covariate_dir)
        if not paths:
            raise SystemExit(f"no covariate rasters found in {args.covariate_dir}")
        print(f"covariates found: {list(paths)}")
        covs = load_covariate_stack(paths, KZ_BOUNDS)
    else:
        covs = load_covariate_stack_from_file(args.covariate_file)
        print(f"loaded pre-clipped covariates from {args.covariate_file}: {covs.names}")
    print(f"covariate grid: {covs.shape}, valid cells: {int(covs.valid_mask().sum())}")

    module = MaxEntModule()
    ctx = RunContext(
        disease=args.disease,
        events=events,
        params={
            "covariates": covs,
            "n_iterations": args.n_iterations,
            "background_size": args.background_size,
        },
    )
    warnings = module.validate(ctx)
    for w in warnings:
        print("WARNING:", w)

    result = module.run(ctx)
    print(json.dumps(result.summary, indent=2))
    print("\njackknife variable importance:")
    print(result.tables["jackknife"].to_string(index=False))

    out_name = args.disease + (f"_{args.reservoir_category}" if args.reservoir_category else "")
    out_dir = RESULTS_DIR / "maxent" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    result.tables["jackknife"].to_csv(out_dir / "jackknife.csv", index=False)
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()

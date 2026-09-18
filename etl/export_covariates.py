"""
Clip the downloaded WorldClim bioclim+elevation rasters to the Kazakhstan
bounding box used by run_maxent.py and save as one small multi-band GeoTIFF,
so the repo doesn't need the ~100MB of full global rasters for reproducibility
-- only this ~2MB clipped file plus the download instructions below.

To regenerate from scratch:
    curl -O https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip
    curl -O https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_elev.zip
    unzip wc2.1_10m_bio.zip -d worldclim/ && unzip wc2.1_10m_elev.zip -d worldclim/
    python3 etl/export_covariates.py --source-dir worldclim --out data/covariates/kazakhstan_bioclim.tif

Usage:
    python3 export_covariates.py --source-dir /path/to/worldclim --out data/covariates/kazakhstan_bioclim.tif
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import rasterio

from pipeline.covariates import load_covariate_stack

KZ_BOUNDS = (45.0, 39.0, 88.5, 56.5)  # must match run_maxent.py


def build_covariate_paths(source_dir: str) -> dict[str, str]:
    paths = {}
    elev = glob.glob(os.path.join(source_dir, "*elev*.tif"))
    if elev:
        paths["elevation"] = elev[0]
    for bio_path in sorted(glob.glob(os.path.join(source_dir, "*bio_*.tif"))):
        name = os.path.basename(bio_path).split("_")[-1].replace(".tif", "")
        paths[f"bio{name}"] = bio_path
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = build_covariate_paths(args.source_dir)
    print(f"clipping {len(paths)} covariates to Kazakhstan bounds {KZ_BOUNDS}")
    covs = load_covariate_stack(paths, KZ_BOUNDS)
    print(f"clipped grid: {covs.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=covs.shape[0], width=covs.shape[1],
        count=len(covs.names), dtype=np.float32, crs=covs.crs, transform=covs.transform,
        nodata=np.nan, compress="lzw",
    ) as dst:
        for i, name in enumerate(covs.names, start=1):
            dst.write(covs.data[i - 1], i)
            dst.set_band_description(i, name)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

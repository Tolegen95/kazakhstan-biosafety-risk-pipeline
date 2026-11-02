"""
Export the input of the space-time scan module to SaTScan, so that SaTScan
(the reference implementation of the space-time permutation scan statistic)
can be run on EXACTLY the same aggregated data:

  * locations = districts (region_oblast / district_raion) with at least one
    case, placed at the mean coordinate of their cases -- as in the module;
  * time       = calendar year, one row per district-year with its case count.

Writes to data/normalized/results/satscan/<disease>/:
  <disease>.cas   case file:        <location id> <number of cases> <year>
  <disease>.geo   coordinates file: <location id> <latitude> <longitude>
  locations.csv   mapping location id -> region_oblast / district_raion
  SETTINGS.txt    the SaTScan settings to use (mirroring the module defaults)

Usage:  python3 benchmarks/satscan_export.py --disease anthrax
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True)
    ap.add_argument("--events-csv", default=None)
    ap.add_argument("--max-spatial-percent", type=float, default=50)
    ap.add_argument("--max-temporal-percent", type=float, default=50)
    ap.add_argument("--monte-carlo", type=int, default=999)
    args = ap.parse_args()

    ev = pd.read_csv(args.events_csv or HERE / "data" / "normalized" / f"{args.disease}_normalized.csv")
    for c in ("latitude", "longitude", "year"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    # same filter as the module: a usable district, coordinates and year
    ev = ev.dropna(subset=["region_oblast", "district_raion", "latitude", "longitude", "year"])
    ev["year"] = ev["year"].astype(int)

    d = (ev.groupby(["region_oblast", "district_raion"], as_index=False)
           .agg(lat=("latitude", "mean"), lon=("longitude", "mean"), n=("focus_id", "count")))
    d["loc_id"] = [f"D{i+1:04d}" for i in range(len(d))]
    ev = ev.merge(d[["region_oblast", "district_raion", "loc_id"]], on=["region_oblast", "district_raion"])
    cas = ev.groupby(["loc_id", "year"], as_index=False).size()

    od = HERE / "data" / "normalized" / "results" / "satscan" / args.disease
    od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{args.disease}.cas", "w", encoding="ascii") as fh:
        for r in cas.itertuples():
            fh.write(f"{r.loc_id} {r.size} {r.year}\n")
    with open(od / f"{args.disease}.geo", "w", encoding="ascii") as fh:
        for r in d.itertuples():
            fh.write(f"{r.loc_id} {r.lat:.6f} {r.lon:.6f}\n")
    d[["loc_id", "region_oblast", "district_raion", "lat", "lon", "n"]].to_csv(od / "locations.csv", index=False)

    y0, y1 = int(ev["year"].min()), int(ev["year"].max())
    settings = f"""SaTScan settings for the like-for-like comparison ({args.disease})
================================================================
Input tab
  Case file:                 {args.disease}.cas      (time precision: Year)
  Coordinates file:          {args.disease}.geo      (coordinates: Latitude/Longitude)
  Study period:              {y0}/1/1 to {y1}/12/31
Analysis tab
  Type of analysis:          Retrospective, Space-Time
  Probability model:         Space-Time Permutation
  Scan for areas with:       High rates
  Time aggregation:          1 Year
Advanced analysis features
  Spatial window:            maximum {args.max_spatial_percent:g} percent of the population at risk (= cases)
                             circular window
  Temporal window:           maximum {args.max_temporal_percent:g} percent of the study period
  Inference:                 {args.monte_carlo} Monte Carlo replications
Output tab
  Results file:              {args.disease}.txt  in this folder
  Additional output:         tick "Cluster Information" and "Location Information"
                             (dBase or CSV format)
  Criteria for reporting secondary clusters: No Geographical Overlap

Totals written: {len(ev)} cases, {len(d)} locations (districts), years {y0}-{y1}.
After the run:  python3 benchmarks/satscan_compare.py --disease {args.disease}
"""
    (od / "SETTINGS.txt").write_text(settings, encoding="utf-8")
    print(settings)
    print("written ->", od)


if __name__ == "__main__":
    main()

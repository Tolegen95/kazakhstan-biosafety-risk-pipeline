"""
Compare the clusters of the space-time scan module with SaTScan's clusters
on the same input (see benchmarks/satscan_export.py).

Inputs (all inside data/normalized/results/):
  <disease>/clusters.csv                          module output (run_space_time_scan.py)
  satscan/<disease>/locations.csv                  written by satscan_export.py
  satscan/<disease>/<disease>.col.(dbf|csv)        SaTScan cluster information
  satscan/<disease>/<disease>.gis.(dbf|csv)        SaTScan location information
     (if your SaTScan version names them differently, pass --col and --gis)

For every significant SaTScan cluster the script finds the module cluster with
the highest Jaccard overlap of district-year cells and reports both test
statistics. It also reports the overlap of the two primary clusters.

Usage:  pip install dbfread   (only needed for .dbf files)
        python3 benchmarks/satscan_compare.py --disease anthrax
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "data" / "normalized" / "results"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dbf":
        from dbfread import DBF
        df = pd.DataFrame(iter(DBF(str(path), load=True)))
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def find(df: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"none of {names} found in columns {list(df.columns)}")


def locate(folder: Path, disease: str, kind: str, given: str | None) -> Path:
    if given:
        return Path(given)
    for ext in (".dbf", ".csv"):
        p = folder / f"{disease}.{kind}{ext}"
        if p.exists():
            return p
    hits = sorted(folder.glob(f"*{kind}*"))
    if not hits:
        raise SystemExit(f"no SaTScan '{kind}' file in {folder}; pass --{kind}")
    return hits[0]


def year_of(v) -> int:
    return int(str(v).strip()[:4])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True)
    ap.add_argument("--col", default=None)
    ap.add_argument("--gis", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    sdir = RES / "satscan" / args.disease
    loc = pd.read_csv(sdir / "locations.csv")
    name_of = {r.loc_id: f"{r.region_oblast}/{r.district_raion}" for r in loc.itertuples()}

    col = read_table(locate(sdir, args.disease, "col", args.col))
    gis = read_table(locate(sdir, args.disease, "gis", args.gis))
    c_id, c_p = find(col, "CLUSTER"), find(col, "P_VALUE")
    c_llr = find(col, "TEST_STAT", "LLR", "LOG_LIKELIHOOD_RATIO")
    c_s, c_e = find(col, "START_DATE", "START"), find(col, "END_DATE", "END")
    c_obs = find(col, "OBSERVED", "CASES")
    g_id, g_loc = find(gis, "CLUSTER"), find(gis, "LOC_ID", "LOCATION_ID")

    sat = []
    for r in col.itertuples(index=False):
        r = r._asdict() if hasattr(r, "_asdict") else dict(zip(col.columns, r))
        cid = int(r[c_id])
        members = {name_of.get(str(x).strip(), str(x).strip()) for x in gis.loc[gis[g_id].astype(int) == cid, g_loc]}
        sat.append({"cluster": cid, "districts": members, "start": year_of(r[c_s]), "end": year_of(r[c_e]),
                    "llr": float(r[c_llr]), "p": float(r[c_p]), "observed": int(r[c_obs])})
    sat.sort(key=lambda c: -c["llr"])

    mpath = RES / args.disease / "clusters.csv"  # written by run_space_time_scan.py
    if not mpath.exists():
        raise SystemExit(f"{mpath} not found -- run: python3 run_space_time_scan.py --disease {args.disease} --n-permutations 999")
    mod_tab = pd.read_csv(mpath)
    mod = [{"rank": int(r.cluster_rank), "districts": set(str(r.districts).split("; ")),
            "start": int(r.start_year), "end": int(r.end_year), "llr": float(r.log_likelihood_ratio),
            "p": float(r.p_value), "observed": int(r.observed_cases)} for r in mod_tab.itertuples()]

    def cells(c):
        return {(d, y) for d in c["districts"] for y in range(c["start"], c["end"] + 1)}

    def jac(a, b):
        A, B = cells(a), cells(b)
        return len(A & B) / len(A | B) if A | B else 0.0

    sig_s = [c for c in sat if c["p"] < args.alpha]
    sig_m = [c for c in mod if c["p"] < args.alpha]
    rows = []
    for c in sig_s:
        best = max(sig_m, key=lambda m: jac(c, m), default=None)
        rows.append({
            "satscan_cluster": c["cluster"], "satscan_years": f"{c['start']}-{c['end']}",
            "satscan_n_districts": len(c["districts"]), "satscan_cases": c["observed"],
            "satscan_llr": round(c["llr"], 2), "satscan_p": c["p"],
            "module_rank": best["rank"] if best else None,
            "module_years": f"{best['start']}-{best['end']}" if best else None,
            "module_n_districts": len(best["districts"]) if best else None,
            "module_cases": best["observed"] if best else None,
            "module_llr": round(best["llr"], 2) if best else None,
            "jaccard_district_years": round(jac(c, best), 3) if best else 0.0,
        })
    table = pd.DataFrame(rows)
    summary = {
        "disease": args.disease,
        "n_significant_satscan": len(sig_s),
        "n_significant_module": len(sig_m),
        "primary_llr_satscan": round(sat[0]["llr"], 3) if sat else None,
        "primary_llr_module": round(mod[0]["llr"], 3) if mod else None,
        "primary_jaccard": round(jac(sat[0], mod[0]), 3) if sat and mod else None,
        "satscan_clusters_matched_jaccard_ge_0.5": int((table["jaccard_district_years"] >= 0.5).sum()) if len(table) else 0,
        "satscan_clusters_with_any_overlap": int((table["jaccard_district_years"] > 0).sum()) if len(table) else 0,
    }
    table.to_csv(sdir / "comparison.csv", index=False)
    (sdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

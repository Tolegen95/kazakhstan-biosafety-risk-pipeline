"""
Reuse experiment: run an additional disease (foot-and-mouth disease, FMD)
through the unchanged pipeline and measure how much code is disease-specific.

1. Code accounting (static): counts the lines of every disease-specific
   source loader in etl/normalize.py (functions named load_<disease>) and the
   lines of the shared, disease-independent code (core contract, modules,
   covariates, API, ETL helpers). Run `git diff --stat <commit before FMD>`
   yourself if you want the change history as well.
2. Execution: runs the space-time scan and the MaxEnt-equivalent module on
   data/normalized/fmd_normalized.csv through the same RunContext / RiskModule
   calls used for anthrax and rabies, without any change to module code.

Writes data/normalized/results/reuse_fmd/summary.json (+ clusters.csv).

Usage:  python3 benchmarks/reuse_fifth_disease.py --disease fmd
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from pipeline.core import RunContext  # noqa: E402
from pipeline.covariates import load_covariate_stack_from_file  # noqa: E402
from pipeline.modules.maxent import MaxEntModule  # noqa: E402
from pipeline.modules.space_time_scan import SpaceTimeScanModule  # noqa: E402

SHARED = ["pipeline/core.py", "pipeline/covariates.py", "pipeline/modules/space_time_scan.py",
          "pipeline/modules/maxent.py", "pipeline/modules/pca.py", "pipeline/modules/bayes_beta_binomial.py",
          "api/main.py"]


def nonblank_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#"))


def code_accounting() -> dict:
    src = (HERE / "etl" / "normalize.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    loaders, loader_total = {}, 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("load_"):
            n = nonblank_lines("\n".join(lines[node.lineno - 1: node.end_lineno]))
            loaders[node.name] = n
            loader_total += n
    shared = {p: nonblank_lines((HERE / p).read_text(encoding="utf-8")) for p in SHARED}
    etl_shared = nonblank_lines(src) - loader_total
    return {
        "disease_specific_loader_lines": loaders,
        "shared_lines_by_file": shared,
        "shared_etl_helper_lines": etl_shared,
        "shared_lines_total": sum(shared.values()) + etl_shared,
        "note": "non-blank, non-comment lines; module and contract files contain no disease-specific branches",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="fmd")
    ap.add_argument("--n-permutations", type=int, default=999)
    ap.add_argument("--n-iterations", type=int, default=100)
    args = ap.parse_args()

    out = {"disease": args.disease, "code": code_accounting()}
    ev = pd.read_csv(HERE / "data" / "normalized" / f"{args.disease}_normalized.csv")
    for c in ("latitude", "longitude", "year", "animal_count"):
        if c in ev:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    out["n_rows"] = int(len(ev))
    out["n_rows_with_coordinates"] = int(ev[["latitude", "longitude"]].notna().all(axis=1).sum())
    out["years"] = [int(ev["year"].min()), int(ev["year"].max())]

    od = HERE / "data" / "normalized" / "results" / "reuse_fmd"
    od.mkdir(parents=True, exist_ok=True)

    scan = SpaceTimeScanModule()
    ctx = RunContext(disease=args.disease, events=ev, params={"n_permutations": args.n_permutations})
    w = scan.validate(ctx)
    t0 = time.time()
    res = scan.run(ctx)
    out["space_time_scan"] = {k: res.summary.get(k) for k in (
        "n_events_used", "n_districts", "observed_max_llr", "observed_max_llr_p_value", "n_significant_clusters_p<0.05")}
    out["space_time_scan"]["seconds"] = round(time.time() - t0, 1)
    out["space_time_scan"]["warnings"] = w
    res.tables["clusters"].to_csv(od / "clusters.csv", index=False)

    covs = load_covariate_stack_from_file(str(HERE / "data/covariates/kazakhstan_bioclim.tif"))
    covs = covs.masked_to_boundary(str(HERE / "data/boundaries/kazakhstan.geojson"))
    mx = MaxEntModule()
    ctx = RunContext(disease=args.disease, events=ev, params={"covariates": covs, "n_iterations": args.n_iterations})
    w = mx.validate(ctx)
    t0 = time.time()
    res = mx.run(ctx)
    s = res.summary
    out["maxent"] = {"n_presence_used": s["n_presence_used"], "auc_random_mean": s["auc_mean"], "auc_random_sd": s["auc_std"],
                     "spatial_block_cv": s["spatial_block_cv"], "seconds": round(time.time() - t0, 1), "warnings": w}
    (od / "summary.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print("saved ->", od)


if __name__ == "__main__":
    main()

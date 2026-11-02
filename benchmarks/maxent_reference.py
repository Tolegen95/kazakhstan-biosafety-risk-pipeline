"""
Like-for-like benchmark of the MaxEnt-equivalent module against a full MaxEnt
implementation on IDENTICAL inputs.

Both models receive exactly the same presence points, the same background
points (inside the national boundary), the same covariates, the same random
75/25 splits and the same spatial-block folds. Only the model differs:

  module   : this repository's L2 logistic regression on linear + quadratic
             terms (pipeline/modules/maxent.py)
  maxent   : elapid.MaxentModel (Anderson 2023, JOSS 8(84):4930), a Python
             implementation of Phillips' MaxEnt with linear, quadratic, hinge
             and product features, beta multiplier 1.0 and cloglog output --
             the defaults of the MaxEnt software for >= 80 presence records.

Reported per model: AUC (random splits, mean +/- sd), AUC (spatial block CV,
mean over repeats of the k-fold mean), fit time, and the Spearman correlation
of the two suitability surfaces over all valid cells.

Usage (from the repository root):
    pip install elapid
    python3 benchmarks/maxent_reference.py --disease anthrax
    python3 benchmarks/maxent_reference.py --disease rabies --reservoir-category livestock
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from pipeline.covariates import load_covariate_stack_from_file  # noqa: E402
from pipeline.modules.maxent import _block_ids, _feature_matrix  # noqa: E402


def fit_module(p_tr, b_tr, l2_c=1.0):
    m, s = p_tr.mean(axis=0), p_tr.std(axis=0)
    s[s == 0] = 1.0
    clf = LogisticRegression(C=l2_c, max_iter=1000)
    clf.fit(_feature_matrix(np.concatenate([p_tr, b_tr]), m, s),
            np.concatenate([np.ones(len(p_tr)), np.zeros(len(b_tr))]))
    return lambda x: clf.predict_proba(_feature_matrix(x, m, s))[:, 1]


def fit_maxent(p_tr, b_tr, seed):
    import elapid
    model = elapid.MaxentModel(
        feature_types=["linear", "quadratic", "hinge", "product"],
        beta_multiplier=1.0, transform="cloglog", random_state=seed,
    )
    x = np.concatenate([p_tr, b_tr])
    y = np.concatenate([np.ones(len(p_tr)), np.zeros(len(b_tr))])
    model.fit(pd.DataFrame(x), y)
    return lambda z: np.asarray(model.predict(pd.DataFrame(z))).ravel()


FITTERS = {"module": lambda p, b, seed: fit_module(p, b), "maxent": fit_maxent}


def auc(pred, p_te, b_te):
    s = np.concatenate([pred(p_te), pred(b_te)])
    return roc_auc_score(np.concatenate([np.ones(len(p_te)), np.zeros(len(b_te))]), s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="anthrax")
    ap.add_argument("--reservoir-category", default=None)
    ap.add_argument("--covariate-file", default="data/covariates/kazakhstan_bioclim.tif")
    ap.add_argument("--boundary", default="data/boundaries/kazakhstan.geojson")
    ap.add_argument("--background-size", type=int, default=5000)
    ap.add_argument("--random-splits", type=int, default=20)
    ap.add_argument("--block-deg", type=float, default=2.0)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--events-csv", default=None, help="default: data/normalized/<disease>_normalized.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    covs = load_covariate_stack_from_file(str(HERE / args.covariate_file))
    if args.boundary:
        covs = covs.masked_to_boundary(str(HERE / args.boundary))

    ev = pd.read_csv(args.events_csv or HERE / "data" / "normalized" / f"{args.disease}_normalized.csv")
    if args.reservoir_category:
        ev = ev[ev["reservoir_category"] == args.reservoir_category]
    for c in ("latitude", "longitude"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev = ev.dropna(subset=["latitude", "longitude"])
    px = covs.sample(ev["longitude"].to_numpy(), ev["latitude"].to_numpy())
    keep = ~np.isnan(px).any(axis=1)
    px, plon, plat = px[keep], ev["longitude"].to_numpy()[keep], ev["latitude"].to_numpy()[keep]
    blon, blat = covs.random_background(args.background_size, rng)
    bx = covs.sample(blon, blat)
    print(f"{len(px)} presence points inside the boundary, {len(bx)} background points, {len(covs.names)} covariates")

    # identical random splits for both models
    splits = []
    for _ in range(args.random_splits):
        pi, bi = rng.permutation(len(px)), rng.permutation(len(bx))
        splits.append((pi[: int(0.75 * len(px))], pi[int(0.75 * len(px)):], bi[: int(0.75 * len(bx))], bi[int(0.75 * len(bx)):]))
    # identical block folds for both models
    pb, bb = _block_ids(plon, plat, args.block_deg), _block_ids(blon, blat, args.block_deg)
    blocks = np.unique(np.concatenate([pb, bb]))
    folds = []
    for _ in range(args.cv_repeats):
        f = dict(zip(blocks.tolist(), rng.permutation(len(blocks)) % args.cv_folds))
        folds.append((np.array([f[b] for b in pb.tolist()]), np.array([f[b] for b in bb.tolist()])))

    rows, out = [], {"disease": args.disease, "reservoir_category": args.reservoir_category,
                     "n_presence": int(len(px)), "n_background": int(len(bx))}
    surfaces = {}
    grid = covs.data.reshape(len(covs.names), -1).T
    valid = ~np.isnan(grid).any(axis=1)
    for name, fitter in FITTERS.items():
        t0 = time.time()
        r_aucs = [auc(fitter(px[a], bx[c], args.seed), px[b], bx[d]) for a, b, c, d in splits]
        b_means = []
        for rep, (pf, bf) in enumerate(folds):
            fa = []
            for k in range(args.cv_folds):
                if (pf == k).sum() == 0 or (bf == k).sum() == 0 or (pf != k).sum() < 5:
                    continue
                a_ = auc(fitter(px[pf != k], bx[bf != k], args.seed), px[pf == k], bx[bf == k])
                fa.append(a_)
                rows.append({"model": name, "repeat": rep, "fold": k, "auc": round(a_, 4)})
            b_means.append(np.mean(fa))
        full = fitter(px, bx, args.seed)
        surfaces[name] = full(grid[valid])
        out[name] = {
            "auc_random_mean": round(float(np.mean(r_aucs)), 4), "auc_random_sd": round(float(np.std(r_aucs)), 4),
            "auc_block_mean": round(float(np.mean(b_means)), 4), "auc_block_sd_across_repeats": round(float(np.std(b_means)), 4),
            "seconds_total": round(time.time() - t0, 1),
        }
        print(name, out[name])
    rho = spearmanr(surfaces["module"], surfaces["maxent"]).correlation
    out["spearman_surfaces"] = round(float(rho), 4)
    top = lambda s: s >= np.quantile(s, 0.8)
    a, b = top(surfaces["module"]), top(surfaces["maxent"])
    out["top20pct_overlap_jaccard"] = round(float((a & b).sum() / (a | b).sum()), 4)
    print("Spearman correlation of surfaces:", out["spearman_surfaces"], "| Jaccard of top-20% cells:", out["top20pct_overlap_jaccard"])

    tag = args.disease + (f"_{args.reservoir_category}" if args.reservoir_category else "")
    od = HERE / "data" / "normalized" / "results" / "benchmark_maxent" / tag
    od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(od / "block_folds.csv", index=False)
    print("saved ->", od)


if __name__ == "__main__":
    main()

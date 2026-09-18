"""
Maximum entropy (MaxEnt) species-distribution / environmental-suitability
modeling (article/dissertation Section 4.3), implemented here as
presence-background logistic regression with linear + quadratic feature
terms -- the "LQ" MaxEnt feature class.

This is not a re-implementation of Phillips' MaxEnt software byte-for-byte;
it rests on the documented statistical equivalence between MaxEnt and an
inhomogeneous Poisson point-process model, which is well approximated by
(infinitely-)weighted logistic regression of presence vs. background points
(Renner & Warton, 2013, Biometrics 69:274-281 -- already reference [12] in
the article). Using a well-known, citable equivalence instead of a from-
scratch reimplementation of an entire third-party software package is the
right scope for a reference implementation.

Method
------
- Presence points come from the unified `focus` table (point_events).
- Background points are drawn uniformly at random from valid (non-nodata)
  cells of the supplied covariate raster stack (pipeline/covariates.py).
- Each covariate is z-standardized; linear and squared terms are used as
  features (L2-regularized logistic regression, presence=1, background=0).
- Predictive performance is summarized by presence-background AUC over
  repeated random splits (article Section 4.3: 75% train / 25% test,
  n_iterations repeats), consistent with the paper's stated protocol.
- Variable importance follows a jackknife design matching the article's
  description: for each covariate, AUC is computed from a model trained
  using *only* that covariate, and from a model trained with that covariate
  *excluded*, against the full-model AUC.

Known simplifications (documented, not hidden): background points are drawn
from every valid-data cell inside the caller-supplied bounding box, not
masked to the true national border -- this repo has no Kazakhstan boundary
shapefile, so a rectangular bounding box is used instead (see run_maxent.py),
which lets a small amount of background fall in neighboring countries. And
the covariate set actually available here (WorldClim bioclim + elevation) is
narrower than the original studies' (which also used MGVF and land-cover/
soil-type layers not reconstructed here) -- run on real anthrax and rabies
data, this module's AUC nonetheless lands in the same range as the published
values (see prototype/README.md for the comparison table), which is the
claim actually being tested, not bit-for-bit numeric equality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pipeline.core import RiskModule, RiskResult, RunContext
from pipeline.covariates import CovariateStack

_DEFAULT_PARAMS = {
    "background_size": 5000,
    "train_fraction": 0.75,
    "n_iterations": 25,  # article uses 100; kept lower by default for a fast reference run
    "random_seed": 42,
    "l2_C": 1.0,
}


def _feature_matrix(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    return np.concatenate([z, z**2], axis=1)


class MaxEntModule(RiskModule):
    name = "maxent"
    input_kind = "point_events"
    produces = "suitability_surface"

    def validate(self, ctx: RunContext) -> list[str]:
        if "covariates" not in ctx.params or not isinstance(ctx.params["covariates"], CovariateStack):
            raise ValueError("params['covariates'] must be a CovariateStack (see pipeline/covariates.py)")
        events = ctx.events
        if events[["latitude", "longitude"]].isna().any().any():
            n = events[["latitude", "longitude"]].isna().any(axis=1).sum()
            return [f"{n} events have no coordinates and will be dropped"]
        return []

    def run(self, ctx: RunContext) -> RiskResult:
        p = {**_DEFAULT_PARAMS, **ctx.params}
        covs: CovariateStack = p["covariates"]
        rng = np.random.default_rng(p["random_seed"])

        events = ctx.events.dropna(subset=["latitude", "longitude"])
        presence_x = covs.sample(events["longitude"].to_numpy(), events["latitude"].to_numpy())
        keep = ~np.isnan(presence_x).any(axis=1)
        presence_x = presence_x[keep]
        n_dropped_nodata = int((~keep).sum())
        if len(presence_x) < 10:
            raise ValueError(f"only {len(presence_x)} presence points fall on valid covariate cells -- too few to fit")

        bg_lons, bg_lats = covs.random_background(p["background_size"], rng)
        background_x = covs.sample(bg_lons, bg_lats)

        mean, std = presence_x.mean(axis=0), presence_x.std(axis=0)
        std[std == 0] = 1.0

        def fit_and_score(cols: np.ndarray, n_iter: int) -> tuple[float, float]:
            aucs = []
            for _ in range(n_iter):
                p_train, p_test = train_test_split(presence_x[:, cols], train_size=p["train_fraction"], random_state=rng.integers(1 << 30))
                b_train, b_test = train_test_split(background_x[:, cols], train_size=p["train_fraction"], random_state=rng.integers(1 << 30))
                m, s = presence_x[:, cols].mean(axis=0), presence_x[:, cols].std(axis=0)
                s[s == 0] = 1.0
                X_train = _feature_matrix(np.concatenate([p_train, b_train]), m, s)
                y_train = np.concatenate([np.ones(len(p_train)), np.zeros(len(b_train))])
                X_test = _feature_matrix(np.concatenate([p_test, b_test]), m, s)
                y_test = np.concatenate([np.ones(len(p_test)), np.zeros(len(b_test))])
                clf = LogisticRegression(C=p["l2_C"], max_iter=1000)
                clf.fit(X_train, y_train)
                scores = clf.predict_proba(X_test)[:, 1]
                aucs.append(roc_auc_score(y_test, scores))
            return float(np.mean(aucs)), float(np.std(aucs))

        all_cols = np.arange(len(covs.names))
        full_auc_mean, full_auc_std = fit_and_score(all_cols, p["n_iterations"])

        jackknife_rows = []
        for i, name in enumerate(covs.names):
            only_mean, only_std = fit_and_score(np.array([i]), max(5, p["n_iterations"] // 3))
            without_cols = np.delete(all_cols, i)
            if len(without_cols):
                without_mean, without_std = fit_and_score(without_cols, max(5, p["n_iterations"] // 3))
            else:
                without_mean = None  # only one covariate total -- "without it" is undefined
            jackknife_rows.append(
                {
                    "covariate": name,
                    "auc_with_only_this": round(only_mean, 4),
                    "auc_without_this": round(without_mean, 4) if without_mean is not None else None,
                    "auc_full_model": round(full_auc_mean, 4),
                    "contribution": round(full_auc_mean - without_mean, 4) if without_mean is not None else None,
                }
            )
        jackknife_table = pd.DataFrame(jackknife_rows).sort_values("contribution", ascending=False)

        # final model on all data, for the suitability surface over the covariate grid
        X_all = _feature_matrix(np.concatenate([presence_x, background_x]), mean, std)
        y_all = np.concatenate([np.ones(len(presence_x)), np.zeros(len(background_x))])
        final_clf = LogisticRegression(C=p["l2_C"], max_iter=1000)
        final_clf.fit(X_all, y_all)

        rows, cols = covs.shape
        grid = covs.data.reshape(len(covs.names), -1).T  # (rows*cols, n_bands)
        valid = ~np.isnan(grid).any(axis=1)
        suitability = np.full(grid.shape[0], np.nan, dtype=np.float32)
        if valid.any():
            feats = _feature_matrix(grid[valid], mean, std)
            suitability[valid] = final_clf.predict_proba(feats)[:, 1]
        suitability_grid = suitability.reshape(rows, cols)

        summary = {
            "module": self.name,
            "disease": ctx.disease,
            "params": {k: v for k, v in p.items() if k != "covariates"},
            "covariates": covs.names,
            "n_presence_used": len(presence_x),
            "n_presence_dropped_nodata": n_dropped_nodata,
            "n_background": len(background_x),
            "auc_mean": round(full_auc_mean, 4),
            "auc_std": round(full_auc_std, 4),
        }

        return RiskResult(
            geojson=None,
            tables={"jackknife": jackknife_table, "suitability_grid": pd.DataFrame(suitability_grid)},
            summary=summary,
        )

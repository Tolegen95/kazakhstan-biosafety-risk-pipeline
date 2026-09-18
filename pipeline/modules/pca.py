"""
Principal component analysis for multi-pathogen classification (article
Section 4.5). Input is long-format stratum x pathogen prevalence counts;
pathogens are the PCA "observations" (rows) and strata (age group, farm
size class, ...) are the "variables" (columns) -- matching the dissertation
/ Kadyrov et al. (2023) convention: each row vector x_i is one pathogen's
prevalence profile across the stratification levels. Each variable/column
is z-standardized across pathogens, and PCA is the eigendecomposition of
the resulting correlation matrix.

Validated against Kadyrov, Ussenbayev, Kurenkeyeva, Kurenkey, Abdrakhmanov &
Tashatov (2023), "Principal component analysis in the epidemiology of
diarrhoea in calves" (IJEECS 30(3)): run on the age-group stratification
(data/literature/calf_diarrhea_prevalence.csv), this module's PC1/PC2
explained-variance shares reproduce the paper's reported 68.9%/22.4%
(cumulative 91.3%); on the farm-size stratification, PC1/PC2 reproduce the
paper's reported 61.5%/28.8% (cumulative 90.3%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.core import RiskModule, RiskResult, RunContext

_DEFAULT_PARAMS = {"stratification": None}  # which `stratification` value to run PCA on; None = use whatever is in events


class PCAModule(RiskModule):
    name = "pca"
    input_kind = "farm_survey"
    produces = "components"

    def validate(self, ctx: RunContext) -> list[str]:
        required = {"stratum", "pathogen", "n_examined", "n_positive"}
        missing = required - set(ctx.events.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        return []

    def run(self, ctx: RunContext) -> RiskResult:
        p = {**_DEFAULT_PARAMS, **ctx.params}
        events = ctx.events.copy()
        if p["stratification"] and "stratification" in events.columns:
            events = events[events["stratification"] == p["stratification"]]
        if events.empty:
            raise ValueError("no rows for the requested stratification")

        events["prevalence"] = events["n_positive"] / events["n_examined"]
        matrix = events.pivot_table(index="pathogen", columns="stratum", values="prevalence")
        matrix = matrix.dropna(how="any")
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            raise ValueError("need at least 2 pathogens and 2 strata for PCA")

        x = matrix.to_numpy()
        z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        cov = np.cov(z, rowvar=False, ddof=1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        explained = eigvals / eigvals.sum()

        variance_table = pd.DataFrame(
            {
                "component": [f"PC{i+1}" for i in range(len(eigvals))],
                "eigenvalue": np.round(eigvals, 5),
                "explained_variance": np.round(explained, 4),
                "cumulative_variance": np.round(np.cumsum(explained), 4),
            }
        )

        scores = z @ eigvecs  # (n_pathogens, n_components) -- new coordinates, cf. article Table 6/7
        n_report = min(2, scores.shape[1])
        coords_table = pd.DataFrame(
            scores[:, :n_report],
            columns=[f"PC{i+1}" for i in range(n_report)],
            index=matrix.index,
        ).reset_index().rename(columns={"index": "pathogen"})

        loadings_table = pd.DataFrame(
            eigvecs[:, :n_report],
            columns=[f"PC{i+1}" for i in range(n_report)],
            index=matrix.columns,
        ).reset_index().rename(columns={"index": "stratum"})

        summary = {
            "module": self.name,
            "disease": ctx.disease,
            "params": p,
            "n_pathogens": matrix.shape[0],
            "n_strata": matrix.shape[1],
            "pc1_explained_variance": round(float(explained[0]), 4),
            "pc2_explained_variance": round(float(explained[1]), 4) if len(explained) > 1 else None,
            "cumulative_pc1_pc2": round(float(explained[:2].sum()), 4) if len(explained) > 1 else round(float(explained[0]), 4),
        }

        return RiskResult(
            geojson=None,
            tables={"variance": variance_table, "coordinates": coords_table, "loadings": loadings_table},
            summary=summary,
        )

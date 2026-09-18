"""
Run the Bayesian and PCA modules against literature-derived data
(data/literature/*.csv) and check the results against the published numbers
they are reconstructed from. No institutional data or colleague input
needed -- see data/literature/README.md for provenance.

Usage:
    python3 run_literature_modules.py
"""
from __future__ import annotations

import pandas as pd

from pipeline.core import RunContext
from pipeline.modules.bayes_beta_binomial import BayesBetaBinomialModule
from pipeline.modules.pca import PCAModule

DATA_DIR = "data/literature"

# fitted in data/literature/README.md / cryptosporidiosis_strata.csv derivation
CRYPTO_PRIOR = {"alpha0": 24.969, "beta0": 184.287}

PUBLISHED_CRYPTO = {
    "overall": (0.0543, 0.0418, 0.0685),
    "female": (0.0644, 0.0473, 0.0839),
    "male": (0.0649, 0.0471, 0.0855),
    "<1 month": (0.0953, 0.0695, 0.1248),
    "1-3 months": (0.0747, 0.0514, 0.1018),
    "4-12 months": (0.0775, 0.0527, 0.1067),
}

PUBLISHED_PCA = {
    "age_group": {"pc1": 0.6888, "pc2": 0.2243, "cumulative": 0.9131},
    "farm_size": {"pc1": 0.6154, "pc2": 0.2881, "cumulative": 0.9035},
}


def run_bayes() -> None:
    print("=" * 70)
    print("BAYESIAN BETA-BINOMIAL -- cryptosporidiosis (Ussenbayev et al. 2020)")
    print("=" * 70)
    df = pd.read_csv(f"{DATA_DIR}/cryptosporidiosis_strata.csv")
    module = BayesBetaBinomialModule()

    for stratum_type, label, published_keys in [
        ("overall", "overall", ["overall"]),
        ("sex", "by sex", ["female", "male"]),
        ("age_group", "by age group", ["<1 month", "1-3 months", "4-12 months"]),
    ]:
        subset = df[df["stratum_type"] == stratum_type]
        ctx = RunContext(disease="cryptosporidiosis", events=subset, params={**CRYPTO_PRIOR, "group_col": "stratum"})
        module.validate(ctx)
        result = module.run(ctx)
        print(f"\n-- {label} --")
        print(result.tables["posterior"].to_string(index=False))
        for key in published_keys:
            row = result.tables["posterior"][result.tables["posterior"]["stratum"] == key]
            if row.empty:
                continue
            mean, lo, hi = PUBLISHED_CRYPTO[key]
            r = row.iloc[0]
            match = abs(r.posterior_mean - mean) < 0.001 and abs(r.credible_low - lo) < 0.001 and abs(r.credible_high - hi) < 0.001
            print(f"   published {key}: mean={mean} CI=[{lo},{hi}]  match={'YES' if match else 'no'}")


def run_pca() -> None:
    print("\n" + "=" * 70)
    print("PCA -- calf diarrhea enteropathogens (Kadyrov et al. 2023)")
    print("=" * 70)
    df = pd.read_csv(f"{DATA_DIR}/calf_diarrhea_prevalence.csv")
    module = PCAModule()

    for stratification in ("age_group", "farm_size"):
        subset = df[df["stratification"] == stratification]
        ctx = RunContext(disease="calf_diarrhea", events=subset, params={})
        module.validate(ctx)
        result = module.run(ctx)
        s = result.summary
        pub = PUBLISHED_PCA[stratification]
        print(f"\n-- stratification: {stratification} --")
        print(result.tables["variance"].to_string(index=False))
        print(result.tables["coordinates"].to_string(index=False))
        print(
            f"   PC1={s['pc1_explained_variance']} (published {pub['pc1']}), "
            f"PC2={s['pc2_explained_variance']} (published {pub['pc2']}), "
            f"cumulative={s['cumulative_pc1_pc2']} (published {pub['cumulative']})"
        )


if __name__ == "__main__":
    run_bayes()
    run_pca()

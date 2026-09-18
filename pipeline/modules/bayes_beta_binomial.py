"""
Bayesian beta-binomial prevalence estimation (article/dissertation Section
4.6): the number infected out of n examined is modeled as Binomial(n, theta);
because Beta is conjugate to the Binomial likelihood, a Beta(alpha, beta)
prior gives a Beta(alpha + y, beta + n - y) posterior in closed form.

Validated against Ussenbayev, Kurenkeyeva, Bauer & Kadyrov (2020), "Prevalence
of Calves' Cryptosporidiosis in Northern Kazakhstan" (ICCSA 2020): with prior
Beta(24.97, 184.29) -- fitted in this project to match the paper's reported
overall posterior, see data/literature/README.md -- this module reproduces
the paper's overall posterior mean 0.0543 and 95% credible interval
[0.0418, 0.0685] from n=894, y=35 cases, to the published rounding.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from pipeline.core import RiskModule, RiskResult, RunContext

_DEFAULT_PARAMS = {
    "alpha0": 1.0,
    "beta0": 1.0,
    "group_col": None,  # column to stratify by; None = one overall estimate
    "credible_level": 0.95,
}


class BayesBetaBinomialModule(RiskModule):
    name = "bayes_beta_binomial"
    input_kind = "farm_survey"
    produces = "posterior"

    def validate(self, ctx: RunContext) -> list[str]:
        warnings = []
        for col in ("n_examined", "n_positive"):
            if col not in ctx.events.columns:
                raise ValueError(f"missing required column: {col}")
        bad = ctx.events.dropna(subset=["n_examined", "n_positive"])
        bad = bad[bad["n_positive"] > bad["n_examined"]]
        if len(bad):
            raise ValueError(f"{len(bad)} rows have n_positive > n_examined")
        n_missing = ctx.events["n_examined"].isna().sum()
        if n_missing:
            warnings.append(f"{n_missing} rows have no n_examined and are excluded")
        return warnings

    def run(self, ctx: RunContext) -> RiskResult:
        p = {**_DEFAULT_PARAMS, **ctx.params}
        events = ctx.events.dropna(subset=["n_examined", "n_positive"]).copy()
        if events.empty:
            raise ValueError("no usable rows (need n_examined and n_positive)")

        group_col = p["group_col"]
        if group_col and group_col in events.columns:
            groups = events.groupby(group_col, sort=False)
        else:
            events["_all"] = "overall"
            groups = events.groupby("_all", sort=False)

        alpha0, beta0 = p["alpha0"], p["beta0"]
        level = p["credible_level"]
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2

        rows = []
        for stratum, g in groups:
            n = int(g["n_examined"].sum())
            y = int(g["n_positive"].sum())
            if n == 0:
                continue
            post = stats.beta(alpha0 + y, beta0 + n - y)
            rows.append(
                {
                    "stratum": stratum,
                    "n_examined": n,
                    "n_positive": y,
                    "mle_prevalence": round(y / n, 4),
                    "posterior_mean": round(float(post.mean()), 4),
                    "credible_low": round(float(post.ppf(lo_q)), 4),
                    "credible_high": round(float(post.ppf(hi_q)), 4),
                }
            )
        table = pd.DataFrame(rows)

        summary = {
            "module": self.name,
            "disease": ctx.disease,
            "params": p,
            "n_strata": len(table),
            "n_events_used": int(events["n_examined"].sum()),
        }
        return RiskResult(geojson=None, tables={"posterior": table}, summary=summary)

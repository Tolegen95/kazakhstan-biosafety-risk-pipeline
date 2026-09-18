"""
Generate the four results figures the article was missing (reviewer point
#9), from the real data already computed by this project -- no illustrative
placeholders. Saves PNGs into figures/ for insertion into the manuscript.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.core import RunContext
from pipeline.modules.pca import PCAModule

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------- Figure 3: anthrax clusters
def fig_clusters():
    df = pd.read_csv("data/normalized/results/anthrax/clusters.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sig = df["p_value"] < 0.05
    for flag, color, label in [(True, "#b3261e", "significant (p<0.05)"), (False, "#8a8a8a", "not significant")]:
        sub = df[sig == flag]
        ax.scatter(
            sub["centroid_lon"], sub["centroid_lat"],
            s=np.clip(sub["radius_km"], 8, None) * 2.2,
            c=color, alpha=0.45 if not flag else 0.55, edgecolors=color, linewidths=1.2,
            label=label,
        )
    # label only the top-6 significant clusters by observed cases, to keep it legible
    top = df[sig].sort_values("observed_cases", ascending=False).head(6)
    offsets = [(0, 14), (0, -16), (18, 6), (-18, 6), (0, 16), (0, -16)]
    for (_, r), (dx, dy) in zip(top.iterrows(), offsets):
        ax.annotate(
            f"{r.start_year}–{r.end_year} (n={r.observed_cases})",
            (r.centroid_lon, r.centroid_lat), fontsize=7,
            xytext=(dx, dy), textcoords="offset points", ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#555"),
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Space–time scan clusters, anthrax registry", fontsize=12)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.set_aspect(1.4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_anthrax_clusters.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4: MaxEnt jackknife
def fig_jackknife():
    df = pd.read_csv("data/normalized/results/maxent/anthrax/jackknife.csv")
    df = df.sort_values("contribution", ascending=True).tail(12)  # top 12 for readability
    fig, ax = plt.subplots(figsize=(6.5, 5))
    y = np.arange(len(df))
    full_auc = df["auc_full_model"].iloc[0]
    ax.barh(y, df["auc_with_only_this"], color="#7fb3d5", label="AUC with only this variable")
    ax.barh(y, df["auc_without_this"], color="#f2c14e", alpha=0.6, height=0.4, label="AUC without this variable")
    ax.axvline(full_auc, color="#b3261e", linestyle="--", linewidth=1.2, label=f"full-model AUC = {full_auc:.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(df["covariate"])
    ax.set_xlabel("AUC")
    ax.set_title("MaxEnt-equivalent jackknife variable importance, anthrax", fontsize=12)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_xlim(0.4, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_maxent_jackknife.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5: PCA biplot
def fig_pca_biplot():
    events = pd.read_csv("data/literature/calf_diarrhea_prevalence.csv")
    events = events[events["stratification"] == "age_group"]
    result = PCAModule().run(RunContext(disease="calf_diarrhea", events=events, params={}))
    coords = result.tables["coordinates"]
    loadings = result.tables["loadings"]
    var = result.tables["variance"]
    pc1_pct = var.loc[var.component == "PC1", "explained_variance"].iloc[0] * 100
    pc2_pct = var.loc[var.component == "PC2", "explained_variance"].iloc[0] * 100

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.scatter(coords["PC1"], coords["PC2"], color="#2a6f97", s=45, zorder=3)
    point_offsets = {
        "Cparvum_BRV": (-6, 10), "Cparvum_BCV": (6, 12), "EcoliK99": (6, -10),
        "BRV_BCV": (6, 6),
    }
    for _, r in coords.iterrows():
        dx, dy = point_offsets.get(r["pathogen"], (5, 5))
        ax.annotate(r["pathogen"], (r["PC1"], r["PC2"]), fontsize=8,
                    xytext=(dx, dy), textcoords="offset points")

    scale = coords[["PC1", "PC2"]].abs().to_numpy().max() * 0.9
    legend_lines = []
    for i, (_, r) in enumerate(loadings.iterrows(), start=1):
        ax.arrow(0, 0, r["PC1"] * scale, r["PC2"] * scale, color="#b3261e", width=0.01,
                  head_width=0.08, length_includes_head=True, alpha=0.8)
        ax.annotate(str(i), (r["PC1"] * scale * 1.08, r["PC2"] * scale * 1.08),
                    fontsize=8, color="#b3261e", ha="center", fontweight="bold")
        legend_lines.append(f"{i}: {r['stratum']}")
    ax.text(0.02, 0.02, "\n".join(legend_lines), transform=ax.transAxes, fontsize=7.5,
            color="#b3261e", va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#b3261e", alpha=0.85))

    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance)")
    ax.set_title("PCA biplot: enteropathogen prevalence by calf age group")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pca_biplot.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6: Bayesian posterior
def fig_bayesian_posterior():
    alpha0, beta0 = 24.969, 184.287
    n, y = 894, 35
    prior = stats.beta(alpha0, beta0)
    post = stats.beta(alpha0 + y, beta0 + n - y)
    mle = y / n

    x = np.linspace(0, 0.2, 1000)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(x, prior.pdf(x), color="#7fb3d5", label="prior Beta(24.97, 184.29)")
    ax.fill_between(x, post.pdf(x), color="#2a6f97", alpha=0.35)
    ax.plot(x, post.pdf(x), color="#2a6f97", label="posterior Beta(59.97, 1043.29)")
    ax.axvline(mle, color="#b3261e", linestyle="--", label=f"MLE = {mle:.4f}")
    lo, hi = post.ppf(0.025), post.ppf(0.975)
    ax.axvspan(lo, hi, color="#2a6f97", alpha=0.12)
    ax.annotate(f"95% CI\n[{lo:.4f}, {hi:.4f}]", (post.mean(), post.pdf(post.mean()) * 0.4),
                fontsize=8, ha="center", color="#2a6f97")
    ax.set_xlabel("Cryptosporidiosis prevalence θ")
    ax.set_ylabel("Density")
    ax.set_title("Bayesian beta-binomial posterior, cryptosporidiosis", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bayesian_posterior.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    fig_clusters()
    fig_jackknife()
    fig_pca_biplot()
    fig_bayesian_posterior()
    for f in sorted(OUT.glob("*.png")):
        print(f, f.stat().st_size, "bytes")

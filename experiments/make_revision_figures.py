"""Revised Figures 3 and 4 (Reviewer 1, point 8).

Figure 3: significant anthrax space–time clusters on a map with first-level administrative
boundaries; each cluster's spatial extent is the convex hull of its member districts
(district = mean coordinate of its events, as in the scan module).
Figure 4: block-CV predictor-group ablation and paired single-predictor jackknife with SDs.
"""
from pathlib import Path
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import MultiPoint

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'results' / 'revision'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
INK, MUTED, ACCENT = '#222222', '#8a8a8a', '#b3261e'
LABEL_SHIFT = {4: (-1.6, 1.1), 8: (-1.4, -0.6), 11: (1.6, -0.9)}  # separate overlapping labels


def fig_clusters():
    regions = gpd.read_file(ROOT / 'data/boundaries/kazakhstan_admin1.geojson')
    country = gpd.read_file(ROOT / 'data/boundaries/kazakhstan.geojson')
    land = country.union_all()
    ev = pd.read_csv(ROOT / 'data/normalized/anthrax_normalized.csv')
    ev = ev.dropna(subset=['latitude', 'longitude', 'year', 'district_raion'])
    ev['key'] = ev.region_oblast + '/' + ev.district_raion
    centroid = ev.groupby('key')[['longitude', 'latitude']].mean()
    cl = pd.read_csv(ROOT / 'data/normalized/results/anthrax/clusters.csv')
    cl = cl[cl.p_value < 0.05]

    fig, (ax, side) = plt.subplots(1, 2, figsize=(7.4, 4.4), gridspec_kw={'width_ratios': [3.1, 1.25]})
    regions.boundary.plot(ax=ax, color=MUTED, linewidth=0.5)
    country.boundary.plot(ax=ax, color=INK, linewidth=0.9)
    ax.scatter(ev.longitude, ev.latitude, s=1.2, color=MUTED, alpha=0.35, linewidths=0, zorder=1)
    for r in cl.itertuples():
        pts = centroid.loc[[k.strip() for k in r.districts.split(';')]].to_numpy()
        raw = MultiPoint([tuple(p) for p in pts]).convex_hull.buffer(0.35)
        hull = raw.intersection(land)
        if hull.area < 0.05:  # district centroid on or beyond the border: keep the unclipped marker
            hull = raw
        gpd.GeoSeries([hull]).plot(ax=ax, facecolor=ACCENT, alpha=0.18, edgecolor=ACCENT, linewidth=0.9, zorder=2)
        c = hull.centroid
        dx, dy = LABEL_SHIFT.get(r.cluster_rank, (0, 0))
        if dx or dy:
            ax.plot([c.x, c.x + dx], [c.y, c.y + dy], color=INK, lw=0.6, zorder=3)
        ax.text(c.x + dx, c.y + dy, str(r.cluster_rank), ha='center', va='center', fontsize=8, fontweight='bold', color=INK,
                zorder=4, bbox=dict(boxstyle='circle,pad=0.18', fc='white', ec=ACCENT, lw=0.8))
    ax.set_xlim(46, 88); ax.set_ylim(40, 56)
    ax.set_aspect(1.45)
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    side.axis('off')
    rows = [f'{r.cluster_rank:>2}  {r.start_year}–{r.end_year}  {r.n_districts:>2}  {r.observed_cases:>3}  {r.log_likelihood_ratio:6.1f}'
            for r in cl.itertuples()]
    side.text(0, 1, ' #  Years      D   Cases   LLR\n' + '\n'.join(rows), va='top', ha='left',
              family='DejaVu Sans Mono', fontsize=7.2, color=INK, transform=side.transAxes)
    side.text(0, 0.02, 'D: districts in cluster\nShaded: convex hull of member\ndistricts, clipped to the border;\ngrey dots: events; '
              'lines: region boundaries', va='bottom', fontsize=6.8, color=MUTED, transform=side.transAxes)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_anthrax_clusters_map.png', dpi=300)
    plt.close(fig)


def fig_jackknife():
    ab = pd.read_csv(EXP / 'predictor_ablation.csv')
    names = {'all': 'All 20 predictors', 'bioclim_without_elevation': '19 bioclimatic (no elevation)',
             'temperature_only': 'Temperature only', 'without_precipitation': 'Without precipitation',
             'precipitation_only': 'Precipitation only', 'without_temperature': 'Without temperature',
             'elevation_only': 'Elevation only'}
    ab = ab.sort_values('mean')
    jk = pd.read_csv(EXP / 'jackknife_paired.csv').sort_values('delta_auc')
    jk['label'] = [p.upper() if p.startswith('bio') else p.capitalize() for p in jk.predictor]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 4.2), gridspec_kw={'width_ratios': [1, 1.05]})
    y = range(len(ab))
    a.barh(y, ab['mean'] - 0.6, left=0.6, height=0.55, color='#4a7fb5')
    a.errorbar(ab['mean'], y, xerr=ab['sd_repeat'], fmt='none', ecolor=INK, elinewidth=0.9, capsize=2)
    a.set_yticks(list(y)); a.set_yticklabels([names[g] for g in ab.group])
    a.axvline(ab.loc[ab.group == 'all', 'mean'].item(), color=INK, lw=0.8, ls='--')
    a.set_xlim(0.6, 0.8); a.set_xlabel('Block-CV AUC (mean ± SD)')
    a.set_title('(a) Predictor-group ablation', fontsize=9, loc='left')

    y = range(len(jk))
    b.barh(y, jk.delta_auc, height=0.6, color=['#4a7fb5' if d > 0 else MUTED for d in jk.delta_auc])
    b.errorbar(jk.delta_auc, y, xerr=jk.sd_delta_across_repeats, fmt='none', ecolor=INK, elinewidth=0.9, capsize=2)
    b.axvline(0, color=INK, lw=0.8)
    b.set_yticks(list(y)); b.set_yticklabels(jk.label, fontsize=7.5)
    b.set_xlabel('AUC decrease when removed (± SD)')
    b.set_title('(b) Single-predictor jackknife', fontsize=9, loc='left')
    for ax in (a, b):
        ax.grid(axis='x', alpha=0.25)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_predictor_importance.png', dpi=300)
    plt.close(fig)


fig_clusters()
fig_jackknife()
print('saved', *sorted(p.name for p in OUT.glob('fig*.png')))

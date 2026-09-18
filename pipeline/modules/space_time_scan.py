"""
Space-time permutation scan statistic (Kulldorff, Heffernan, Hartman,
Assuncao & Mostashari, 2005) -- the same method the article describes as
"implemented via SaTScan" (Section 4.4). This is a from-scratch, pure
NumPy/pandas implementation with no SaTScan binary dependency, using the
case-only permutation model, which needs no population-at-risk denominator
-- appropriate here since no district-level population data is available.

Method
------
- Space is discretized to administrative districts (`district_raion`), each
  represented by the mean coordinate of its confirmed cases. Candidate
  spatial zones = each district plus its geometrically-spaced-size sets of
  nearest neighbouring districts (1, 2, 3, 4, 7, 11, ... members), up to a
  caller-supplied maximum fraction of total cases. Geometric spacing bounds
  the number of candidate zones to a tractable size instead of enumerating
  every possible neighbour count.
- Time is discretized to calendar years. Candidate temporal windows = all
  contiguous year ranges up to a caller-supplied maximum fraction of the
  study period.
- For a space-time cylinder Z with observed case count c and expected count
  mu = n_zone * n_window / N (the product of marginals -- the null of no
  space-time interaction), the log-likelihood ratio is
      LLR(Z) = c*ln(c/mu) + (C-c)*ln((C-c)/(C-mu))   if c > mu, else 0
  and the most likely cluster is the zone maximizing LLR.
- Significance is assessed by Monte Carlo permutation: shuffle the year
  label of each case while holding its district fixed. This preserves both
  the per-district and per-year marginal totals exactly (a permutation of a
  fixed multiset of year labels across a fixed multiset of districts), which
  is the space-time permutation null model. Recompute max LLR each
  replicate; p = (1 + #{replicates with max LLR >= observed}) / (R + 1).
  Non-primary clusters are tested against the same reference distribution
  ("secondary clusters"), following standard scan-statistic practice.

This trades some statistical power for a tractable pure-Python runtime:
district-level rather than point-level spatial discretization, and
geometrically-spaced neighbour counts rather than exhaustive radii. That
trade-off is documented, not hidden -- this is a reference implementation,
not a claim to reproduce SaTScan's numeric output exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.core import RiskModule, RiskResult, RunContext

_DEFAULT_PARAMS = {
    "max_spatial_fraction": 0.5,
    "max_temporal_fraction": 0.5,
    "n_permutations": 199,
    "random_seed": 42,
    "max_clusters_reported": 20,
}


def _district_centroids(events: pd.DataFrame) -> pd.DataFrame:
    d = (
        events.groupby(["region_oblast", "district_raion"], as_index=False)
        .agg(lat=("latitude", "mean"), lon=("longitude", "mean"), n_cases=("focus_id", "count"))
        .reset_index(drop=True)
    )
    d["district_id"] = np.arange(len(d))
    return d


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _build_spatial_zones(districts: pd.DataFrame, max_spatial_fraction: float, total_cases: int):
    """Returns a list of (member_district_indices: np.ndarray, radius_km: float)."""
    n = len(districts)
    lat = districts["lat"].to_numpy()
    lon = districts["lon"].to_numpy()
    n_cases = districts["n_cases"].to_numpy()
    max_cases = max_spatial_fraction * total_cases

    dist = np.stack([_haversine_km(lat[i], lon[i], lat, lon) for i in range(n)])

    zones: list[tuple[np.ndarray, float]] = []
    growth_steps = sorted({max(1, round(1.6**k)) for k in range(20)} & set(range(1, n + 1)))
    for i in range(n):
        order = np.argsort(dist[i])
        cum_cases = np.cumsum(n_cases[order])
        for k in growth_steps:
            if cum_cases[k - 1] > max_cases:
                break
            zones.append((order[:k], float(dist[i, order[k - 1]])))
    return zones


def _build_temporal_windows(years: np.ndarray, max_temporal_fraction: float):
    y0, y1 = int(years.min()), int(years.max())
    span = y1 - y0 + 1
    max_len = max(1, round(max_temporal_fraction * span))
    windows = [
        (start, start + length - 1)
        for start in range(y0, y1 + 1)
        for length in range(1, max_len + 1)
        if start + length - 1 <= y1
    ]
    return windows, y0, y1


class SpaceTimeScanModule(RiskModule):
    name = "space_time_scan"
    input_kind = "point_events"
    produces = "cluster_table"

    def validate(self, ctx: RunContext) -> list[str]:
        warnings = []
        events = ctx.events
        if "year" not in events or events["year"].isna().all():
            raise ValueError("no usable `year` values in input events")
        if events["district_raion"].isna().any():
            warnings.append(
                f"{events['district_raion'].isna().sum()} events have no district_raion and are excluded"
            )
        if events["latitude"].isna().any():
            warnings.append(f"{events['latitude'].isna().sum()} events have no coordinates and are excluded")
        return warnings

    def run(self, ctx: RunContext) -> RiskResult:
        p = {**_DEFAULT_PARAMS, **ctx.params}

        events = ctx.events.dropna(subset=["district_raion", "latitude", "longitude", "year"]).copy()
        events["year"] = events["year"].astype(int)
        if events.empty:
            raise ValueError("no usable events after dropping rows with missing district/coordinates/year")

        districts = _district_centroids(events)
        district_lookup = {
            (row.region_oblast, row.district_raion): row.district_id for row in districts.itertuples()
        }
        events["district_id"] = [
            district_lookup[(o, d)] for o, d in zip(events["region_oblast"], events["district_raion"])
        ]

        windows, y0, y1 = _build_temporal_windows(events["year"].to_numpy(), p["max_temporal_fraction"])
        n_years_axis = y1 - y0 + 1
        n_districts = len(districts)
        total_cases = len(events)

        zones = _build_spatial_zones(districts, p["max_spatial_fraction"], total_cases)
        if not zones or not windows:
            raise ValueError("parameter combination produced zero candidate space-time zones")

        zone_membership = np.zeros((len(zones), n_districts), dtype=np.float64)
        for zi, (members, _radius) in enumerate(zones):
            zone_membership[zi, members] = 1.0

        window_starts = np.array([w[0] - y0 for w in windows])
        window_ends = np.array([w[1] - y0 for w in windows])
        district_id_arr = events["district_id"].to_numpy()

        def counts_by_district_year(year_labels: np.ndarray) -> np.ndarray:
            flat = district_id_arr * n_years_axis + (year_labels - y0)
            return np.bincount(flat, minlength=n_districts * n_years_axis).reshape(n_districts, n_years_axis)

        def window_sums(m: np.ndarray) -> np.ndarray:
            """(n_districts, n_years_axis) -> (n_districts, n_windows) summed over each window."""
            cum = np.concatenate([np.zeros((n_districts, 1)), np.cumsum(m, axis=1)], axis=1)
            return cum[:, window_ends + 1] - cum[:, window_starts]

        def max_llr(m: np.ndarray) -> tuple[float, np.ndarray]:
            ws = window_sums(m)  # (n_districts, n_windows)
            obs = zone_membership @ ws  # (n_zones, n_windows)
            zone_totals = zone_membership @ m.sum(axis=1)  # (n_zones,)
            year_totals_cum = np.concatenate([np.zeros(1), np.cumsum(m.sum(axis=0))])
            window_totals = year_totals_cum[window_ends + 1] - year_totals_cum[window_starts]  # (n_windows,)
            mu = np.outer(zone_totals, window_totals) / total_cases

            c = obs
            rest_c = total_cases - c
            rest_mu = total_cases - mu
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = np.where((c > 0) & (mu > 0), c * np.log(c / np.where(mu > 0, mu, 1)), 0.0)
                t2 = np.where((rest_c > 0) & (rest_mu > 0), rest_c * np.log(rest_c / np.where(rest_mu > 0, rest_mu, 1)), 0.0)
            llr = np.where(c > mu, t1 + t2, 0.0)
            return float(llr.max()), llr

        observed_counts = counts_by_district_year(events["year"].to_numpy())
        observed_max, observed_llr = max_llr(observed_counts)

        rng = np.random.default_rng(p["random_seed"])
        year_labels = events["year"].to_numpy()
        perm_max = np.empty(p["n_permutations"])
        for i in range(p["n_permutations"]):
            shuffled_years = rng.permutation(year_labels)
            perm_max[i], _ = max_llr(counts_by_district_year(shuffled_years))

        def p_value(llr_value: float) -> float:
            return float((1 + np.sum(perm_max >= llr_value)) / (p["n_permutations"] + 1))

        order = np.dstack(np.unravel_index(np.argsort(-observed_llr, axis=None), observed_llr.shape))[0]
        observed_window_sums = window_sums(observed_counts)

        chosen: list[tuple[int, int]] = []
        used_districts: set[int] = set()
        for zi, wi in order:
            if observed_llr[zi, wi] <= 0 or len(chosen) >= p["max_clusters_reported"]:
                break
            members = set(zones[zi][0].tolist())
            if members & used_districts:
                continue
            chosen.append((zi, wi))
            used_districts |= members

        rows = []
        for zi, wi in chosen:
            members, radius_km = zones[zi]
            start_year, end_year = windows[wi]
            member_districts = districts.iloc[members]
            centroid_lat = float(np.average(member_districts["lat"], weights=member_districts["n_cases"]))
            centroid_lon = float(np.average(member_districts["lon"], weights=member_districts["n_cases"]))
            llr_val = float(observed_llr[zi, wi])
            obs_count = int(zone_membership[zi] @ observed_window_sums[:, wi])
            rows.append(
                {
                    "cluster_rank": len(rows) + 1,
                    "districts": "; ".join(
                        f"{r.region_oblast}/{r.district_raion}" for r in member_districts.itertuples()
                    ),
                    "n_districts": len(members),
                    "centroid_lat": round(centroid_lat, 5),
                    "centroid_lon": round(centroid_lon, 5),
                    "radius_km": round(radius_km, 1),
                    "start_year": start_year,
                    "end_year": end_year,
                    "observed_cases": obs_count,
                    "log_likelihood_ratio": round(llr_val, 3),
                    "p_value": round(p_value(llr_val), 4),
                }
            )
        cluster_table = pd.DataFrame(rows)

        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r["centroid_lon"], r["centroid_lat"]]},
                    "properties": {k: v for k, v in r.items() if k not in ("centroid_lat", "centroid_lon")},
                }
                for r in rows
            ],
        }

        summary = {
            "module": self.name,
            "disease": ctx.disease,
            "params": p,
            "n_events_used": total_cases,
            "n_events_dropped": len(ctx.events) - total_cases,
            "n_districts": n_districts,
            "n_candidate_zones": len(zones),
            "n_candidate_windows": len(windows),
            "observed_max_llr": round(observed_max, 3),
            "observed_max_llr_p_value": round(p_value(observed_max), 4),
            "n_significant_clusters_p<0.05": int(sum(1 for r in rows if r["p_value"] < 0.05)),
        }

        return RiskResult(geojson=geojson, tables={"clusters": cluster_table}, summary=summary)

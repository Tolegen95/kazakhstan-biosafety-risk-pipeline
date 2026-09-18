"""
Generate a small, clearly-synthetic point-event dataset in the unified focus
schema, with two injected space-time clusters plus background noise, so
someone without access to the real institutional registries can still run
the space-time-scan module end-to-end (via the API, the web UI, or
run_space_time_scan.py) and see a non-trivial result.

This is NOT real data. Settlement names, coordinates and dates are
synthetic. See README.md "Running the API + web UI" for how to use it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUT = "data/examples/example_focus_events.csv"
KZ_LAT, KZ_LON = (40.0, 55.5), (47.0, 87.0)

rng = np.random.default_rng(7)
rows = []


def add(lat, lon, year, oblast, district, settlement):
    rows.append(
        {
            "focus_id": f"EX-{len(rows)+1:04d}",
            "disease": "anthrax",
            "region_oblast": oblast,
            "district_raion": district,
            "settlement_name": settlement,
            "latitude": round(float(lat), 5),
            "longitude": round(float(lon), 5),
            "event_date": f"{year}-06-15",
            "year": year,
            "species": rng.choice(["cattle", "sheep", "horse"]),
            "reservoir_category": None,
            "animal_count": int(rng.integers(1, 8)),
            "confirmation_status": "suspected",
            "data_source": "synthetic_example",
            "notes": "synthetic example row -- not real surveillance data",
        }
    )


# cluster 1: Akmola-ish area, 1990-1995
for _ in range(35):
    add(51.2 + rng.normal(0, 0.3), 71.4 + rng.normal(0, 0.3), int(rng.integers(1990, 1996)),
        "Akmola", "Example-District-A", "Example-village-1")

# cluster 2: East Kazakhstan-ish area, 2010-2012
for _ in range(25):
    add(49.9 + rng.normal(0, 0.25), 82.6 + rng.normal(0, 0.25), int(rng.integers(2010, 2013)),
        "East Kazakhstan", "Example-District-B", "Example-village-2")

# scattered background noise, whole country, whole period
for _ in range(60):
    lat = rng.uniform(*KZ_LAT)
    lon = rng.uniform(*KZ_LON)
    year = int(rng.integers(1970, 2020))
    add(lat, lon, year, "Example-Oblast", "Example-District-C", f"Example-village-{rng.integers(3,50)}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"wrote {len(df)} synthetic rows -> {OUT}")

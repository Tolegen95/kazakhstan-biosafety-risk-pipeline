"""
ETL: normalize raw registry files into the unified spatio-temporal-species
data model (one `focus` table for anthrax, rabies and FMD).

Unified schema (matches data_templates/*_template.csv):
    focus_id, disease, region_oblast, district_raion, settlement_name,
    latitude, longitude, event_date, year, species, reservoir_category,
    animal_count, confirmation_status, data_source, notes

Design point being demonstrated: the same script, the same target schema,
and the same downstream `RiskModule` contract serve three structurally
different nosologies. Disease identity is a column value, not a different
table or a different pipeline.

Usage:
    python normalize.py --raw-dir ../../dataset --out-dir ../data/normalized
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import pandas as pd

# --- oblast name normalization -------------------------------------------------
# Raw files spell the same region differently (typos, transliteration
# variants, trailing spaces). Canonicalize to one name per region so the
# space-time module doesn't treat "Akmolinskaya" and "Akmolinskay" as two
# different places.
OBLAST_MAP = {
    "akmolinskaya": "Akmola", "akmolinskay": "Akmola", "akmola": "Akmola",
    "aktubinskaya": "Aktobe", "aktobe": "Aktobe",
    "almatinskaya": "Almaty region", "almaty": "Almaty region",
    "almaty city": "Almaty city", "shymkent city": "Shymkent city",
    "atyrauskaya": "Atyrau", "atyrauskya": "Atyrau", "atyrau": "Atyrau",
    "vostochno-kazakhstanskaya": "East Kazakhstan", "vostochno-kazakhstanskay": "East Kazakhstan",
    "zhambylskaya": "Zhambyl", "zambylskaya": "Zhambyl", "zhambylskay": "Zhambyl", "zhambyl": "Zhambyl",
    "zapadno-kazakhstanskaya": "West Kazakhstan", "zapadno-kazakhstanskay": "West Kazakhstan",
    "karagandinskaya": "Karaganda", "karagandy": "Karaganda",
    "kostanayskaya": "Kostanay", "kostanay": "Kostanay",
    "kyzylordinskya": "Kyzylorda", "kyzilordinskaya": "Kyzylorda", "kyzylordinskay": "Kyzylorda", "kyzylorda": "Kyzylorda",
    "mangystau": "Mangystau", "mangistauskay": "Mangystau",
    "pavlodarskaya": "Pavlodar", "pavlodar": "Pavlodar",
    "severo-kazakhstanskaya": "North Kazakhstan", "severo-kazahstanskay": "North Kazakhstan",
    "uzno-kazakhstanskaya": "South Kazakhstan (Turkestan)",
    "yuzhno-kazakhstanskaya": "South Kazakhstan (Turkestan)", "yuzhno-kazakhstanskay": "South Kazakhstan (Turkestan)",
}


def normalize_oblast(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    key = str(raw).strip().lower()
    return OBLAST_MAP.get(key, str(raw).strip())


_CLEAN_FLOAT = re.compile(r"^-?\d+\.\d+$")

# Kazakhstan's real extent is roughly lat 40.6-55.4, lon 46.5-87.3; this
# bounding box is deliberately a bit more generous so it doesn't reject
# legitimate border-area foci.
KZ_LAT_RANGE = (39.0, 56.5)
KZ_LON_RANGE = (45.0, 88.5)


def parse_loose_coordinate(raw) -> tuple[float | None, str | None]:
    """Coordinates in the anthrax source occasionally contain typos such as
    "50.,115556" or "5,.458728" (a stray comma next to the decimal point).
    Rather than guess the intended value, treat anything that isn't a clean
    float as unparseable and flag it -- do not invent a coordinate."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    s = str(raw).strip()
    if _CLEAN_FLOAT.match(s):
        return float(s), None
    return None, f"unparseable coordinate in source: {s!r}"


def check_kazakhstan_bounds(lat: float | None, lon: float | None) -> tuple[float | None, float | None, str | None]:
    """111 of 4307 anthrax rows have a cleanly-parsed but geographically
    implausible coordinate (e.g. lat==lon, or values far outside Kazakhstan
    -- almost certainly a data-entry error in the original spreadsheet, not
    a parsing artifact here). Drop rather than guess a correction."""
    if lat is None or lon is None:
        return lat, lon, None
    if not (KZ_LAT_RANGE[0] <= lat <= KZ_LAT_RANGE[1]) or not (KZ_LON_RANGE[0] <= lon <= KZ_LON_RANGE[1]):
        return None, None, f"coordinate outside plausible Kazakhstan bounding box in source: lat={lat}, lon={lon}"
    return lat, lon, None


def parse_loose_count(raw) -> tuple[int | None, str | None]:
    """`animal_count` in the anthrax source sometimes holds more than one
    number in one cell (e.g. "7, 1", "12, 4, 12"), apparently per-species or
    per-sub-event counts recorded together. We sum the numbers found and
    flag the row rather than silently pick one and drop the rest."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    if isinstance(raw, (int, float)):
        return int(raw), None
    s = str(raw).strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return None, f"unparseable animal_count in source: {s!r}"
    total = sum(int(n) for n in nums)
    if len(nums) > 1:
        return total, f"composite animal_count in source ({s!r}) summed to {total}"
    return total, None


# --- messy date/year parsing (anthrax file mixes datetime / int / "MM.YYYY") ---
def parse_anthrax_year(raw) -> tuple[str | None, int | None, str | None]:
    """Returns (event_date_iso, year, note) for the anthrax `year` column."""
    if raw is None:
        return None, None, "no date/year recorded in source"
    if isinstance(raw, dt.datetime):
        return raw.date().isoformat(), raw.year, None
    if isinstance(raw, int):
        if 1900 <= raw <= 2100:
            return None, raw, None
        # two known rows carry an Excel date serial instead of a year
        try:
            d = dt.date(1899, 12, 30) + dt.timedelta(days=raw)
            return d.isoformat(), d.year, "year field held an Excel date serial in source; decoded"
        except (OverflowError, ValueError):
            return None, None, f"unparseable year value: {raw!r}"
    s = str(raw).strip()
    m = re.match(r"^(\d{1,2})[.,](\d{4})$", s)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return None, year, "only month.year known in source"
    m = re.search(r"(\d{4})", s)
    if m and 1900 <= int(m.group(1)) <= 2100:
        return None, int(m.group(1)), f"irregular date string in source: {s!r}"
    return None, None, f"unparseable year value: {s!r}"


RABIES_RESERVOIR_MAP = {
    "agricultiral": "livestock",
    "agricultural": "livestock",
    "companion": "companion",
    "wild life": "wildlife",
    "wildlife": "wildlife",
}


def load_anthrax(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(raw_dir / "База с-я 2017.xlsx", sheet_name="РК")
    # header has two columns literally named "quantity ill" (dead + sick head
    # counts in the source); pandas auto-suffixes the second as
    # "quantity ill.1" -- keep the second one, it is the populated column.
    qty_col = "quantity ill.1" if "quantity ill.1" in df.columns else "quantity ill"

    records = []
    for i, row in df.iterrows():
        event_date, year, note = parse_anthrax_year(row.get("year"))
        species_raw = str(row.get("species") or "").strip()
        human_linked = species_raw.lower().startswith("man")
        lat, lat_note = parse_loose_coordinate(row.get("lat_dd (широта)"))
        lon, lon_note = parse_loose_coordinate(row.get("lon_dd"))
        lat, lon, bounds_note = check_kazakhstan_bounds(lat, lon)
        count, count_note = parse_loose_count(row.get(qty_col))
        notes = [n for n in (note, lat_note, lon_note, bounds_note, count_note) if n]
        if human_linked:
            notes.append("source species field indicates a linked human case: " + species_raw)
        records.append({
            "focus_id": f"AX-{i+1:05d}",
            "disease": "anthrax",
            "region_oblast": normalize_oblast(row.get("oblast")),
            "district_raion": str(row.get("rayon") or "").strip() or None,
            "settlement_name": str(row.get("selsky okrug") or "").strip() or None,
            "latitude": lat,
            "longitude": lon,
            "event_date": event_date,
            "year": year,
            "species": species_raw.replace("man, ", "").replace("man", "unknown") or "unknown",
            "reservoir_category": None,
            "animal_count": count,
            "confirmation_status": "confirmed",  # source is a historical registry of registered foci
            "data_source": "База с-я 2017.xlsx",
            "notes": "; ".join(notes) or None,
        })
    return pd.DataFrame.from_records(records)


def load_rabies(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(raw_dir / "Rabies_2018.xlsx", sheet_name="Лист1")
    records = []
    for i, row in df.iterrows():
        d = row.get("date_")
        event_date = d.date().isoformat() if isinstance(d, dt.datetime) else None
        reservoir = RABIES_RESERVOIR_MAP.get(str(row.get("species") or "").strip().lower(), "unknown")
        records.append({
            "focus_id": f"RB-{i+1:05d}",
            "disease": "rabies",
            "region_oblast": normalize_oblast(row.get("oblast")),
            "district_raion": str(row.get("district") or "").strip() or None,
            "settlement_name": str(row.get("rural_coun") or "").strip() or None,
            "latitude": row.get("lat_dd"),
            "longitude": row.get("lon_dd"),
            "event_date": event_date,
            "year": row.get("year"),
            "species": "unknown",  # source records reservoir category, not the literal animal species
            "reservoir_category": reservoir,
            "animal_count": row.get("quantity_i"),
            "confirmation_status": "confirmed",
            "data_source": "Rabies_2018.xlsx",
            "notes": None,
        })
    return pd.DataFrame.from_records(records)


def load_fmd(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(raw_dir / "Baza FMD_2013.xlsx", sheet_name="Лист1")
    records = []
    for i, row in df.iterrows():
        year = row.get("year")
        month = row.get("month")
        event_date = None
        try:
            if pd.notna(year) and pd.notna(month):
                event_date = dt.datetime.strptime(f"{int(year)} {month}", "%Y %B").date().isoformat()
        except ValueError:
            pass
        serotype = row.get("Serotype")
        tot_anim = row.get("tot_anim")
        notes = []
        if pd.notna(serotype):
            notes.append(f"serotype {serotype}")
        if pd.notna(tot_anim):
            notes.append(f"herd size at time of outbreak: {int(tot_anim)}")
        records.append({
            "focus_id": f"FMD-{i+1:05d}",
            "disease": "fmd",
            "region_oblast": normalize_oblast(row.get("oblast")),
            "district_raion": str(row.get("Raion") or "").strip() or None,
            "settlement_name": str(row.get("Selo") or "").strip() or None,
            "latitude": row.get("lat_dd"),
            "longitude": row.get("lon_dd"),
            "event_date": event_date,
            "year": int(year) if pd.notna(year) else None,
            "species": str(row.get("species") or "unknown").strip(),
            "reservoir_category": None,
            "animal_count": row.get("quantity ill"),
            "confirmation_status": "confirmed",
            "data_source": "Baza FMD_2013.xlsx",
            "notes": "; ".join(notes) or None,
        })
    return pd.DataFrame.from_records(records)


def quality_report(name: str, df: pd.DataFrame) -> list[str]:
    lines = [f"## {name}", f"- rows: {len(df)}"]
    missing_coords = df["latitude"].isna().sum() + df["longitude"].isna().sum()
    lines.append(f"- rows missing latitude or longitude: {df['latitude'].isna().sum()} / {df['longitude'].isna().sum()}")
    lines.append(f"- rows with no event_date and no year: {(df['event_date'].isna() & df['year'].isna()).sum()}")
    if "year" in df:
        years = df["year"].dropna()
        if len(years):
            lines.append(f"- year range: {int(years.min())}–{int(years.max())}")
    lines.append(f"- distinct region_oblast values after normalization: {df['region_oblast'].nunique()}")
    dupes = df.duplicated(subset=["region_oblast", "district_raion", "settlement_name", "event_date", "species"]).sum()
    lines.append(f"- exact duplicate rows (same place/date/species): {dupes}")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[2] / "dataset")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "normalized")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaders = {"anthrax": load_anthrax, "rabies": load_rabies, "fmd": load_fmd}
    report_lines = ["# Data quality report (auto-generated by etl/normalize.py)", ""]

    all_frames = []
    for name, loader in loaders.items():
        df = loader(args.raw_dir)
        out_path = args.out_dir / f"{name}_normalized.csv"
        df.to_csv(out_path, index=False)
        print(f"{name}: {len(df)} rows -> {out_path}")
        report_lines += quality_report(name, df)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = args.out_dir / "focus_unified.csv"
    combined.to_csv(combined_path, index=False)
    print(f"combined: {len(combined)} rows -> {combined_path}")

    report_path = args.out_dir / "QUALITY_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"quality report -> {report_path}")


if __name__ == "__main__":
    main()

"""
build_synthetic.py
------------------
Transforms raw 5-minute speed-band snapshots into a full 24-hour synthetic
hourly dataset.

For each unique (date, LinkID) present in the raw data, all 24 UTC hours are
generated. Hours with actual collected snapshots use the observed speed; hours
without observations get a synthetically estimated speed derived from the road
category's free-flow speed scaled by a Singapore time-of-day speed profile.

Output fields:
    timestamp_hour  – hour-truncated UTC timestamp (all 24 hours per date/link)
    LinkID          – road segment identifier
    RoadName        – road name
    RoadCategory    – LTA road category (1=Expressway … 6=Minor Access)
    weekday         – 0=Monday … 6=Sunday
    speed           – speed (km/h): observed or synthetically estimated
    lower_speed     – lower bound of the speed band
    upper_speed     – upper bound of the speed band
    speed_source    – 'observed' or 'synthetic'
    volume          – vehicle count (veh/hr), model or real-calibrated
    volume_source   – 'real' or 'model'
"""

import os
import glob
import json
import datetime
import numpy as np
import pandas as pd

# =========================
# PATHS
# =========================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_DIR         = os.path.join(BASE_DIR, "..", "..", "data_collection", "speed_band", "raw_datasets")
VOLUME_RAW_DIR  = os.path.join(BASE_DIR, "..", "..", "data_collection", "traffic_volume", "raw_datasets")
OUTPUT_DIR      = os.path.join(BASE_DIR, "..", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORY_NAME = {
    1: "expressway",
    2: "major_arterial",
    3: "arterial",
    4: "minor_arterial",
    5: "local_access",
    6: "minor_access",
}

# =========================
# ROAD CATEGORY PARAMETERS
#   capacity  : max vehicles/hr (one direction, typical Singapore values)
#   free_flow : free-flow speed in km/h
# =========================

ROAD_PARAMS = {
    1: {"capacity": 4500, "free_flow": 90},   # Expressway
    2: {"capacity": 1800, "free_flow": 70},   # Major Arterial
    3: {"capacity": 1200, "free_flow": 60},   # Arterial
    4: {"capacity":  800, "free_flow": 50},   # Minor Arterial
    5: {"capacity":  400, "free_flow": 40},   # Local Access
    6: {"capacity":  200, "free_flow": 30},   # Minor Access
}

DEFAULT_PARAMS = {"capacity": 800, "free_flow": 50}

# LTA speed band boundaries (band index → (min_speed, max_speed))
SPEED_BANDS = {
    1: (0,  9),  2: (10, 19), 3: (20, 29), 4: (30, 39),
    5: (40, 49), 6: (50, 59), 7: (60, 69), 8: (70, 79),
}


def speed_to_band(speed: float) -> tuple:
    for band, (lo, hi) in SPEED_BANDS.items():
        if lo <= speed <= hi:
            return float(lo), float(hi)
    # Clamp to nearest boundary
    if speed < 0:
        return 0.0, 9.0
    return 70.0, 79.0


# =========================
# TIME-OF-DAY PROFILES (SGT-based)
# =========================

# Demand multiplier (0–1) by SGT hour
_DEMAND_PROFILE = {
    0: 0.15, 1: 0.10, 2: 0.08, 3: 0.07, 4: 0.08,
    5: 0.15, 6: 0.35, 7: 0.75, 8: 0.95, 9: 0.80,
    10: 0.65, 11: 0.70, 12: 0.80, 13: 0.75, 14: 0.65,
    15: 0.70, 16: 0.85, 17: 1.00, 18: 0.95, 19: 0.80,
    20: 0.65, 21: 0.55, 22: 0.40, 23: 0.25,
}

_AVG_DEMAND = sum(_DEMAND_PROFILE.values()) / 24


def time_of_day_factor(utc_hour: int) -> float:
    """Demand multiplier for the given UTC hour."""
    return _DEMAND_PROFILE.get((utc_hour + 8) % 24, 0.5)


def synthetic_speed(road_cat: int, utc_hour: int, rng: np.random.Generator) -> float:
    """
    Estimate speed for a missing-observation hour.
    High demand → congested → lower speed; low demand → near free-flow.
    Speed factor = 1 - 0.45 * demand_factor  (so peak demand → ~55% of free-flow)
    """
    params     = ROAD_PARAMS.get(int(road_cat), DEFAULT_PARAMS)
    free_flow  = params["free_flow"]
    demand     = time_of_day_factor(utc_hour)
    spd_factor = 1.0 - 0.45 * demand
    speed      = free_flow * spd_factor
    noise      = rng.normal(loc=1.0, scale=0.05)   # ±5 % noise on speed
    return round(max(1.0, speed * noise), 1)


# =========================
# SYNTHETIC VOLUME MODEL (triangular fundamental diagram)
# =========================

def synthetic_volume(speed: float, road_cat: int, utc_hour: int, weekday: int,
                     rng: np.random.Generator) -> int:
    params    = ROAD_PARAMS.get(int(road_cat), DEFAULT_PARAMS)
    capacity  = params["capacity"]
    free_flow = params["free_flow"]
    critical  = 0.6 * free_flow

    if speed <= 0:
        ratio = 0.0
    elif speed <= critical:
        ratio = speed / critical
    else:
        ratio = 1.0 - (speed - critical) / max(free_flow - critical, 1)
        ratio = max(ratio, 0.0)

    tod_mult     = time_of_day_factor(utc_hour)
    weekend_mult = 0.75 if weekday >= 5 else 1.0
    volume       = capacity * ratio * tod_mult * weekend_mult
    noise        = rng.normal(loc=1.0, scale=0.10)
    return max(0, int(round(volume * noise)))


# =========================
# LOAD HISTORICAL VOLUME DATA
# =========================

print("Loading historical traffic volume data …")

volume_files = glob.glob(os.path.join(VOLUME_RAW_DIR, "traffic_flow_*.json"))

if volume_files:
    vol_records = []
    for vf in volume_files:
        with open(vf, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        vol_records.extend(raw.get("Value", []))

    vol_df = pd.DataFrame(vol_records)
    vol_df["LinkID"]  = vol_df["LinkID"].astype(str)
    vol_df["Volume"]  = pd.to_numeric(vol_df["Volume"], errors="coerce")
    vol_df = vol_df.dropna(subset=["LinkID", "Volume"])

    vol_lookup = (
        vol_df.groupby("LinkID")["Volume"]
        .mean()
        .reset_index()
        .rename(columns={"Volume": "ref_volume"})
    )
    print(f"  Volume lookup: {len(vol_lookup):,} unique links with historical counts")
else:
    vol_lookup = pd.DataFrame(columns=["LinkID", "ref_volume"])
    print("  No traffic volume files found — will use model for all links.")


# =========================
# LOAD RAW SPEED BAND DATA
# =========================

print("Loading raw speed band CSV files …")

csv_files = glob.glob(os.path.join(RAW_DIR, "speedbands_*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No raw CSV files found in: {RAW_DIR}")

print(f"  Found {len(csv_files)} file(s).")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
df = df.dropna(subset=["Timestamp", "LinkID", "RoadCategory", "MinimumSpeed", "MaximumSpeed"])

# Filter invalid road categories (PRD defines categories 1-6 only)
valid_cats = set(ROAD_PARAMS.keys())
before = len(df)
df = df[df["RoadCategory"].isin(valid_cats)]
print(f"  Dropped {before - len(df):,} rows with invalid RoadCategory (kept: {sorted(df['RoadCategory'].unique())})")

# Fix LTA sentinel: MaximumSpeed=999 means SpeedBand 8 has no upper limit.
# Replace with the road category free-flow speed so the midpoint is realistic.
sentinel_mask = df["MaximumSpeed"] >= 200
df.loc[sentinel_mask, "MaximumSpeed"] = df.loc[sentinel_mask, "RoadCategory"].map(
    {k: v["free_flow"] for k, v in ROAD_PARAMS.items()}
)
print(f"  Fixed {sentinel_mask.sum():,} sentinel MaximumSpeed=999 rows using road category free-flow speed")

df["speed_obs"]      = (df["MinimumSpeed"] + df["MaximumSpeed"]) / 2.0
df["timestamp_hour"] = df["Timestamp"].dt.floor("h")

print(f"  Total rows after cleaning: {len(df):,}")


# =========================
# HOURLY AGGREGATION OF OBSERVED DATA
# =========================

print("Aggregating observed snapshots to hourly …")

observed = (
    df.groupby(["timestamp_hour", "LinkID", "RoadName", "RoadCategory"])
    .agg(
        speed       = ("speed_obs",    "mean"),
        lower_speed = ("MinimumSpeed", "mean"),
        upper_speed = ("MaximumSpeed", "mean"),
    )
    .reset_index()
)
observed["speed"]       = observed["speed"].round(1)
observed["lower_speed"] = observed["lower_speed"].round(1)
observed["upper_speed"] = observed["upper_speed"].round(1)
observed["speed_source"] = "observed"

print(f"  Observed hourly rows: {len(observed):,}")


# =========================
# EXPAND TO FULL 24-HOUR GRID PER DATE PER LINK
# =========================

print("Expanding to full 24-hour grid …")

rng = np.random.default_rng(seed=42)

# Unique link metadata (one row per link with stable RoadName/RoadCategory/coords)
link_meta = (
    df[["LinkID", "RoadName", "RoadCategory", "StartLat", "StartLon", "EndLat", "EndLon"]]
    .drop_duplicates(subset=["LinkID"])
    .reset_index(drop=True)
)
link_meta["LinkID"] = link_meta["LinkID"].astype(int)

# Unique UTC dates present in the raw data (tz-aware)
dates = pd.DatetimeIndex(df["timestamp_hour"].dt.normalize().unique())

# Build full grid: every (date, hour 0-23) × every link
hours_range = pd.timedelta_range(start="0h", periods=24, freq="h")
date_hours  = pd.DatetimeIndex([d + h for d in dates for h in hours_range])

grid = pd.DataFrame({"timestamp_hour": date_hours})
grid = grid.merge(link_meta.assign(key=1), how="cross" if hasattr(pd.DataFrame, "merge") else "inner")

# Cross join
grid = link_meta.assign(_key=1).merge(
    pd.DataFrame({"timestamp_hour": date_hours, "_key": 1}),
    on="_key"
).drop(columns="_key")

# Merge observed data in
grid = grid.merge(
    observed.assign(LinkID=lambda x: x["LinkID"].astype(int)),
    on=["timestamp_hour", "LinkID", "RoadName", "RoadCategory"],
    how="left"
)

# Fill missing hours with synthetic speed — VECTORIZED
missing_mask = grid["speed_source"].isna()
print(f"  Missing hour-link slots to synthesise: {missing_mask.sum():,}")

if missing_mask.any():
    mg = grid[missing_mask].copy()
    utc_hours = mg["timestamp_hour"].dt.hour.values
    sgt_hours = (utc_hours + 8) % 24

    # Demand factor per row
    demand_arr = np.array([_DEMAND_PROFILE.get(int(h), 0.5) for h in sgt_hours])

    # Free-flow speed per road category
    ff_arr = mg["RoadCategory"].map(
        {k: v["free_flow"] for k, v in ROAD_PARAMS.items()}
    ).fillna(DEFAULT_PARAMS["free_flow"]).values

    # Synthetic speed: high demand → congested → lower speed; clamp [1, free_flow]
    noise_spd = rng.normal(loc=1.0, scale=0.05, size=len(mg))
    spd_arr   = np.clip((1.0 - 0.45 * demand_arr) * ff_arr * noise_spd, 1.0, ff_arr).round(1)

    # Speed band lower/upper from speed value
    lo_arr = np.zeros(len(spd_arr))
    hi_arr = np.zeros(len(spd_arr))
    for band, (lo, hi) in SPEED_BANDS.items():
        mask_band = (spd_arr >= lo) & (spd_arr <= hi)
        lo_arr[mask_band] = float(lo)
        hi_arr[mask_band] = float(hi)
    # Clamp out-of-range
    out_of_range = lo_arr == 0
    lo_arr[out_of_range & (spd_arr < 0)]  = 0.0
    hi_arr[out_of_range & (spd_arr < 0)]  = 9.0
    lo_arr[out_of_range & (spd_arr > 79)] = 70.0
    hi_arr[out_of_range & (spd_arr > 79)] = 79.0

    grid.loc[missing_mask, "speed"]        = spd_arr
    grid.loc[missing_mask, "lower_speed"]  = lo_arr
    grid.loc[missing_mask, "upper_speed"]  = hi_arr
    grid.loc[missing_mask, "speed_source"] = "synthetic"

grid["weekday"] = grid["timestamp_hour"].dt.weekday

print(f"  Total grid rows: {len(grid):,}")


# =========================
# VOLUME: REAL-CALIBRATED OR MODEL
# =========================

# Volume assignment — VECTORIZED
print("Assigning volume …")

grid["LinkID_str"] = grid["LinkID"].astype(str)

if not vol_lookup.empty:
    grid = grid.merge(vol_lookup, left_on="LinkID_str", right_on="LinkID",
                      how="left", suffixes=("", "_vol"))
else:
    grid["ref_volume"] = np.nan

utc_h    = grid["timestamp_hour"].dt.hour.values
sgt_h    = (utc_h + 8) % 24
tod_arr  = np.array([_DEMAND_PROFILE.get(int(h), 0.5) for h in sgt_h])
wknd_arr = np.where(grid["weekday"].values >= 5, 0.75, 1.0)

# Road category arrays for model fallback
ff_vol   = grid["RoadCategory"].map({k: v["free_flow"]  for k, v in ROAD_PARAMS.items()}).fillna(DEFAULT_PARAMS["free_flow"]).values
cap_arr  = grid["RoadCategory"].map({k: v["capacity"]   for k, v in ROAD_PARAMS.items()}).fillna(DEFAULT_PARAMS["capacity"]).values
spd_arr  = grid["speed"].values
crit_arr = 0.6 * ff_vol

# Triangular flow ratio
ratio_arr = np.where(
    spd_arr <= 0, 0.0,
    np.where(
        spd_arr <= crit_arr,
        spd_arr / np.maximum(crit_arr, 1),
        np.clip(1.0 - (spd_arr - crit_arr) / np.maximum(ff_vol - crit_arr, 1), 0.0, 1.0)
    )
)
model_vol = cap_arr * ratio_arr * tod_arr * wknd_arr
noise_vol = rng.normal(loc=1.0, scale=0.10, size=len(grid))
model_vol = np.maximum(0, np.round(model_vol * noise_vol)).astype(int)

# Real-calibrated volume where ref_volume exists
has_real  = grid["ref_volume"].notna().values
scaled    = grid["ref_volume"].fillna(0).values * (tod_arr / _AVG_DEMAND) * wknd_arr
noise_rv  = rng.normal(loc=1.0, scale=0.10, size=len(grid))
real_vol  = np.maximum(0, np.round(scaled * noise_rv)).astype(int)

grid["volume"]       = np.where(has_real, real_vol, model_vol)
grid["volume_source"] = np.where(has_real, "real", "model")

real_pct = has_real.mean() * 100
print(f"  Real volume used:  {real_pct:.1f}%")
print(f"  Model fallback:    {100 - real_pct:.1f}%")


# =========================
# FINAL COLUMN ORDER & SAVE
# =========================

# Convert timestamp from UTC → SGT (UTC+8)
SGT = datetime.timezone(datetime.timedelta(hours=8))
grid["timestamp_hour"] = grid["timestamp_hour"].dt.tz_convert(SGT)

# Derive hour (integer) and day_type from the SGT timestamp
grid["hour"]     = grid["timestamp_hour"].dt.hour
grid["day_type"] = grid["weekday"].apply(lambda w: "Weekend" if w >= 5 else "Weekday")

# Keep only PRD-specified columns + optional feature columns
# ─── Synthetic optional columns ──────────────────────────────────────────────
# Assign per-LinkID so the same road segment always carries the same code.
# expressway  : E1–E8, only for RoadCategory=1; NaN for all others
# road_direction: D1/D2, one fixed direction per LinkID
# source       : S1/S2/S3, random per row (simulates mixed data sources)

link_ids = grid["LinkID"].unique()
link_rng = np.random.default_rng(seed=99)

expressway_codes = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
direction_codes  = ["D1", "D2"]
source_codes     = ["S1", "S2", "S3"]

# Per-LinkID assignments (consistent across rows)
link_expressway = {
    lid: link_rng.choice(expressway_codes) for lid in link_ids
}
link_direction = {
    lid: link_rng.choice(direction_codes) for lid in link_ids
}

grid["expressway"]     = grid.apply(
    lambda r: link_expressway[r["LinkID"]] if r["RoadCategory"] == 1 else np.nan, axis=1
)
grid["road_direction"] = grid["LinkID"].map(link_direction)
grid["source"]         = link_rng.choice(source_codes, size=len(grid))

output = grid[[
    "timestamp_hour",
    "LinkID",
    "RoadCategory",
    "hour",
    "day_type",
    "speed",
    "volume",
    "expressway",
    "road_direction",
    "source",
]].sort_values(["LinkID", "timestamp_hour"]).reset_index(drop=True)

# Save one CSV per road category
for cat, cat_df in output.groupby("RoadCategory"):
    cat_label = CATEGORY_NAME.get(cat, f"category_{cat}")
    out_path = os.path.join(OUTPUT_DIR, f"synthetic_hourly_cat{cat}_{cat_label}.csv")
    cat_df.sort_values(["LinkID", "timestamp_hour"]).reset_index(drop=True).to_csv(out_path, index=False)
    print(f"  Cat {cat} ({cat_label}): {len(cat_df):,} rows → {os.path.basename(out_path)}")

print(f"\nTotal rows: {len(output):,}  |  Columns: {list(output.columns)}")
print(f"Unique hours per day: {output['hour'].nunique()}")
print(f"Unique dates: {output['timestamp_hour'].dt.date.nunique()}")
print(f"Unique LinkIDs: {output['LinkID'].nunique():,}")
print(f"Day type distribution:\n{output['day_type'].value_counts().to_string()}")

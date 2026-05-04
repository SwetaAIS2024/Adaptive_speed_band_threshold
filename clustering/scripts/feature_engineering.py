"""
Feature engineering for adaptive speed band clustering.

For each (RoadCategory x day_type) subset:
  - Cyclic-encode hour: sin(2πh/24), cos(2πh/24)
  - Binary-encode day_type: Weekday=0, Weekend=1
  - MinMax-normalise speed and volume *within the subset*

Reads (in order of preference):
  1. Per-category CSVs: synthetic_dataset/processed/synthetic_hourly_cat*.csv
  2. Single input file:  data/input/traffic_hourly.csv
Writes: clustering/features/features_<cat>_<day>.parquet  (one file per subset)
        clustering/features/feature_metadata.json          (scaler params + row counts)
"""

import json
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from clustering.pipeline import (
    VALID_CATEGORIES, DAY_TYPES, CATEGORY_PREFIX, DAY_PREFIX,
    REQUIRED_COLS, FEATURE_COLS, engineer_subset,
)


SYNTHETIC_DIR = Path("synthetic_dataset/processed")
INPUT_FILE = Path("data/input/traffic_hourly.csv")
OUTPUT_DIR = Path("clustering/features")


def load_input() -> pd.DataFrame:
    """Load from per-category CSVs if available, otherwise fall back to single input file."""
    cat_files = sorted(SYNTHETIC_DIR.glob("synthetic_hourly_cat*.csv"))
    if cat_files:
        print(f"Loading {len(cat_files)} per-category CSV(s) from {SYNTHETIC_DIR} ...")
        frames = [pd.read_csv(f, parse_dates=["timestamp_hour"]) for f in cat_files]
        df = pd.concat(frames, ignore_index=True)
        print(f"  {len(df):,} rows loaded from {len(cat_files)} file(s)")
        return df

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp_hour"])
    print(f"  {len(df):,} rows loaded")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input dataset is missing columns: {missing}")

    metadata = {}

    for cat in sorted(df["RoadCategory"].unique()):
        if cat not in VALID_CATEGORIES:
            print(f"  [SKIP] RoadCategory={cat} not in valid set {VALID_CATEGORIES}")
            continue

        for day in DAY_TYPES:
            subset = df[(df["RoadCategory"] == cat) & (df["day_type"] == day)]

            if len(subset) == 0:
                print(f"  [SKIP] Cat={cat} {day}: no rows")
                continue

            engineered, scaler, active_cols = engineer_subset(subset)

            save_cols = [c for c in df.columns] + active_cols
            out_path = OUTPUT_DIR / f"features_{cat}_{day.lower()}.parquet"
            engineered[save_cols].to_parquet(out_path, index=False)

            key = f"{cat}_{day}"
            metadata[key] = {
                "road_category": int(cat),
                "day_type": day,
                "cat_prefix": CATEGORY_PREFIX.get(cat, f"C{cat}"),
                "day_prefix": DAY_PREFIX[day],
                "n_rows": len(subset),
                "speed_min": float(subset["speed"].min()),
                "speed_max": float(subset["speed"].max()),
                "volume_min": float(subset["volume"].min()),
                "volume_max": float(subset["volume"].max()),
                "scaler_data_min": scaler.data_min_.tolist(),
                "scaler_data_max": scaler.data_max_.tolist(),
                "active_cols": active_cols,
                "feature_file": str(out_path),
            }
            print(f"  Cat={cat} {day:8s}: {len(subset):>8,} rows -> {out_path.name}")

    meta_path = OUTPUT_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {meta_path}")
    print("Feature engineering complete.")


if __name__ == "__main__":
    main()

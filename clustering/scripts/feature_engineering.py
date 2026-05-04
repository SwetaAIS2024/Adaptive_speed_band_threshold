"""
Feature engineering for adaptive speed band clustering.

For each preprocessed file in data/pre_processed_dataset/:
  - Cyclic-encode  hour     : drop raw 'hour'; insert hour_sin = sin(2πh/24)
                              and hour_cos = cos(2πh/24) at the same position
  - Binary-encode  day_type : Weekday → 0, Weekend → 1  (in-place, same column)
  - MinMax-normalise speed  : in-place, per file
  - MinMax-normalise volume : in-place, per file (only if column is present)

No columns are added beyond the encoding replacements listed above.

Reads : data/pre_processed_dataset/*_cleaned.csv
        (skips preprocessing_summary.csv and eda_report.txt)
Writes: clustering/features/features_<stem>.parquet  (one per input file)
        clustering/features/feature_metadata.json    (scaler params + encoding info)

Run:
    python clustering/scripts/feature_engineering.py
"""

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo root ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
PREPROC_DIR = BASE_DIR / "data" / "pre_processed_dataset"
OUTPUT_DIR  = BASE_DIR / "clustering" / "features"

# ── Load EDA module for file discovery (absolute path, Docker-safe) ───────────
_eda_path = BASE_DIR / "data" / "data_understanding_EDA.py"
_spec = importlib.util.spec_from_file_location("data_understanding_EDA", str(_eda_path))
_eda  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eda)

# Files to skip inside the preprocessed directory
SKIP_FILES = {"preprocessing_summary.csv", "eda_report.txt"}

DAY_TYPE_MAP = {"weekday": 0, "weekend": 1}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column matching any candidate name (case-insensitive)."""
    col_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


# ── Per-file feature engineering ─────────────────────────────────────────────

def engineer_file(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load one preprocessed CSV, apply all encodings, and return
    (engineered_df, metadata_dict).

    Transformations applied in-place (no extra columns beyond replacements):
      hour     → hour_sin + hour_cos  (raw hour column removed)
      day_type → 0 / 1               (same column, overwritten)
      speed    → MinMax [0, 1]        (same column, overwritten)
      volume   → MinMax [0, 1]        (same column, overwritten, if present)
    """
    df = pd.read_csv(path)
    fname = os.path.basename(path)
    meta: dict = {"file": fname, "rows_raw": len(df), "transformations": []}

    # Detect relevant columns (case-insensitive)
    hour_col    = _find_col(df, "hour")
    daytype_col = _find_col(df, "day_type", "daytype")
    speed_col   = _find_col(df, "speed")
    volume_col  = _find_col(df, "volume")

    # 1. Cyclic encode hour — replace raw column with sin/cos pair ────────────
    if hour_col:
        h = df[hour_col].astype(float)
        idx = df.columns.get_loc(hour_col)
        df.insert(idx,     "hour_sin", np.sin(2 * math.pi * h / 24))
        df.insert(idx + 1, "hour_cos", np.cos(2 * math.pi * h / 24))
        df = df.drop(columns=[hour_col])
        meta["transformations"].append(
            f"'{hour_col}' cyclic-encoded → hour_sin, hour_cos (raw column dropped)"
        )
    else:
        meta["transformations"].append("hour: column not found — skipped")

    # 2. Binary encode day_type in-place ──────────────────────────────────────
    if daytype_col:
        encoded = (
            df[daytype_col]
            .astype(str).str.strip().str.lower()
            .map(DAY_TYPE_MAP)
        )
        unmapped = int(encoded.isna().sum())
        df[daytype_col] = encoded.fillna(-1).astype(int)
        note = f"'{daytype_col}' binary-encoded in-place (Weekday=0, Weekend=1)"
        if unmapped:
            note += f"; {unmapped} unmapped values set to -1"
        meta["transformations"].append(note)
    else:
        meta["transformations"].append("day_type: column not found — skipped")

    # 3. MinMax normalise speed in-place ──────────────────────────────────────
    if speed_col:
        s_min = float(df[speed_col].min())
        s_max = float(df[speed_col].max())
        rng = s_max - s_min
        df[speed_col] = (df[speed_col] - s_min) / rng if rng > 0 else 0.0
        meta["speed_min_raw"] = s_min
        meta["speed_max_raw"] = s_max
        meta["transformations"].append(
            f"'{speed_col}' MinMax-normalised in-place [{s_min:.2f}, {s_max:.2f}] → [0, 1]"
        )
    else:
        meta["transformations"].append("speed: column not found — skipped")

    # 4. MinMax normalise volume in-place (if present) ────────────────────────
    if volume_col:
        v_min = float(df[volume_col].min())
        v_max = float(df[volume_col].max())
        rng = v_max - v_min
        df[volume_col] = (df[volume_col] - v_min) / rng if rng > 0 else 0.0
        meta["volume_min_raw"] = v_min
        meta["volume_max_raw"] = v_max
        meta["transformations"].append(
            f"'{volume_col}' MinMax-normalised in-place [{v_min:.2f}, {v_max:.2f}] → [0, 1]"
        )
    else:
        meta["transformations"].append("volume: column not found — skipped")

    meta["rows_out"] = len(df)
    meta["columns"]  = list(df.columns)
    return df, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover preprocessed CSVs via EDA module — no hardcoded filenames
    all_files = _eda.discover_raw_files(str(PREPROC_DIR))
    input_files = {
        name: path
        for name, path in all_files.items()
        if os.path.basename(path) not in SKIP_FILES
    }

    if not input_files:
        print(f"No preprocessed files found in {PREPROC_DIR}.")
        print("Run  python data/preprocess.py  first.")
        sys.exit(1)

    print(f"\nFound {len(input_files)} preprocessed file(s) in {PREPROC_DIR}")
    all_metadata = {}

    for display_name, path in input_files.items():
        fname = os.path.basename(path)
        print(f"\n{'─' * 60}")
        print(f"  Processing: {fname}")

        df_out, meta = engineer_file(path)
        stem     = Path(fname).stem
        out_path = OUTPUT_DIR / f"features_{stem}.parquet"
        df_out.to_parquet(out_path, index=False)

        all_metadata[stem] = meta

        print(f"  Rows    : {meta['rows_out']:,}")
        print(f"  Columns : {meta['columns']}")
        for t in meta["transformations"]:
            print(f"  • {t}")
        print(f"  Saved → {out_path}")

    meta_path = OUTPUT_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'─' * 60}")
    print(f"Metadata saved → {meta_path}")
    print(f"Feature engineering complete. {len(input_files)} file(s) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


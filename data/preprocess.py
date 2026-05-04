"""
Preprocessing Script
====================
Discovers raw files dynamically from data/raw_dataset/ (via the EDA module),
uses EDA quality-check results to decide what to fix per file, and saves one
cleaned CSV per raw file into data/pre_processed_dataset/.

No filenames are hardcoded. The number of output files always equals the
number of raw input files discovered.

Fixes applied are driven entirely by what the EDA functions actually find
in each file:

  - Duplicate rows            → drop exact duplicates
  - Null values in key cols   → drop rows where any key column is null
                                (key cols = all non-datetime columns)
  - Zero speed                → drop if a speed column is detected
  - Zero volume               → drop if a volume column is detected
  - DATE_TIME string type     → parse to datetime; add derived `hour` column
                                and `date` column (date part only)
  - day_type                  → added to every file: "Weekday" or "Weekend"
                                derived from the source filename keyword
  - String column whitespace  → strip leading/trailing whitespace
  - Road-name case variants   → normalise to uppercase (e.g. "Cte Tunnel"
                                → "CTE TUNNEL") using upper-cased canonical
                                values derived from the actual data

Run:
    python data/preprocess.py
"""

import importlib.util
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "raw_dataset")
OUT_DIR  = os.path.join(BASE_DIR, "pre_processed_dataset")
SEPARATOR = "=" * 70

# ── Load EDA module by absolute path (no sys.path dependency) ────────────────
_spec = importlib.util.spec_from_file_location(
    "data_understanding_EDA",
    os.path.join(BASE_DIR, "data_understanding_EDA.py"),
)
_eda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eda)


def log(msg: str):
    print(msg)


# ── Generic preprocessing driven by EDA results ──────────────────────────────

def preprocess_file(raw_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load one raw file, run EDA quality checks, apply all fixes found,
    and return (cleaned_df, report_dict).
    """
    df = _eda.load_any_file(raw_path)
    dc = _eda._detect_cols(df)
    quality = _eda.get_quality_checks(df)

    report = {"rows_in": len(df), "file": os.path.basename(raw_path), "fixes": []}
    n = len(df)

    # 1. Strip whitespace from all string columns
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip()
    if str_cols:
        report["fixes"].append(f"Stripped whitespace from: {str_cols}")

    # 2. Normalise string columns to consistent case
    #    For columns that look like categorical codes / names (not datetime, not IDs),
    #    detect mixed-case variants of the same value and unify to uppercase.
    non_id_str_cols = [
        c for c in str_cols
        if c != dc["dt"]
        and c != dc["id"]
        and df[c].nunique() < 500   # skip high-cardinality ID-like columns
    ]
    normalised_cols = []
    for col in non_id_str_cols:
        # Build a map: lowercase → most-common original form (then force upper)
        vc = df[col].dropna().value_counts()
        canonical = {v.lower(): v.upper() for v in vc.index}
        before = df[col].nunique()
        df[col] = df[col].dropna().map(lambda x: canonical.get(x.lower(), x.upper()) if isinstance(x, str) else x)
        # Re-apply to full series (map loses NaN rows)
        df[col] = df[col].where(df[col].notna(), other=np.nan)
        after = df[col].nunique()
        if after < before:
            normalised_cols.append(f"{col}: {before} → {after} unique values")
    if normalised_cols:
        report["fixes"].append(f"Normalised case in: {normalised_cols}")

    # 3. Drop exact duplicate rows (EDA finding: TIQ ~40/54, SpeedGraph ~49.5%)
    dup_count = quality["dup_count"]
    if dup_count > 0:
        df = df.drop_duplicates()
        report["dropped_duplicates"] = int(dup_count)
        report["fixes"].append(f"Dropped {dup_count:,} duplicate rows")

    # 4. Drop rows with nulls in key columns (all non-datetime cols)
    key_cols = [c for c in df.columns if c != dc["dt"]]
    null_mask = df[key_cols].isnull().any(axis=1)
    null_count = int(null_mask.sum())
    if null_count > 0:
        df = df[~null_mask]
        report["dropped_nulls_in_key_cols"] = null_count
        report["fixes"].append(f"Dropped {null_count:,} rows with nulls in key columns ({key_cols})")

    # 5. Drop zero-speed rows (EDA finding: SpeedGraph has zero-speed records)
    if dc["speed"]:
        sc = dc["speed"]
        zero_speed = int((df[sc] == 0).sum())
        if zero_speed > 0:
            df = df[df[sc] > 0]
            report["dropped_zero_speed"] = zero_speed
            report["fixes"].append(f"Dropped {zero_speed:,} rows where {sc} == 0")

    # 6. Drop zero-volume rows (EDA finding: TIQ has zero-volume records)
    if dc["volume"]:
        vc_col = dc["volume"]
        zero_vol = int((df[vc_col] == 0).sum())
        if zero_vol > 0:
            df = df[df[vc_col] > 0]
            report["dropped_zero_volume"] = zero_vol
            report["fixes"].append(f"Dropped {zero_vol:,} rows where {vc_col} == 0")

    # 7. Parse datetime column and extract hour, date
    if dc["dt"]:
        dt_col = dc["dt"]
        df = df.copy()
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        unparseable = int(df[dt_col].isnull().sum())
        if unparseable > 0:
            df = df.dropna(subset=[dt_col])
            report["dropped_unparseable_datetime"] = unparseable
            report["fixes"].append(f"Dropped {unparseable:,} rows with unparseable {dt_col}")
        df["hour"] = df[dt_col].dt.hour
        df["date"] = df[dt_col].dt.date
        report["fixes"].append(f"Parsed {dt_col} to datetime; added 'hour' and 'date' columns")

    # 8. Add day_type derived from the source filename
    fname_lower = os.path.basename(raw_path).lower()
    if "weekday" in fname_lower:
        day_type_val = "Weekday"
    elif "weekend" in fname_lower:
        day_type_val = "Weekend"
    else:
        day_type_val = "Unknown"
    df["day_type"] = day_type_val
    report["fixes"].append(f"Added 'day_type' = '{day_type_val}' (derived from filename)")

    report["rows_out"] = len(df)
    report["dropped_total"] = report["rows_in"] - len(df)
    return df, report


def print_report(report: dict):
    log(f"\n{SEPARATOR}")
    log(f"  PREPROCESSING REPORT: {report['file']}")
    log(SEPARATOR)
    log(f"  Rows in        : {report['rows_in']:>10,}")
    log(f"  Rows out       : {report['rows_out']:>10,}")
    log(f"  Rows dropped   : {report['dropped_total']:>10,}")
    retention = report["rows_out"] / report["rows_in"] * 100
    log(f"  Retention      : {retention:>9.2f}%")
    log(f"\n  Fixes applied:")
    for fix in report["fixes"]:
        log(f"    • {fix}")


def run_preprocessing():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Discover raw files from EDA module — no hardcoding
    raw_files = _eda.discover_raw_files(RAW_DIR)  # {display_name: abs_path}

    if not raw_files:
        log(f"No raw files found in {RAW_DIR}. Exiting.")
        sys.exit(1)

    log(f"\nFound {len(raw_files)} raw file(s) in {RAW_DIR}")
    summary_rows = []

    for display_name, raw_path in raw_files.items():
        fname = os.path.basename(raw_path)
        log(f"\n{'#' * 70}")
        log(f"  Processing: {fname}")
        log(f"{'#' * 70}")

        df_clean, report = preprocess_file(raw_path)
        print_report(report)

        # Output filename = original stem + _cleaned.csv
        stem = os.path.splitext(fname)[0]
        out_name = f"{stem}_cleaned.csv"
        out_path = os.path.join(OUT_DIR, out_name)
        df_clean.to_csv(out_path, index=False)
        log(f"\n  Saved → {out_path}")

        summary_rows.append({
            "raw_file":      fname,
            "rows_raw":      report["rows_in"],
            "rows_cleaned":  report["rows_out"],
            "rows_dropped":  report["dropped_total"],
            "retention_%":   round(report["rows_out"] / report["rows_in"] * 100, 2),
            "output_file":   out_name,
        })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "preprocessing_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    log(f"\n{SEPARATOR}")
    log("  PREPROCESSING SUMMARY")
    log(SEPARATOR)
    log(summary_df.to_string(index=False))
    log(f"\n  Summary saved → {summary_path}")
    log(f"\n  Input files  : {len(raw_files)}")
    log(f"  Output files : {len(summary_rows)}  (1 per raw file)")
    log(f"\n{SEPARATOR}")
    log("  PREPROCESSING COMPLETE")
    log(SEPARATOR)


if __name__ == "__main__":
    run_preprocessing()

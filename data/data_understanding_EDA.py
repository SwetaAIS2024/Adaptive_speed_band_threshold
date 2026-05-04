import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_dataset")
OUT_DIR = os.path.join(BASE_DIR, "pre_processed_dataset")
OUT_FILE = os.path.join(OUT_DIR, "eda_report.txt")

FILES = {
    "TIQ_weekday": {
        "path": os.path.join(RAW_DIR, "TIQ sample data - weekday 20220713.xlsx"),
        "reader": "excel",
        "day_type": "weekday",
        "source": "TIQ",
    },
    "TIQ_weekend": {
        "path": os.path.join(RAW_DIR, "TIQ sample data-  weekend 20220710.xlsx"),
        "reader": "excel",
        "day_type": "weekend",
        "source": "TIQ",
    },
    "SpeedGraph_weekday": {
        "path": os.path.join(RAW_DIR, "Speed graph sample data - weekday 20210113.csv"),
        "reader": "csv",
        "day_type": "weekday",
        "source": "SpeedGraph",
    },
    "SpeedGraph_weekend": {
        "path": os.path.join(RAW_DIR, "Speed graph sample data - weekend 20210110.csv"),
        "reader": "csv",
        "day_type": "weekend",
        "source": "SpeedGraph",
    },
}

SEPARATOR = "=" * 70


def load_file(meta: dict) -> pd.DataFrame:
    if meta["reader"] == "excel":
        df = pd.read_excel(meta["path"])
    else:
        df = pd.read_csv(meta["path"])
    # Normalize column names: strip whitespace and surrounding quotes
    df.columns = df.columns.str.strip().str.strip('"').str.strip("'")
    return df


def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def explore_features(name: str, df: pd.DataFrame):
    section(f"[{name}] 1. FEATURE EXPLORATION")
    print(f"Shape           : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns         : {list(df.columns)}")
    print("\nData Types:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    print("\nLast 5 rows:")
    print(df.tail(5).to_string(index=False))


def check_data_quality(name: str, df: pd.DataFrame):
    section(f"[{name}] 2. DATA QUALITY CHECKS")

    print("\n-- Null / Missing Values --")
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().mean() * 100).round(2)
    null_report = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
    print(null_report.to_string())

    print("\n-- Duplicate Rows --")
    dup_count = df.duplicated().sum()
    print(f"Total duplicates : {dup_count} ({dup_count / len(df) * 100:.2f}%)")

    print("\n-- Negative / Zero Values in Numeric Columns --")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        neg = (df[col] < 0).sum()
        zeros = (df[col] == 0).sum()
        print(f"  {col:20s} -> negatives: {neg:5d} | zeros: {zeros:5d}")

    if "SPEED" in df.columns:
        print("\n-- Speed Anomalies (SPEED < 0 or SPEED > 200) --")
        anomalies = df[(df["SPEED"] < 0) | (df["SPEED"] > 200)]
        print(f"  Anomalous speed rows: {len(anomalies)}")
        if len(anomalies) > 0:
            print(anomalies.to_string(index=False))

    if "VOLUME" in df.columns:
        print("\n-- Volume Anomalies (VOLUME < 0) --")
        neg_vol = df[df["VOLUME"] < 0]
        print(f"  Negative volume rows: {len(neg_vol)}")

    print("\n-- DATE_TIME Parsing Check --")
    try:
        parsed = pd.to_datetime(df["DATE_TIME"])
        invalid_dates = parsed.isnull().sum()
        print(f"  Unparseable DATE_TIME values: {invalid_dates}")
        print(f"  Date range: {parsed.min()}  -->  {parsed.max()}")
    except Exception as e:
        print(f"  Could not parse DATE_TIME: {e}")


def explore_categories(name: str, df: pd.DataFrame):
    section(f"[{name}] 3. CATEGORICAL FEATURE ANALYSIS")

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    date_like = ["DATE_TIME"]
    cat_cols = [c for c in cat_cols if c not in date_like]

    if not cat_cols:
        print("  No categorical columns found (excluding DATE_TIME).")
        return

    for col in cat_cols:
        unique_vals = df[col].nunique()
        print(f"\n  Column: {col}")
        print(f"    Unique count : {unique_vals}")
        val_counts = df[col].value_counts(dropna=False)
        print(f"    Top 15 values:")
        print(val_counts.head(15).to_string())


def general_statistics(name: str, df: pd.DataFrame):
    section(f"[{name}] 4. GENERAL STATISTICAL SUMMARY")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        print("\n-- Descriptive Statistics (Numeric) --")
        print(df[numeric_cols].describe().round(2).to_string())

    print("\n-- Temporal Analysis --")
    try:
        parsed = pd.to_datetime(df["DATE_TIME"])
        df = df.copy()
        df["_dt"] = parsed
        df["_hour"] = df["_dt"].dt.hour
        df["_date"] = df["_dt"].dt.date

        print(f"\n  Unique dates     : {df['_date'].nunique()}")
        print(f"  Date list:")
        print(f"  {sorted(df['_date'].unique())}")

        print(f"\n  Records per hour (all dates combined):")
        hour_counts = df["_hour"].value_counts().sort_index()
        print(hour_counts.to_string())

        if "SPEED" in df.columns:
            print(f"\n  Mean SPEED by hour:")
            print(df.groupby("_hour")["SPEED"].mean().round(2).to_string())

        if "VOLUME" in df.columns:
            print(f"\n  Mean VOLUME by hour:")
            print(df.groupby("_hour")["VOLUME"].mean().round(2).to_string())

    except Exception as e:
        print(f"  Temporal analysis skipped: {e}")

    if "ROAD_NAME" in df.columns and "SPEED" in df.columns:
        print(f"\n-- Mean SPEED by ROAD_NAME --")
        road_speed = df.groupby("ROAD_NAME")["SPEED"].agg(["count", "mean", "min", "max"]).round(2)
        road_speed.columns = ["count", "mean_speed", "min_speed", "max_speed"]
        print(road_speed.to_string())

    if "EQUIP_ID" in df.columns:
        print(f"\n-- Equipment summary (top 20 by record count) --")
        equip_summary = (
            df.groupby("EQUIP_ID")
            .agg(records=("SPEED", "count"), mean_speed=("SPEED", "mean"), mean_volume=("VOLUME", "mean"))
            .round(2)
            .sort_values("records", ascending=False)
            .head(20)
        )
        print(equip_summary.to_string())

    if "LINK_ID" in df.columns and "SPEED" in df.columns:
        print(f"\n-- LINK_ID summary (top 20 by record count) --")
        link_summary = (
            df.groupby("LINK_ID")
            .agg(records=("SPEED", "count"), mean_speed=("SPEED", "mean"), min_speed=("SPEED", "min"), max_speed=("SPEED", "max"))
            .round(2)
            .sort_values("records", ascending=False)
            .head(20)
        )
        print(link_summary.to_string())


def cross_dataset_comparison():
    section("5. CROSS-DATASET COMPARISON (TIQ vs SpeedGraph | Weekday vs Weekend)")

    summaries = []
    for name, meta in FILES.items():
        df = load_file(meta)
        if "DATE_TIME" in df.columns:
            parsed_dt = pd.to_datetime(df["DATE_TIME"], errors="coerce")
            date_min = str(parsed_dt.min())
            date_max = str(parsed_dt.max())
        else:
            date_min = date_max = "N/A"
        row = {
            "dataset": name,
            "source": meta["source"],
            "day_type": meta["day_type"],
            "rows": len(df),
            "columns": list(df.columns),
            "null_total": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "mean_speed": round(df["SPEED"].mean(), 2) if "SPEED" in df.columns else None,
            "min_speed": df["SPEED"].min() if "SPEED" in df.columns else None,
            "max_speed": df["SPEED"].max() if "SPEED" in df.columns else None,
            "mean_volume": round(df["VOLUME"].mean(), 2) if "VOLUME" in df.columns else None,
            "date_min": date_min,
            "date_max": date_max,
        }
        summaries.append(row)

    summary_df = pd.DataFrame(summaries).set_index("dataset")
    print(summary_df.T.to_string())


def run_eda():
    for name, meta in FILES.items():
        print(f"\n{'#' * 70}")
        print(f"  DATASET: {name}")
        print(f"  Source : {meta['source']}  |  Day Type: {meta['day_type']}")
        print(f"  File   : {os.path.basename(meta['path'])}")
        print(f"{'#' * 70}")

        df = load_file(meta)

        explore_features(name, df)
        check_data_quality(name, df)
        explore_categories(name, df)
        general_statistics(name, df)

    cross_dataset_comparison()
    print(f"\n{SEPARATOR}")
    print("  EDA COMPLETE")
    print(SEPARATOR)


# ─── Data-returning API (used by Streamlit app) ───────────────────────────────

import glob as _glob


def discover_raw_files(raw_dir: str) -> dict:
    """Scan *raw_dir* for CSV/Excel files. Returns {display_name: abs_path}."""
    found = {}
    for pattern in ("*.csv", "*.xlsx", "*.xls"):
        for p in sorted(_glob.glob(os.path.join(raw_dir, pattern))):
            fname = os.path.basename(p)
            ext = os.path.splitext(p)[1].upper().lstrip(".")
            found[f"{fname}  [{ext}]"] = p
    return found


def load_any_file(path: str) -> pd.DataFrame:
    """Load a CSV or Excel file and normalise column names."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.strip('"').str.strip("'")
    return df


def _detect_cols(df: pd.DataFrame) -> dict:
    """Heuristically detect common column roles."""
    cols = df.columns.tolist()
    return {
        "dt":     next((c for c in cols if "date" in c.lower() or "time" in c.lower()), None),
        "speed":  next((c for c in cols if "speed" in c.lower()), None),
        "volume": next((c for c in cols if "volume" in c.lower() or "vol" in c.lower()), None),
        "road":   next((c for c in cols if "road" in c.lower() or "name" in c.lower()), None),
        "id":     next((c for c in cols if "equip" in c.lower() or "link" in c.lower() or (c.lower().endswith("_id") and "equip" not in c.lower())), None),
    }


def get_feature_info(df: pd.DataFrame) -> dict:
    """Return shape, dtypes DataFrame, head and tail DataFrames."""
    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["Column", "Type"]
    dtype_df["Type"] = dtype_df["Type"].astype(str)  # ensure Arrow-serialisable
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": dtype_df,
        "head": df.head(10),
        "tail": df.tail(5),
    }


def get_quality_checks(df: pd.DataFrame) -> dict:
    """Return null summary, duplicate count, neg/zero counts, speed anomalies, date range."""
    dc = _detect_cols(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    null_df = pd.DataFrame({
        "Column": df.columns,
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().mean() * 100).round(2).values,
    })

    dup_count = int(df.duplicated().sum())

    neg_zero = pd.DataFrame([{
        "Column": c,
        "Negatives": int((df[c] < 0).sum()),
        "Zeros": int((df[c] == 0).sum()),
    } for c in numeric_cols])

    speed_anomalies = None
    if dc["speed"]:
        sc = dc["speed"]
        speed_anomalies = df[(df[sc] < 0) | (df[sc] > 200)]

    date_range = None
    if dc["dt"]:
        parsed = pd.to_datetime(df[dc["dt"]], errors="coerce")
        date_range = {
            "col": dc["dt"],
            "min": parsed.min(),
            "max": parsed.max(),
            "unparseable": int(parsed.isnull().sum()),
        }

    return {
        "null_df": null_df,
        "dup_count": dup_count,
        "dup_pct": round(dup_count / len(df) * 100, 2),
        "neg_zero": neg_zero,
        "speed_col": dc["speed"],
        "speed_anomalies": speed_anomalies,
        "date_range": date_range,
    }


def get_category_analysis(df: pd.DataFrame) -> dict:
    """Return {col: {unique, value_counts}} for each non-datetime object column."""
    dc = _detect_cols(df)
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != dc["dt"]]
    result = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, "Count"]
        vc["Count %"] = (vc["Count"] / len(df) * 100).round(2)
        result[col] = {"unique": int(df[col].nunique()), "value_counts": vc}
    return result


def get_statistics(df: pd.DataFrame) -> dict:
    """Return descriptive stats and hourly/road/id aggregations."""
    dc = _detect_cols(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    describe = df[numeric_cols].describe().round(2) if numeric_cols else None

    hourly_speed = hourly_volume = road_speed = id_summary = None

    if dc["dt"]:
        _df = df.copy()
        _df["_dt"] = pd.to_datetime(_df[dc["dt"]], errors="coerce")
        _df["_hour"] = _df["_dt"].dt.hour

        if dc["speed"]:
            hourly_speed = _df.groupby("_hour")[dc["speed"]].mean().round(2).reset_index()
            hourly_speed.columns = ["Hour", f"Mean {dc['speed']}"]

        if dc["volume"]:
            hourly_volume = _df.groupby("_hour")[dc["volume"]].mean().round(2).reset_index()
            hourly_volume.columns = ["Hour", f"Mean {dc['volume']}"]

        if dc["road"] and dc["speed"]:
            road_speed = (
                _df.groupby(dc["road"])[dc["speed"]]
                .mean().round(2).reset_index()
                .sort_values(dc["speed"], ascending=False)
                .rename(columns={dc["speed"]: f"Mean {dc['speed']}"})
            )

        if dc["id"] and dc["speed"]:
            agg = {"Records": (dc["speed"], "count"), f"Mean {dc['speed']}": (dc["speed"], "mean")}
            if dc["volume"]:
                agg[f"Mean {dc['volume']}"] = (dc["volume"], "mean")
            id_summary = (
                _df.groupby(dc["id"]).agg(**agg)
                .round(2)
                .sort_values("Records", ascending=False)
                .head(20)
                .reset_index()
            )

    return {
        "describe": describe,
        "hourly_speed": hourly_speed,
        "hourly_volume": hourly_volume,
        "road_speed": road_speed,
        "id_summary": id_summary,
        "speed_col": dc["speed"],
        "volume_col": dc["volume"],
        "road_col": dc["road"],
        "id_col": dc["id"],
    }


def get_cross_comparison(files_dict: dict) -> pd.DataFrame:
    """
    Summarise all files side by side.
    files_dict: {display_name: abs_path_str}
    """
    rows = []
    for name, path in files_dict.items():
        df = load_any_file(path)
        dc = _detect_cols(df)
        date_min = date_max = "N/A"
        if dc["dt"]:
            parsed = pd.to_datetime(df[dc["dt"]], errors="coerce")
            date_min, date_max = str(parsed.min()), str(parsed.max())
        rows.append({
            "File": os.path.basename(path),
            "Rows": len(df),
            "Columns": ", ".join(df.columns.tolist()),
            "Nulls": int(df.isnull().sum().sum()),
            "Duplicates": int(df.duplicated().sum()),
            "Mean Speed": round(df[dc["speed"]].mean(), 2) if dc["speed"] else None,
            "Min Speed": df[dc["speed"]].min() if dc["speed"] else None,
            "Max Speed": df[dc["speed"]].max() if dc["speed"] else None,
            "Mean Volume": round(df[dc["volume"]].mean(), 2) if dc["volume"] else None,
            "Date Min": date_min,
            "Date Max": date_max,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        sys.stdout = f
        run_eda()
    sys.stdout = sys.__stdout__
    print(f"EDA report saved to: {OUT_FILE}")

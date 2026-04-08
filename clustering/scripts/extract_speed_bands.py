"""
Speed band extraction, medoid identification, and per-LinkID validation.

For each cluster:
  - lower_band = P10 of cluster speeds
  - upper_band = P90 of cluster speeds
  - medoid     = actual record nearest to cluster centroid in feature space

Per-LinkID validation flags spatial outliers — LinkIDs whose mean speed
deviates more than LINKID_DEVIATION_THRESHOLD km/h from the cluster band
midpoint. These are flagged for review but do NOT change the cluster bands.

Reads:  clustering/results/assignments_<cat>_<day>.parquet  (from run_clustering.py)
Writes: data/output/traffic_hourly_with_bands.csv           (full dataset + cluster_id, lower_band, upper_band)
        clustering/results/cluster_summary_<cat>_<day>.csv  (per-cluster diagnostics)
        clustering/results/cluster_medoids.csv              (one medoid row per cluster)
        clustering/results/linkid_validation.csv            (per-LinkID deviation report)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("clustering/results")
OUTPUT_DIR = Path("data/output")

FEATURE_COLS = ["sin_h", "cos_h", "day_bin", "speed_norm", "volume_norm"]
LINKID_DEVIATION_THRESHOLD = 15.0  # km/h


def extract_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P10/P90 speed bands per cluster_id."""
    bands = (
        df.groupby("cluster_id")["speed"]
        .agg(
            lower_band=lambda x: round(x.quantile(0.10), 2),
            upper_band=lambda x: round(x.quantile(0.90), 2),
            cluster_speed_mean=lambda x: round(x.mean(), 2),
            cluster_speed_std=lambda x: round(x.std(), 2),
            cluster_size="count",
        )
        .reset_index()
    )
    return bands


def extract_medoids(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster, find the actual record closest to the cluster centroid
    in normalised feature space: i* = argmin ||x_i - mu_c||_2
    """
    medoids = []
    for cluster_id, grp in df.groupby("cluster_id"):
        X = grp[FEATURE_COLS].values
        centroid = X.mean(axis=0)
        nearest_idx = np.argmin(np.linalg.norm(X - centroid, axis=1))
        row = grp.iloc[nearest_idx].copy()
        row["is_medoid"] = True
        medoids.append(row)
    return pd.DataFrame(medoids)


def validate_linkids(df: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """
    Flag LinkIDs whose mean speed deviates more than LINKID_DEVIATION_THRESHOLD
    from the cluster band midpoint.
    """
    merged = df.merge(
        bands[["cluster_id", "lower_band", "upper_band"]],
        on="cluster_id",
        how="left",
    )
    merged["band_midpoint"] = (merged["lower_band"] + merged["upper_band"]) / 2

    link_stats = (
        merged.groupby(["cluster_id", "LinkID"])
        .agg(
            link_speed_mean=("speed", "mean"),
            link_speed_std=("speed", "std"),
            link_count=("speed", "count"),
            band_midpoint=("band_midpoint", "first"),
        )
        .reset_index()
    )
    link_stats["deviation"] = (
        (link_stats["link_speed_mean"] - link_stats["band_midpoint"]).abs().round(2)
    )
    link_stats["flagged"] = link_stats["deviation"] > LINKID_DEVIATION_THRESHOLD
    return link_stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assign_files = sorted(RESULTS_DIR.glob("assignments_*.parquet"))
    if not assign_files:
        raise FileNotFoundError(
            f"No assignment files found in {RESULTS_DIR}. Run run_clustering.py first."
        )

    all_assignments = []
    all_summaries = []
    all_validations = []
    all_medoids = []

    for path in assign_files:
        tag = path.stem.replace("assignments_", "")  # e.g. "1_weekday"
        df = pd.read_parquet(path)

        # Speed bands
        bands = extract_bands(df)

        # Medoids
        medoids = extract_medoids(df)
        medoids["subset"] = tag
        all_medoids.append(medoids)

        # Cluster summary
        summary = bands.copy()
        summary["subset"] = tag
        all_summaries.append(summary)
        summary_path = RESULTS_DIR / f"cluster_summary_{tag}.csv"
        bands.to_csv(summary_path, index=False)

        # Per-LinkID validation (must run before bands are merged into df)
        val = validate_linkids(df, bands)

        # Map bands back onto every row
        df = df.merge(
            bands[["cluster_id", "lower_band", "upper_band"]],
            on="cluster_id",
            how="left",
        )
        all_assignments.append(df)
        val["subset"] = tag
        all_validations.append(val)

        n_flagged = val["flagged"].sum()
        print(
            f"{tag}: {len(bands)} clusters | {n_flagged} flagged LinkID-cluster pair(s)"
        )

    # Combine all subsets
    result = pd.concat(all_assignments, ignore_index=True)

    # Drop internal feature and intermediate columns before saving
    drop_cols = [c for c in FEATURE_COLS + ["cluster_raw"] if c in result.columns]
    result = result.drop(columns=drop_cols)

    out_path = OUTPUT_DIR / "traffic_hourly_with_bands.csv"
    result.to_csv(out_path, index=False)
    print(f"\nOutput saved → {out_path}  ({len(result):,} rows)")

    # LinkID validation report
    val_all = pd.concat(all_validations, ignore_index=True)
    val_path = RESULTS_DIR / "linkid_validation.csv"
    val_all.to_csv(val_path, index=False)
    total_flagged = val_all["flagged"].sum()
    print(f"LinkID validation: {total_flagged} flagged pairs → {val_path}")

    # Medoid report
    med_all = pd.concat(all_medoids, ignore_index=True)
    med_path = RESULTS_DIR / "cluster_medoids.csv"
    med_all.to_csv(med_path, index=False)
    print(f"Medoids saved → {med_path}")

    print("\nBand extraction complete.")


if __name__ == "__main__":
    main()

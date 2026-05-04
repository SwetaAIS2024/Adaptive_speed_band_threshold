"""
KMeans clustering per (RoadCategory x day_type) subset.

For each subset:
  1. Sweep K=2..10, score silhouette + Davies-Bouldin
  2. Select K with highest silhouette score
  3. Merge clusters below MIN_CLUSTER_SIZE into nearest centroid
  4. Flag clusters with speed spread exceeding quality thresholds
  5. Assign composite cluster_id e.g. "EXP_WD_03"

Reads:  clustering/features/feature_metadata.json
        clustering/features/features_<cat>_<day>.parquet
Writes: clustering/results/elbow_silhouette_<cat>_<day>.csv   (K-sweep diagnostics)
        clustering/results/cluster_centers_<cat>_<day>.csv    (final centroids)
        clustering/results/assignments_<cat>_<day>.parquet    (rows + cluster_id)
"""

import json
import sys
import warnings
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from clustering.pipeline import (
    FEATURE_COLS, VALID_CATEGORIES, CATEGORY_PREFIX, DAY_PREFIX,
    K_RANGE, MIN_CLUSTER_SIZE, MIN_SILHOUETTE,
    sweep_k, pick_optimal_k, run_kmeans,
    merge_small_clusters, flag_quality, build_cluster_id,
)

warnings.filterwarnings("ignore")

FEATURES_DIR = Path("clustering/features")
RESULTS_DIR = Path("clustering/results")


def main():
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run feature_engineering.py first."
        )

    with open(meta_path) as f:
        metadata = json.load(f)

    for key, meta in metadata.items():
        cat = meta["road_category"]
        day = meta["day_type"]
        cat_prefix = meta["cat_prefix"]
        day_prefix = meta["day_prefix"]

        feat_file = Path(meta["feature_file"])
        if not feat_file.exists():
            print(f"[SKIP] Feature file not found: {feat_file}")
            continue

        df = pd.read_parquet(feat_file)
        active_cols = meta.get("active_cols", FEATURE_COLS)
        X = df[active_cols].values
        print(f"\nCat={cat} ({cat_prefix}) {day}: {len(df):,} rows, {len(active_cols)} features")

        # K sweep
        diag = sweep_k(X)
        diag_path = RESULTS_DIR / f"elbow_silhouette_{cat}_{day.lower()}.csv"
        diag.to_csv(diag_path, index=False)

        k_opt = pick_optimal_k(diag)
        best_sil = diag.loc[diag["k"] == k_opt, "silhouette"].values[0]
        print(f"  Optimal K={k_opt}  silhouette={best_sil}")

        if best_sil < MIN_SILHOUETTE:
            print(f"  WARNING: silhouette {best_sil} < threshold {MIN_SILHOUETTE} — clusters may not be well-separated")

        # Final clustering at optimal K
        labels, centers = run_kmeans(X, k_opt)
        df["cluster_raw"] = labels

        # Merge clusters that are too small
        merged_labels, merged_centers = merge_small_clusters(df["cluster_raw"].values, centers, MIN_CLUSTER_SIZE)
        df["cluster_raw"] = merged_labels
        k_final = len(set(merged_labels))
        if k_final != k_opt:
            print(f"  After merging small clusters: K reduced {k_opt} → {k_final}")

        # Quality flags
        flags = flag_quality(df)
        flagged = flags[flags["flag_std"] | flags["flag_iqr"]]
        if len(flagged):
            flagged_ids = flagged["cluster_raw"].tolist()
            print(f"  WARNING: {len(flagged)} cluster(s) with wide speed spread (flag_std/flag_iqr): {flagged_ids}")

        # Composite cluster IDs
        df["cluster_id"] = df["cluster_raw"].apply(
            lambda idx: build_cluster_id(cat_prefix, day_prefix, idx)
        )

        # Save cluster centres
        centers_df = pd.DataFrame(merged_centers, columns=active_cols)
        centers_df.insert(0, "cluster_raw", range(len(merged_centers)))
        centers_df["cluster_id"] = centers_df["cluster_raw"].apply(
            lambda idx: build_cluster_id(cat_prefix, day_prefix, idx)
        )
        centers_path = RESULTS_DIR / f"cluster_centers_{cat}_{day.lower()}.csv"
        centers_df.to_csv(centers_path, index=False)

        # Save assignments (original rows + engineered features + cluster_id)
        assign_path = RESULTS_DIR / f"assignments_{cat}_{day.lower()}.parquet"
        df.to_parquet(assign_path, index=False)

        print(f"  Cluster IDs: {sorted(df['cluster_id'].unique())}")
        print(f"  Saved → {assign_path.name}")

    print("\nClustering complete.")


if __name__ == "__main__":
    main()

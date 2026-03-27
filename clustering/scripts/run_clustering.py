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
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore")

FEATURES_DIR = Path("clustering/features")
RESULTS_DIR = Path("clustering/results")

K_RANGE = range(2, 11)
MIN_CLUSTER_SIZE = 30
MAX_SPEED_STD = 15.0
MAX_SPEED_IQR = 20.0
MIN_SILHOUETTE = 0.3

FEATURE_COLS = ["sin_h", "cos_h", "day_bin", "speed_norm", "volume_norm"]
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10


def sweep_k(X: np.ndarray) -> pd.DataFrame:
    """Run KMeans for every K in K_RANGE; return diagnostic dataframe."""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
        labels = km.fit_predict(X)
        sample_size = min(10_000, len(X))
        sil = silhouette_score(X, labels, sample_size=sample_size, random_state=KMEANS_RANDOM_STATE)
        db = davies_bouldin_score(X, labels)
        rows.append(
            {
                "k": k,
                "inertia": round(km.inertia_, 2),
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
            }
        )
    return pd.DataFrame(rows)


def pick_optimal_k(diag: pd.DataFrame) -> int:
    """Select K with the highest silhouette score."""
    return int(diag.loc[diag["silhouette"].idxmax(), "k"])


def merge_small_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
    min_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively merge clusters below min_size into their nearest neighbour
    (by centroid Euclidean distance). Re-indexes labels after each merge so
    cluster indices are always contiguous starting from 0.
    """
    labels = labels.copy()
    centers = centers.copy()

    while True:
        sizes = pd.Series(labels).value_counts()
        small_ids = sizes[sizes < min_size].index.tolist()
        if not small_ids:
            break

        target = small_ids[0]
        target_center = centers[target]
        dists = np.linalg.norm(centers - target_center, axis=1)
        dists[target] = np.inf
        nearest = int(np.argmin(dists))

        labels[labels == target] = nearest

        # Re-index to keep labels contiguous
        unique = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique)}
        labels = np.array([remap[lbl] for lbl in labels])
        centers = np.array([centers[old] for old in unique])

    return labels, centers


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-cluster quality flags (speed std and IQR thresholds)."""
    stats = (
        df.groupby("cluster_raw")["speed"]
        .agg(
            cluster_size="count",
            speed_std="std",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    stats["speed_iqr"] = stats["p75"] - stats["p25"]
    stats["flag_size"] = stats["cluster_size"] < MIN_CLUSTER_SIZE
    stats["flag_std"] = stats["speed_std"] > MAX_SPEED_STD
    stats["flag_iqr"] = stats["speed_iqr"] > MAX_SPEED_IQR
    return stats


def build_cluster_id(cat_prefix: str, day_prefix: str, local_index: int) -> str:
    return f"{cat_prefix}_{day_prefix}_{local_index:02d}"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = FEATURES_DIR / "feature_metadata.json"
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
        X = df[FEATURE_COLS].values
        print(f"\nCat={cat} ({cat_prefix}) {day}: {len(df):,} rows")

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
        km = KMeans(n_clusters=k_opt, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
        df["cluster_raw"] = km.fit_predict(X)
        centers = km.cluster_centers_.copy()

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
        centers_df = pd.DataFrame(merged_centers, columns=FEATURE_COLS)
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

"""
Shared pipeline functions for feature engineering, clustering, and band extraction.

Imported by both the CLI scripts (feature_engineering.py, run_clustering.py,
extract_speed_bands.py) and the Streamlit app. Do NOT put any I/O or path
logic here — only pure, reusable computation functions.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler

# ─── Shared constants ────────────────────────────────────────────────────────

VALID_CATEGORIES = {1, 2, 3, 4, 5, 6}
DAY_TYPES = ["Weekday", "Weekend"]
CATEGORY_PREFIX = {1: "EXP", 2: "MJR", 3: "ART", 4: "MIN", 5: "LOC", 6: "ACC"}
DAY_PREFIX = {"Weekday": "WD", "Weekend": "WE"}
REQUIRED_COLS = {"LinkID", "RoadCategory", "hour", "day_type", "speed", "volume"}
FEATURE_COLS = ["sin_h", "cos_h", "day_bin", "speed_norm", "volume_norm"]

# Optional future-dataset columns and their ordinal code→int maps.
# If these columns are present in the subset they are encoded and added as
# additional clustering features (normalised to [0, 1]).
OPTIONAL_FEATURE_MAPS: dict[str, dict[str, int]] = {
    "expressway":     {"E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4, "E6": 5, "E7": 6, "E8": 7},
    "road_direction": {"D1": 0, "D2": 1},
    "source":         {"S1": 0, "S2": 1, "S3": 2},
}

K_RANGE = range(2, 11)
MIN_CLUSTER_SIZE = 30
MAX_SPEED_STD = 15.0
MAX_SPEED_IQR = 20.0
MIN_SILHOUETTE = 0.3
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10

LINKID_DEVIATION_THRESHOLD = 15.0  # km/h


# ─── Feature engineering ─────────────────────────────────────────────────────

def engineer_subset(subset: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler, list[str]]:
    """
    Return (feature-augmented df, fitted MinMaxScaler, active_feature_cols).

    Always-on transformations:
      - sin_h, cos_h  : cyclic encoding of hour (2π h / 24)
      - day_bin       : Weekday = 0, Weekend = 1
      - speed_norm    : MinMax scaled within this subset
      - volume_norm   : MinMax scaled within this subset

    Optional transformations (applied when the column is present in *subset*):
      - expressway_enc    : ordinal encoding of expressway code (E1-E8), normalised to [0, 1]
      - road_direction_enc: ordinal encoding of road_direction code (D1/D2), normalised to [0, 1]
      - source_enc        : ordinal encoding of source code (S1-S3), normalised to [0, 1]
    """
    df = subset.copy()
    df["sin_h"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_h"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_bin"] = (df["day_type"] == "Weekend").astype(int)
    scaler = MinMaxScaler()
    df[["speed_norm", "volume_norm"]] = scaler.fit_transform(df[["speed", "volume"]])

    active_cols = list(FEATURE_COLS)

    for col, code_map in OPTIONAL_FEATURE_MAPS.items():
        if col in df.columns and df[col].notna().any():
            enc_col = f"{col}_enc"
            max_val = max(code_map.values())
            df[enc_col] = df[col].map(code_map).fillna(0).clip(0, max_val) / max_val
            active_cols.append(enc_col)

    return df, scaler, active_cols


# ─── Clustering ──────────────────────────────────────────────────────────────

def sweep_k(X: np.ndarray) -> pd.DataFrame:
    """
    Run KMeans for every K in K_RANGE.
    Returns a DataFrame with columns: k, inertia, silhouette, davies_bouldin.
    """
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
        labels = km.fit_predict(X)
        sample_size = min(10_000, len(X))
        sil = silhouette_score(X, labels, sample_size=sample_size, random_state=KMEANS_RANDOM_STATE)
        db = davies_bouldin_score(X, labels)
        rows.append({
            "k": k,
            "inertia": round(km.inertia_, 2),
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
        })
    return pd.DataFrame(rows)


def pick_optimal_k(diag: pd.DataFrame) -> int:
    """Select K with the highest silhouette score."""
    return int(diag.loc[diag["silhouette"].idxmax(), "k"])


def run_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans at a fixed K. Returns (labels, cluster_centers)."""
    km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_.copy()


def merge_small_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
    min_size: int = MIN_CLUSTER_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iteratively merge clusters below *min_size* into the nearest neighbour
    by Euclidean centroid distance. Re-indexes labels to be contiguous after
    each merge.
    """
    labels = labels.copy()
    centers = centers.copy()

    while True:
        sizes = pd.Series(labels).value_counts()
        small_ids = sizes[sizes < min_size].index.tolist()
        if not small_ids:
            break

        target = small_ids[0]
        dists = np.linalg.norm(centers - centers[target], axis=1)
        dists[target] = np.inf
        nearest = int(np.argmin(dists))

        labels[labels == target] = nearest

        unique = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique)}
        labels = np.array([remap[lbl] for lbl in labels])
        centers = np.array([centers[old] for old in unique])

    return labels, centers


def flag_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-cluster quality flags.
    Expects *df* to have columns: cluster_raw, speed.
    """
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
    """e.g. build_cluster_id("EXP", "WD", 3) → "EXP_WD_03" """
    return f"{cat_prefix}_{day_prefix}_{local_index:02d}"


# ─── Band extraction ─────────────────────────────────────────────────────────

def extract_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P10 / P90 speed bands per cluster_id.
    Returns DataFrame: cluster_id, lower_band, upper_band,
                       cluster_speed_mean, cluster_speed_std, cluster_size.
    """
    return (
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


def extract_medoids(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """
    For each cluster find the actual record with minimum Euclidean distance
    to the cluster centroid in normalised feature space:
        i* = argmin ||x_i − μ_c||₂

    *feature_cols* defaults to FEATURE_COLS; pass the active_feature_cols returned
    by engineer_subset when optional features were encoded.
    """
    _cols = feature_cols if feature_cols is not None else FEATURE_COLS
    medoids = []
    for cluster_id, grp in df.groupby("cluster_id"):
        X = grp[_cols].values
        centroid = X.mean(axis=0)
        nearest_idx = np.argmin(np.linalg.norm(X - centroid, axis=1))
        row = grp.iloc[nearest_idx].copy()
        row["is_medoid"] = True
        medoids.append(row)
    return pd.DataFrame(medoids)


def validate_linkids(df: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """
    Flag LinkIDs whose mean speed deviates more than LINKID_DEVIATION_THRESHOLD
    km/h from the cluster band midpoint.

    *df* must already have cluster_id column.
    *bands* must have: cluster_id, lower_band, upper_band.
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

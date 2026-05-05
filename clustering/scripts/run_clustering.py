"""
KMeans clustering on feature-engineered parquet files.

For every parquet in clustering/features/ two independent clusterings are run:
  • Speed clustering  — features: hour_sin, hour_cos, SPEED, day_type
  • Volume clustering — features: hour_sin, hour_cos, VOLUME, day_type
                        (only when a VOLUME column is present)

Per clustering run:
  1. K-sweep K=2..8 (silhouette + Davies-Bouldin); sample up to K_SWEEP_SAMPLE rows
  2. Select K with highest silhouette score
  3. Fit final K-Means on all rows at chosen K
  4. Merge clusters below MIN_CLUSTER_FRAC of total rows into nearest centroid

Outputs (under clustering/results/):
  elbow_silhouette_<stem>_speed.csv
  elbow_silhouette_<stem>_volume.csv       (if volume present)
  cluster_centers_<stem>_speed.csv
  cluster_centers_<stem>_volume.csv        (if volume present)
  cluster_assignments_<stem>_speed.parquet
  cluster_assignments_<stem>_volume.parquet (if volume present)
  clustering_summary.csv

Public API (imported by the Streamlit app):
  discover_feature_files()  → {stem: Path}
  run_one(df, feature_cols, stem, clustering_type, k_override=None)
      → ClusterResult(diag_df, centers_df, assigned_df, k_opt, best_sil)
  load_results(stem, clustering_type) → ClusterResult | None
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Allow running as a script from the repo root (same pattern as extract_speed_bands.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from clustering.scripts.pipeline import (
    K_RANGE,
    MIN_SILHOUETTE,
    merge_small_clusters,
    pick_optimal_k,
    run_kmeans,
    sweep_k,
)

# ── Paths (all relative to repo root, resolved at import time) ────────────────
_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = _REPO_ROOT / "clustering" / "features"
RESULTS_DIR  = _REPO_ROOT / "clustering" / "results"

# ── Clustering constants ──────────────────────────────────────────────────────
K_SWEEP_SAMPLE   = 50_000            # max rows for K-sweep
MIN_CLUSTER_FRAC = 0.02              # merge clusters < 2 % of total

# Fixed feature sets — column names match what feature_engineering.py produces
_ALWAYS_FEATURES  = ["hour_sin", "hour_cos", "day_type"]
_SPEED_COL        = "SPEED"
_VOLUME_COL       = "VOLUME"

SKIP_FILES = {"preprocessing_summary.csv", "eda_report.txt"}


# ── Public data container ─────────────────────────────────────────────────────

@dataclass
class ClusterResult:
    diag_df:     pd.DataFrame   # K-sweep diagnostics (k, inertia, silhouette, davies_bouldin)
    centers_df:  pd.DataFrame   # cluster centroids (feature cols + cluster_label)
    assigned_df: pd.DataFrame   # all rows with cluster_label appended
    k_opt:       int
    best_sil:    float


# ── File discovery ────────────────────────────────────────────────────────────

def discover_feature_files() -> dict[str, Path]:
    """Return {stem: absolute_path} for every parquet in FEATURES_DIR."""
    return {
        f.stem: f
        for f in sorted(FEATURES_DIR.glob("*.parquet"))
    }


# ── Public: run one clustering ────────────────────────────────────────────────

def run_one(
    df: pd.DataFrame,
    feature_cols: list[str],
    stem: str,
    clustering_type: str,        # "speed" or "volume"
    k_override: int | None = None,
    save: bool = True,
) -> ClusterResult:
    """
    Cluster `df` on `feature_cols`, optionally save results.

    Uses sweep_k, pick_optimal_k, run_kmeans, merge_small_clusters
    from clustering.scripts.pipeline — no logic is duplicated here.

    Parameters
    ----------
    df              : feature-engineered DataFrame
    feature_cols    : columns to use as clustering features
    stem            : filename stem (used for output filenames)
    clustering_type : "speed" or "volume"
    k_override      : if given, skip K-sweep and use this K directly
    save            : write CSV/parquet results to RESULTS_DIR

    Returns
    -------
    ClusterResult
    """
    X_all = df[feature_cols].astype(float).values

    # K-sweep on a sample
    if k_override is None:
        X_sweep = X_all
        if len(X_all) > K_SWEEP_SAMPLE:
            rng = np.random.default_rng(42)
            X_sweep = X_all[rng.choice(len(X_all), K_SWEEP_SAMPLE, replace=False)]
        diag  = sweep_k(X_sweep)
        k_opt = pick_optimal_k(diag)
    else:
        diag  = pd.DataFrame(columns=["k", "inertia", "silhouette", "davies_bouldin"])
        k_opt = k_override

    best_sil = (
        float(diag.loc[diag["k"] == k_opt, "silhouette"].values[0])
        if len(diag) and k_opt in diag["k"].values
        else float("nan")
    )
    if not np.isnan(best_sil) and best_sil < MIN_SILHOUETTE:
        print(f"  WARNING [{stem}/{clustering_type}]: silhouette={best_sil:.3f} < {MIN_SILHOUETTE}")

    # Final fit + merge small clusters (fraction-based threshold)
    labels, centers = run_kmeans(X_all, k_opt)
    min_size = max(1, int(MIN_CLUSTER_FRAC * len(X_all)))
    labels, centers = merge_small_clusters(labels, centers, min_size=min_size)
    k_final = len(np.unique(labels))
    if k_final != k_opt:
        print(f"  [{stem}/{clustering_type}] K merged {k_opt} → {k_final}")

    # Build output DataFrames
    assigned = df.copy()
    assigned["cluster_label"] = labels

    centers_df = pd.DataFrame(centers, columns=feature_cols)
    centers_df.insert(0, "cluster_label", range(len(centers)))

    result = ClusterResult(
        diag_df=diag,
        centers_df=centers_df,
        assigned_df=assigned,
        k_opt=k_final,
        best_sil=best_sil,
    )

    if save:
        _save_result(result, stem, clustering_type)

    return result


def _save_result(result: ClusterResult, stem: str, clustering_type: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{stem}_{clustering_type}"

    if not result.diag_df.empty:
        result.diag_df.to_csv(RESULTS_DIR / f"elbow_silhouette_{tag}.csv", index=False)

    result.centers_df.to_csv(RESULTS_DIR / f"cluster_centers_{tag}.csv", index=False)
    result.assigned_df.to_parquet(RESULTS_DIR / f"cluster_assignments_{tag}.parquet", index=False)


def load_results(stem: str, clustering_type: str) -> "ClusterResult | None":
    """Load previously saved results from disk; return None if not found."""
    tag = f"{stem}_{clustering_type}"
    centers_path = RESULTS_DIR / f"cluster_centers_{tag}.csv"
    assign_path  = RESULTS_DIR / f"cluster_assignments_{tag}.parquet"
    diag_path    = RESULTS_DIR / f"elbow_silhouette_{tag}.csv"

    if not (centers_path.exists() and assign_path.exists()):
        return None

    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    centers  = pd.read_csv(centers_path)
    assigned = pd.read_parquet(assign_path)
    k_opt    = int(assigned["cluster_label"].nunique())
    best_sil = (
        float(diag["silhouette"].max()) if not diag.empty else float("nan")
    )
    return ClusterResult(diag_df=diag, centers_df=centers, assigned_df=assigned,
                         k_opt=k_opt, best_sil=best_sil)


# ── Feature-column detection ──────────────────────────────────────────────────

def feature_cols_for(df: pd.DataFrame, clustering_type: str) -> list[str]:
    """
    Return the feature columns to use for a given clustering type.
    Columns must exist in df (case-insensitive lookup).
    """
    col_map = {c.lower(): c for c in df.columns}
    base = [col_map[c.lower()] for c in _ALWAYS_FEATURES if c.lower() in col_map]

    if clustering_type == "speed":
        target = _SPEED_COL
    else:
        target = _VOLUME_COL

    if target.lower() in col_map:
        base.append(col_map[target.lower()])
    else:
        return []   # signal: this clustering type not possible for this file

    return base


# ── Script entry point ────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_files = discover_feature_files()
    if not feature_files:
        print(f"No parquet files found in {FEATURES_DIR}. Run feature_engineering.py first.")
        return

    print(f"\nFound {len(feature_files)} feature file(s)")
    summary_rows = []

    for stem, path in feature_files.items():
        print(f"\n{'=' * 65}")
        print(f"  File: {path.name}")
        df = pd.read_parquet(path)

        for ctype in ("speed", "volume"):
            fcols = feature_cols_for(df, ctype)
            if not fcols:
                print(f"  [SKIP] {ctype} — required column not found")
                continue

            print(f"\n  ── {ctype.upper()} clustering ({fcols}) ──")
            result = run_one(df, fcols, stem, ctype, save=True)

            print(f"     K={result.k_opt}  silhouette={result.best_sil:.3f}")
            sizes = result.assigned_df["cluster_label"].value_counts().sort_index().to_dict()
            print(f"     Cluster sizes: {sizes}")

            summary_rows.append({
                "file":            path.name,
                "clustering_type": ctype,
                "feature_cols":    ", ".join(fcols),
                "k":               result.k_opt,
                "silhouette":      round(result.best_sil, 4) if not np.isnan(result.best_sil) else "",
                "rows":            len(result.assigned_df),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "clustering_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 65}")
    print("CLUSTERING SUMMARY")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved → {summary_path}")
    print("Clustering complete.")


if __name__ == "__main__":
    main()

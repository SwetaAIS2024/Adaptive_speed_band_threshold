"""
Shared helpers for Speed Threshold Visualisation and Volume Threshold
Visualisation tabs.

Both tabs derive P10/P90 thresholds from K-Means cluster assignments
(not from raw global percentiles), so every cluster gets its own adaptive
threshold band that reflects the traffic behaviour of that road group.
"""

import numpy as np
import pandas as pd
import streamlit as st

from .context import (
    AppContext,
    enrich_with_clusters,
    enriched_result_path,
    load_enriched_result,
    persist_enriched_result,
)

# ── Colour palette (one colour per cluster, cycles if K > 10) ─────────────────
_CLUSTER_PALETTE = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
    "#00ACC1", "#F4511E", "#6D4C41", "#3949AB", "#00897B",
]


def cluster_color(i: int) -> str:
    """Return a distinct hex colour for cluster index *i*."""
    return _CLUSTER_PALETTE[int(i) % len(_CLUSTER_PALETTE)]


# ── Cluster result discovery ──────────────────────────────────────────────────

def discover_cluster_results(ctx: AppContext, ctype: str) -> dict:
    """
    Return {display_stem: cache_key} for all available cluster results of
    the given type (``"speed"`` or ``"volume"``).

     Sources checked in order:
     1. ``st.session_state.cluster_results`` — run in the current session
     2. Saved files on disk (``clustering/results/``) — discovered by filename
         only and loaded lazily when the user selects one.
    """
    found: dict[str, tuple] = {}

    # ── 1. Session state ──────────────────────────────────────────────────────
    for key in st.session_state.get("cluster_results", {}):
        if key[1] == ctype:
            found[key[0]] = key   # stem → cache_key

    # ── 2. Disk (previous sessions) ───────────────────────────────────────────
    results_dir = ctx.data_root.parent / "clustering" / "results"
    suffix = f"_{ctype}.parquet"
    for path in sorted(results_dir.glob(f"cluster_assignments_*{suffix}")):
        stem = path.stem.removeprefix("cluster_assignments_").removesuffix(f"_{ctype}")
        if stem not in found:
            found[stem] = (stem, ctype, None)

    return found


def get_cluster_result(ctx: AppContext, cache_key: tuple):
    """Return a cluster result from session state or load it from disk lazily."""
    result = st.session_state.get("cluster_results", {}).get(cache_key)
    if result is not None:
        return result

    if ctx.cl_mod is None:
        return None

    stem, ctype, _ = cache_key
    result = ctx.cl_mod.load_results(stem, ctype)
    if result is not None:
        st.session_state.setdefault("cluster_results", {})[cache_key] = result
    return result


def get_enriched_df(ctx: AppContext, cache_key: tuple) -> pd.DataFrame | None:
    """
    Return the enriched DataFrame (preprocessed CSV columns + ``cluster_label``)
    for the given cache key.  Uses session state as a cache; builds it from
    ``enrich_with_clusters`` on first call.
    """
    # ── Session cache hit ─────────────────────────────────────────────────────
    enriched = st.session_state.get("cluster_enriched", {}).get(cache_key)
    if enriched is not None:
        return enriched

    stem, ctype, _ = cache_key
    persisted_path = enriched_result_path(stem, ctype, ctx.data_root)
    if persisted_path.exists():
        enriched = load_enriched_result(str(persisted_path))
        if enriched is not None:
            st.session_state.setdefault("cluster_enriched", {})[cache_key] = enriched
            return enriched

    # ── Build from result ─────────────────────────────────────────────────────
    result = get_cluster_result(ctx, cache_key)
    if result is None:
        return None

    enriched = enrich_with_clusters(stem, result.assigned_df, ctx.data_root)
    if enriched is not None:
        st.session_state.setdefault("cluster_enriched", {})[cache_key] = enriched
        persist_enriched_result(enriched, stem, ctype, ctx.data_root)

    return enriched


# ── Band computation ──────────────────────────────────────────────────────────

def compute_cluster_bands(
    df_filt: pd.DataFrame,
    value_col: str,
    hour_col: str,
    cluster_col: str = "cluster_label",
    include_hourly: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute P10/P90 from cluster assignments (not from raw percentiles).

    Returns
    -------
    overall_bands : DataFrame
        One row per cluster — P10/P90 across **all hours** in that cluster.
        Columns: cluster_label, p10, p90, mean, size.
    hourly_bands : DataFrame
        Per cluster × per hour — used for the time-varying threshold lines.
        Columns: cluster_label, hour, p10, p90, mean, count.
    """
    overall = (
        df_filt.groupby(cluster_col)[value_col]
        .agg(
            p10=lambda x: round(x.quantile(0.10), 2),
            p90=lambda x: round(x.quantile(0.90), 2),
            mean=lambda x: round(x.mean(), 2),
            size="count",
        )
        .reset_index()
        .rename(columns={cluster_col: "cluster_label"})
        .sort_values("cluster_label")
    )

    if include_hourly:
        hourly = (
            df_filt.groupby([cluster_col, hour_col])[value_col]
            .agg(
                p10=lambda x: round(x.quantile(0.10), 2),
                p90=lambda x: round(x.quantile(0.90), 2),
                mean=lambda x: round(x.mean(), 2),
                count="count",
            )
            .reset_index()
            .rename(columns={cluster_col: "cluster_label"})
            .sort_values(["cluster_label", hour_col])
        )
    else:
        hourly = pd.DataFrame(
            columns=["cluster_label", hour_col, "p10", "p90", "mean", "count"]
        )

    return overall, hourly


def compute_medoids(
    df_filt: pd.DataFrame,
    value_col: str,
    hour_col: str,
    cluster_col: str = "cluster_label",
) -> pd.DataFrame:
    """
    Find the actual record in each cluster that is closest to the cluster
    centroid in normalised (hour, value) 2D space.  Returns one row per
    cluster with all original columns preserved.
    """
    medoids = []
    for cid, grp in df_filt.groupby(cluster_col):
        if grp.empty:
            continue
        c_hour  = grp[hour_col].mean()
        c_val   = grp[value_col].mean()

        # Normalise each axis so hour and value contribute equally
        h_range = max(float(grp[hour_col].max() - grp[hour_col].min()), 1.0)
        v_range = max(float(grp[value_col].max() - grp[value_col].min()), 1.0)

        dist = np.sqrt(
            ((grp[hour_col] - c_hour) / h_range) ** 2
            + ((grp[value_col] - c_val) / v_range) ** 2
        )
        medoid_row = grp.iloc[int(dist.argmin())].copy()
        medoid_row[cluster_col] = cid
        medoids.append(medoid_row)

    return pd.DataFrame(medoids) if medoids else pd.DataFrame()

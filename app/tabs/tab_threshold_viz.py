"""
Tab: Speed Threshold Visualisation

Cluster-based adaptive thresholds: P10/P90 are derived from K-Means cluster
assignments, not from global raw percentiles.  Each cluster gets its own
threshold band, so roads with similar behaviour profiles share a threshold.

Traces per cluster (all individually toggleable via the Plotly legend):
  • Actual speed observations  — scatter, colour-coded by cluster
  • P10 threshold line         — per cluster × per hour (solid)
  • P90 upper band line        — per cluster × per hour (dashed)
  • Hourly mean line           — per cluster × per hour (dash-dot, optional)
  • Medoid marker              — star at the representative observation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .context import AppContext, get_pipeline_state
from ._threshold_shared import (
    cluster_color,
    compute_cluster_bands,
    compute_medoids,
    discover_cluster_results,
    get_cluster_result,
    get_enriched_df,
)


# ── Column detection ──────────────────────────────────────────────────────────

def _detect_cols(df: pd.DataFrame, ctx: AppContext) -> dict:
    if ctx.eda_mod:
        dc = ctx.eda_mod._detect_cols(df)
    else:
        dc = {}
    col_lower = {c.lower(): c for c in df.columns}
    return {
        "speed":     dc.get("speed")  or col_lower.get("speed"),
        "id":        dc.get("id")     or col_lower.get("link_id") or col_lower.get("equip_id"),
        "day_type":  col_lower.get("day_type"),
        "hour":      col_lower.get("hour"),
        "road_name": col_lower.get("road_name"),
    }


# ── Main render ───────────────────────────────────────────────────────────────

def render(ctx: AppContext) -> None:
    st.header("Speed Threshold Visualisation")
    st.caption(
        "Speed observations colour-coded by K-Means cluster.  "
        "P10 (solid) and P90 (dashed) threshold lines are derived **per cluster** — "
        "each cluster's own speed distribution determines its threshold, "
        "so thresholds adapt to the road's behaviour profile."
    )

    ps = get_pipeline_state(ctx)
    if not ps["clustering_done"]:
        st.warning(
            "⚠️ **Step 3 not complete** — no clustering results found.  "
            "Run **Clustering** (Step 3) first, then return here."
        )
        return

    # ── Cluster result selector ───────────────────────────────────────────────
    avail = discover_cluster_results(ctx, "speed")
    if not avail:
        st.info(
            "No speed clustering results found on disk or in session state.  "
            "Go to the **Clustering** tab and run a **speed** clustering first."
        )
        return

    sel_stem  = st.selectbox(
        "Select speed clustering result", list(avail.keys()), key="thresh_sel_stem"
    )
    cache_key = avail[sel_stem]

    with st.spinner("Loading enriched data …"):
        df_raw = get_enriched_df(ctx, cache_key)

    if df_raw is None:
        st.error(
            "Could not load enriched data — the preprocessed CSV for this "
            "clustering result was not found.  Re-run clustering and try again."
        )
        return

    result = get_cluster_result(ctx, cache_key)
    if result:
        st.success(
            f"K = {result.k_opt}  |  "
            f"Silhouette = {result.best_sil:.3f}" if not np.isnan(result.best_sil)
            else f"K = {result.k_opt}"
        )

    st.markdown("---")

    # ── Trace toggles ─────────────────────────────────────────────────────────
    st.markdown("**Show / hide traces**")
    t1, t2, t3, t4, t5 = st.columns(5)
    show_points  = t1.checkbox("Scatter points",     value=True,  key="thresh_t_pts")
    show_p10     = t2.checkbox("P10 threshold",      value=True,  key="thresh_t_p10")
    show_p90     = t3.checkbox("P90 upper band",     value=True,  key="thresh_t_p90")
    show_mean    = t4.checkbox("Hourly mean",        value=False, key="thresh_t_mean")
    show_medoids = t5.checkbox("Medoid markers",     value=True,  key="thresh_t_med")

    # ── Column detection ──────────────────────────────────────────────────────
    cols          = _detect_cols(df_raw, ctx)
    speed_col     = cols["speed"]
    id_col        = cols["id"]
    day_type_col  = cols["day_type"]
    hour_col      = cols["hour"]
    road_name_col = cols["road_name"]
    cluster_col   = "cluster_label"

    if not speed_col or not hour_col:
        st.error(
            f"Could not detect **SPEED** or **hour** columns.  "
            f"Columns found: `{list(df_raw.columns)}`"
        )
        return

    df = df_raw.copy()
    df[hour_col]    = pd.to_numeric(df[hour_col],    errors="coerce")
    df[speed_col]   = pd.to_numeric(df[speed_col],   errors="coerce")
    df[cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[hour_col, speed_col, cluster_col])
    df[cluster_col] = df[cluster_col].astype(int)

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2 = st.columns([1, 1])
    with fc1:
        day_opts = ["All"]
        if day_type_col and day_type_col in df.columns:
            day_opts += sorted(df[day_type_col].dropna().unique().tolist())
        day_sel = st.selectbox("Day type", day_opts, key="thresh_day")
    with fc2:
        scatter_sample = st.number_input(
            "Max scatter points", min_value=500, max_value=100_000,
            value=10_000, step=500, key="thresh_scatter_n",
        )

    has_road = road_name_col and road_name_col in df.columns
    has_id   = id_col and id_col in df.columns

    road_sel = []
    if has_road:
        all_roads = sorted(df[road_name_col].dropna().unique().tolist())
        prev_road = st.session_state.get("_thresh_prev_road", [])
        road_sel  = st.multiselect(
            f"Filter by {road_name_col} (leave empty = all)",
            all_roads, default=[], key="thresh_road_names",
        )
        if road_sel != prev_road:
            st.session_state["_thresh_prev_road"] = road_sel
            st.session_state.pop("thresh_ids", None)

    id_sel = []
    if has_id:
        id_pool = sorted(
            (df[df[road_name_col].isin(road_sel)] if (has_road and road_sel) else df)
            [id_col].dropna().unique().tolist()
        )
        id_default = id_pool[:5]
        id_sel = st.multiselect(
            f"Filter by {id_col} (leave empty = all)",
            id_pool, default=id_default, key="thresh_ids",
        )
        if has_road and road_sel:
            st.caption(f"{len(id_pool)} {id_col}(s) on selected road(s)")

    # ── Apply filters ─────────────────────────────────────────────────────────
    # Keep cluster thresholds stable across road / link filters by computing
    # them from the day-filtered base set, then applying them to the visible rows.
    df_thresh = df.copy()
    if day_sel != "All" and day_type_col and day_type_col in df_thresh.columns:
        df_thresh = df_thresh[df_thresh[day_type_col] == day_sel]

    df_filt = df_thresh.copy()
    if road_sel and has_road:
        df_filt = df_filt[df_filt[road_name_col].isin(road_sel)]
    if id_sel and has_id:
        df_filt = df_filt[df_filt[id_col].isin(id_sel)]

    if df_filt.empty:
        st.warning("No data matches the current filters.")
        return

    cluster_ids = sorted(df_filt[cluster_col].unique().tolist())
    n_pts = len(df_filt)

    # ── Compute cluster bands & medoids ───────────────────────────────────────
    bands_cache = st.session_state.setdefault("threshold_bands_cache", {})
    band_cache_key = ("speed", cache_key, day_sel)
    if band_cache_key not in bands_cache:
        bands_cache[band_cache_key] = compute_cluster_bands(
            df_thresh, speed_col, hour_col, cluster_col, include_hourly=False
        )[0]
    overall_bands = bands_cache[band_cache_key]

    medoids_df = pd.DataFrame()
    if show_medoids:
        medoid_cache = st.session_state.setdefault("threshold_medoids_cache", {})
        medoid_cache_key = (
            "speed",
            cache_key,
            day_sel,
            tuple(str(v) for v in road_sel),
            tuple(str(v) for v in id_sel),
        )
        if medoid_cache_key not in medoid_cache:
            medoid_cache[medoid_cache_key] = compute_medoids(
                df_filt, speed_col, hour_col, cluster_col
            )
        medoids_df = medoid_cache[medoid_cache_key]

    # ── Build Plotly figure ───────────────────────────────────────────────────
    fig = go.Figure()

    # 1. Scatter — colour-coded by cluster
    if show_points:
        df_scatter = df_filt.sample(min(int(scatter_sample), n_pts), random_state=42)
        for cid in cluster_ids:
            grp = df_scatter[df_scatter[cluster_col] == cid]
            if grp.empty:
                continue
            jitter = np.random.default_rng(cid).uniform(-0.35, 0.35, len(grp))
            fig.add_trace(go.Scatter(
                x=grp[hour_col].values + jitter,
                y=grp[speed_col].values,
                mode="markers",
                name=f"Cluster {cid}",
                marker=dict(color=cluster_color(cid), size=4, opacity=0.35),
                legendgroup=f"c{cid}",
                hovertemplate=(
                    f"Cluster {cid}<br>Hour: %{{x:.1f}}<br>"
                    "Speed: %{y:.1f} km/h<extra></extra>"
                ),
            ))

    # 2. Single continuous threshold lines — dominant cluster per hour
    # For each hour, pick the cluster with the most observations; use its
    # overall P10/P90 (from overall_bands) to form one continuous line.
    dominant_per_hour = (
        df_filt.groupby(hour_col)[cluster_col]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={cluster_col: "dominant_cluster"})
        .sort_values(hour_col)
    )
    band_map = overall_bands.set_index("cluster_label")[["p10", "p90", "mean", "size"]]
    dominant_per_hour = dominant_per_hour.join(band_map, on="dominant_cluster")

    if show_p10:
        fig.add_trace(go.Scatter(
            x=dominant_per_hour[hour_col],
            y=dominant_per_hour["p10"],
            mode="lines+markers",
            name="P10 threshold",
            line=dict(color="#E63946", width=2.5),
            marker=dict(size=5),
            customdata=dominant_per_hour[["dominant_cluster", "size"]].values,
            hovertemplate=(
                "Hour: %{x}<br>"
                "<b>P10 threshold: %{y:.1f} km/h</b><br>"
                "Dominant cluster: %{customdata[0]}<br>"
                "Cluster size: %{customdata[1]:,}<extra></extra>"
            ),
        ))

    if show_p90:
        fig.add_trace(go.Scatter(
            x=dominant_per_hour[hour_col],
            y=dominant_per_hour["p90"],
            mode="lines+markers",
            name="P90 upper band",
            line=dict(color="#1565C0", width=2.5, dash="dot"),
            marker=dict(size=5),
            customdata=dominant_per_hour[["dominant_cluster", "size"]].values,
            hovertemplate=(
                "Hour: %{x}<br>"
                "P90 upper band: %{y:.1f} km/h<br>"
                "Dominant cluster: %{customdata[0]}<br>"
                "Cluster size: %{customdata[1]:,}<extra></extra>"
            ),
        ))

    if show_mean:
        fig.add_trace(go.Scatter(
            x=dominant_per_hour[hour_col],
            y=dominant_per_hour["mean"],
            mode="lines",
            name="Hourly mean",
            line=dict(color="#2D6A4F", width=1.5, dash="dash"),
            hovertemplate=(
                "Hour: %{x}<br>Mean: %{y:.1f} km/h<extra></extra>"
            ),
        ))

    # 3. Medoid markers — one star per cluster
    if show_medoids and not medoids_df.empty:
        for _, row in medoids_df.iterrows():
            cid = int(row[cluster_col])
            if cid not in cluster_ids:
                continue
            fig.add_trace(go.Scatter(
                x=[row[hour_col]],
                y=[row[speed_col]],
                mode="markers",
                name=f"Cluster {cid} — medoid",
                marker=dict(
                    color=cluster_color(cid), size=16, symbol="star",
                    line=dict(width=1.5, color="white"),
                ),
                legendgroup=f"c{cid}",
                showlegend=True,
                hovertemplate=(
                    f"<b>Cluster {cid} medoid</b><br>"
                    f"Hour: {int(row[hour_col])}<br>"
                    f"Speed: {row[speed_col]:.1f} km/h<extra></extra>"
                ),
            ))

    # ── Layout ────────────────────────────────────────────────────────────────
    title = f"Speed vs Hour of Day — {sel_stem}"
    if day_sel != "All":
        title += f"  [{day_sel}]"
    if road_sel:
        road_str = ", ".join(str(r) for r in road_sel[:2])
        if len(road_sel) > 2:
            road_str += f" +{len(road_sel) - 2} more"
        title += f"  |  Road: {road_str}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear", tick0=0, dtick=1,
            range=[-0.5, 23.5], gridcolor="#EEEEEE",
        ),
        yaxis=dict(title="Speed (km/h)", gridcolor="#EEEEEE", rangemode="tozero"),
        plot_bgcolor="white", paper_bgcolor="white", height=540,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#CCCCCC", borderwidth=1,
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        hovermode="closest",
        margin=dict(t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Scatter: {min(int(scatter_sample), n_pts):,} of {n_pts:,} points shown  ·  "
        f"{len(cluster_ids)} cluster(s) visible  ·  "
        "P10/P90 computed per cluster from the day-filtered clustering result.  "
        "Click legend item to toggle; double-click to isolate."
    )

    # ── Cluster band summary table ────────────────────────────────────────────
    with st.expander("Cluster band summary (P10 / P90 / mean per cluster)"):
        display_bands = overall_bands.rename(columns={
            "cluster_label": "Cluster",
            "p10":  "P10 threshold (km/h)",
            "p90":  "P90 upper band (km/h)",
            "mean": "Mean speed (km/h)",
            "size": "Records",
        })
        st.dataframe(display_bands, use_container_width=True, hide_index=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    band_lookup = overall_bands.set_index("cluster_label")[["p10", "p90"]].rename(
        columns={"p10": "cluster_p10_kmh", "p90": "cluster_p90_kmh"}
    )
    df_out = df_filt.copy()
    df_out = df_out.join(band_lookup, on=cluster_col)
    export_cols = [
        c for c in df_filt.columns if c != cluster_col
    ] + [cluster_col, "cluster_p10_kmh", "cluster_p90_kmh"]
    df_out = df_out[export_cols]

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇  Download enriched dataset (CSV)",
            data=df_out.to_csv(index=False),
            file_name="speed_data_with_cluster_thresholds.csv",
            mime="text/csv",
            help=(
                "Filtered rows + cluster_label + cluster_p10_kmh + "
                "cluster_p90_kmh."
            ),
        )
    with dl2:
        display_bands_dl = overall_bands.rename(columns={
            "cluster_label": "Cluster",
            "p10": "P10 threshold (km/h)", "p90": "P90 upper band (km/h)",
            "mean": "Mean speed (km/h)", "size": "Records",
        })
        st.download_button(
            "⬇  Download cluster band summary (CSV)",
            data=display_bands_dl.to_csv(index=False),
            file_name="speed_cluster_bands.csv",
            mime="text/csv",
            help="Per-cluster P10 / P90 / mean / record count.",
        )

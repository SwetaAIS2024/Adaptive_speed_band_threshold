"""
Tab: Volume Bands
Derives P10/P90 volume bands from volume cluster assignments.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .context import AppContext, get_pipeline_state
from ._threshold_shared import discover_cluster_results, get_cluster_result, get_enriched_df


def render(ctx: AppContext) -> None:
    st.header("Volume Bands")

    ps = get_pipeline_state(ctx)
    if not ps["clustering_done"]:
        st.info(
            "🔒 **Step 3 not complete** — no clustering results found.  "
            "Complete **Feature Engineering** (Step 2) then run **Clustering** (Step 3) first."
        )
        return

    volume_results = discover_cluster_results(ctx, "volume")

    if not volume_results:
        st.info(
            "Run a **volume** clustering in the **Clustering** tab first, "
            "then return here."
        )
        return

    sel_stem = st.selectbox(
        "Select volume clustering result",
        list(volume_results.keys()),
    )
    if sel_stem is None:
        return

    cache_key = volume_results[sel_stem]
    result = get_cluster_result(ctx, cache_key)
    if result is None:
        st.error("Could not load the saved clustering result.")
        return

    # Prefer original-unit enriched DF; fall back to normalised feature DF
    enriched_res = get_enriched_df(ctx, cache_key)
    if enriched_res is not None:
        res_df     = enriched_res
        ec_res     = ctx.eda_mod._detect_cols(res_df) if ctx.eda_mod else {}
        volume_col = ec_res.get("volume") or "VOLUME"
        vol_unit   = "veh/hr"
    else:
        res_df     = result.assigned_df
        col_map    = {c.lower(): c for c in res_df.columns}
        volume_col = col_map.get("volume", "VOLUME")
        vol_unit   = "normalised [0-1]"
        st.info(
            "Preprocessed data not available — showing normalised volume values. "
            "Re-run clustering to see original veh/hr values."
        )

    if volume_col not in res_df.columns:
        st.error(f"Volume column `{volume_col}` not found in the dataset.")
        return

    # ── Compute bands ─────────────────────────────────────────────────────────
    bands = (
        res_df.groupby("cluster_label")[volume_col]
        .agg(
            lower_band=lambda x: round(x.quantile(0.10), 2),
            upper_band=lambda x: round(x.quantile(0.90), 2),
            vol_mean=lambda x:   round(x.mean(), 2),
            cluster_size="count",
        )
        .reset_index()
        .sort_values("vol_mean")
        .reset_index(drop=True)
    )
    ordered = bands["cluster_label"].astype(str).tolist()

    # ── Band range chart ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Volume Band Ranges per Cluster")
    st.caption(
        f"Each bar spans **P10 to P90** of the cluster's volume distribution ({vol_unit}).  "
        "The diamond marker shows the mean. Narrow bars = tight, well-defined regime."
    )

    colors    = px.colors.qualitative.Safe
    fig_bands = go.Figure()
    for i, (_, row) in enumerate(bands.iterrows()):
        c   = colors[i % len(colors)]
        lbl = str(int(row["cluster_label"]))
        fig_bands.add_trace(go.Bar(
            name=lbl,
            x=[row["upper_band"] - row["lower_band"]],
            y=[lbl],
            base=[row["lower_band"]],
            orientation="h",
            marker_color=c,
            marker_line=dict(color="rgba(0,0,0,0.25)", width=1),
            hovertemplate=(
                f"<b>Cluster {lbl}</b><br>"
                f"Lower (P10): {row['lower_band']} {vol_unit}<br>"
                f"Upper (P90): {row['upper_band']} {vol_unit}<br>"
                f"Mean: {row['vol_mean']} {vol_unit}<br>"
                f"Width: {row['upper_band'] - row['lower_band']:.2f} {vol_unit}<br>"
                f"Size: {int(row['cluster_size']):,}<extra></extra>"
            ),
            showlegend=False,
        ))
        fig_bands.add_trace(go.Scatter(
            x=[row["vol_mean"]], y=[lbl],
            mode="markers",
            marker=dict(color="black", size=11, symbol="diamond"),
            showlegend=(i == 0), name="Mean volume",
            hoverinfo="skip",
        ))

    fig_bands.update_layout(
        title=f"Volume Bands — {sel_stem}",
        xaxis=dict(title=f"Volume ({vol_unit})", gridcolor="#EEEEEE"),
        yaxis=dict(title="Cluster", categoryorder="array", categoryarray=ordered),
        barmode="overlay",
        height=max(320, 52 * len(bands) + 80),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_bands, use_container_width=True)

    # ── Cluster deep-dive ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Cluster Deep-Dive")
    sel_cluster     = st.selectbox("Select cluster", bands["cluster_label"].tolist())
    if sel_cluster is None:
        return
    cluster_volumes = res_df[res_df["cluster_label"] == sel_cluster][volume_col]
    p10      = cluster_volumes.quantile(0.10)
    p90      = cluster_volumes.quantile(0.90)
    mean_vol = cluster_volumes.mean()

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Cluster Size",      f"{len(cluster_volumes):,}")
    mc2.metric("Lower Band (P10)",  f"{p10:.1f} {vol_unit}")
    mc3.metric("Upper Band (P90)",  f"{p90:.1f} {vol_unit}")
    mc4.metric("Band Width",        f"{p90 - p10:.1f} {vol_unit}")

    vol_margin = max(1.0, (cluster_volumes.max() - cluster_volumes.min()) * 0.02)
    bins       = np.linspace(
        max(0, cluster_volumes.min() - vol_margin),
        cluster_volumes.max() + vol_margin, 55,
    )
    counts_h, edges = np.histogram(cluster_volumes, bins=bins)
    bar_w = np.diff(edges)

    fig_hist = go.Figure()
    for mask, color, label in [
        (edges[:-1] < p10,                          "rgba(180,180,180,0.55)", "Below P10"),
        ((edges[:-1] >= p10) & (edges[:-1] < p90), "rgba(56,100,200,0.75)",  "Volume Band P10 to P90"),
        (edges[:-1] >= p90,                         "rgba(180,180,180,0.55)", "Above P90"),
    ]:
        if mask.any():
            fig_hist.add_trace(go.Bar(
                x=edges[:-1][mask], y=counts_h[mask],
                width=bar_w[mask], name=label,
                marker_color=color, marker_line_width=0, offset=0,
            ))

    for val, color, text, dash in [
        (p10,     "#1565C0", f"P10={p10:.1f}",    "dash"),
        (p90,     "#B71C1C", f"P90={p90:.1f}",    "dash"),
        (mean_vol,"#333333", f"Mean={mean_vol:.1f}","dot"),
    ]:
        fig_hist.add_vline(
            x=val, line_color=color, line_width=2, line_dash=dash,
            annotation_text=f"  {text}",
            annotation_font=dict(color=color, size=11),
        )

    fig_hist.update_layout(
        title=f"Volume Distribution — Cluster {int(sel_cluster)}",
        xaxis=dict(title=f"Volume ({vol_unit})", gridcolor="#EEEEEE"),
        yaxis=dict(title="Count",                gridcolor="#EEEEEE"),
        barmode="stack", bargap=0.02, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Box plots across all clusters ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Volume Distribution across All Clusters")
    fig_box = px.box(
        res_df,
        x=res_df["cluster_label"].astype(str),
        y=volume_col,
        color=res_df["cluster_label"].astype(str),
        category_orders={"x": ordered},
        title=f"Volume Box Plots — {sel_stem}",
        labels={"x": "Cluster", volume_col: f"Volume ({vol_unit})"},
        points=False,
    )
    fig_box.update_layout(showlegend=False, height=380, plot_bgcolor="white")
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Band summary + download ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Band Summary")
    st.dataframe(
        bands.rename(columns={
            "cluster_label": "Cluster",
            "lower_band":    f"Lower P10 ({vol_unit})",
            "upper_band":    f"Upper P90 ({vol_unit})",
            "vol_mean":      f"Mean Volume ({vol_unit})",
            "cluster_size":  "Size",
        }),
        use_container_width=True, hide_index=True,
    )
    export_df = res_df.merge(
        bands[["cluster_label", "lower_band", "upper_band"]],
        on="cluster_label", how="left",
    )
    st.download_button(
        "⬇  Download with bands (CSV)",
        data=export_df.to_csv(index=False),
        file_name=f"volume_bands_{sel_stem}.csv",
        mime="text/csv",
    )

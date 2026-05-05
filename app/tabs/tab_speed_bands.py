"""
Tab: Speed Band Derivation
Derives P10/P90 speed bands from cluster assignments (original km/h values).
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .context import AppContext


def render(ctx: AppContext) -> None:
    st.header("Speed & Volume Band Derivation")

    speed_results = {
        k: v for k, v in st.session_state.cluster_results.items()
        if k[1] == "speed"
    }

    if not speed_results:
        st.info(
            "Run a **speed** clustering in the **Clustering** tab first, "
            "then return here."
        )
        return

    sel_key = st.selectbox(
        "View results for",
        list(speed_results.keys()),
        format_func=lambda k: k[0],
    )
    result = speed_results[sel_key]

    # Prefer original-unit enriched DF; fall back to normalised feature DF
    enriched_res = st.session_state.cluster_enriched.get(sel_key)
    if enriched_res is not None:
        res_df    = enriched_res
        ec_res    = ctx.eda_mod._detect_cols(res_df) if ctx.eda_mod else {}
        speed_col = ec_res.get("speed") or "SPEED"
        spd_unit  = "km/h"
    else:
        res_df    = result.assigned_df
        col_map   = {c.lower(): c for c in res_df.columns}
        speed_col = col_map.get("speed", "SPEED")
        spd_unit  = "normalised [0-1]"
        st.info(
            "Preprocessed data not available — showing normalised speed values. "
            "Re-run clustering to see original km/h values."
        )

    # ── Compute bands ─────────────────────────────────────────────────────────
    bands = (
        res_df.groupby("cluster_label")[speed_col]
        .agg(
            lower_band=lambda x: round(x.quantile(0.10), 2),
            upper_band=lambda x: round(x.quantile(0.90), 2),
            spd_mean=lambda x:   round(x.mean(), 2),
            cluster_size="count",
        )
        .reset_index()
        .sort_values("spd_mean")
        .reset_index(drop=True)
    )
    ordered = bands["cluster_label"].astype(str).tolist()

    # ── Band range chart ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Speed Band Ranges per Cluster")
    st.caption(
        f"Each bar spans **P10 to P90** of the cluster's speed distribution ({spd_unit}).  "
        "The diamond marker shows the mean. Narrow bars = tight, well-defined regime."
    )

    colors    = px.colors.qualitative.Safe
    fig_bands = go.Figure()
    for i, row in bands.iterrows():
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
                f"Lower (P10): {row['lower_band']} {spd_unit}<br>"
                f"Upper (P90): {row['upper_band']} {spd_unit}<br>"
                f"Mean: {row['spd_mean']} {spd_unit}<br>"
                f"Width: {row['upper_band'] - row['lower_band']:.2f} {spd_unit}<br>"
                f"Size: {int(row['cluster_size']):,}<extra></extra>"
            ),
            showlegend=False,
        ))
        fig_bands.add_trace(go.Scatter(
            x=[row["spd_mean"]], y=[lbl],
            mode="markers",
            marker=dict(color="black", size=11, symbol="diamond"),
            showlegend=(i == 0), name="Mean speed",
            hoverinfo="skip",
        ))

    fig_bands.update_layout(
        title=f"Speed Bands — {sel_key[0]}",
        xaxis=dict(title=f"Speed ({spd_unit})", gridcolor="#EEEEEE"),
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
    sel_cluster    = st.selectbox("Select cluster", bands["cluster_label"].tolist())
    cluster_speeds = res_df[res_df["cluster_label"] == sel_cluster][speed_col]
    p10      = cluster_speeds.quantile(0.10)
    p90      = cluster_speeds.quantile(0.90)
    mean_spd = cluster_speeds.mean()

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Cluster Size",     f"{len(cluster_speeds):,}")
    mc2.metric("Lower Band (P10)", f"{p10:.1f} {spd_unit}")
    mc3.metric("Upper Band (P90)", f"{p90:.1f} {spd_unit}")
    mc4.metric("Band Width",        f"{p90 - p10:.1f} {spd_unit}")

    spd_margin = max(1.0, (cluster_speeds.max() - cluster_speeds.min()) * 0.02)
    bins       = np.linspace(
        max(0, cluster_speeds.min() - spd_margin),
        cluster_speeds.max() + spd_margin, 55,
    )
    counts_h, edges = np.histogram(cluster_speeds, bins=bins)
    bar_w = np.diff(edges)

    fig_hist = go.Figure()
    for mask, color, label in [
        (edges[:-1] < p10,                          "rgba(180,180,180,0.55)", "Below P10"),
        ((edges[:-1] >= p10) & (edges[:-1] < p90), "rgba(56,169,70,0.75)",   "Speed Band P10 to P90"),
        (edges[:-1] >= p90,                         "rgba(180,180,180,0.55)", "Above P90"),
    ]:
        if mask.any():
            fig_hist.add_trace(go.Bar(
                x=edges[:-1][mask], y=counts_h[mask],
                width=bar_w[mask], name=label,
                marker_color=color, marker_line_width=0, offset=0,
            ))

    for val, color, text, dash in [
        (p10,      "#1565C0", f"P10={p10:.1f}",      "dash"),
        (p90,      "#B71C1C", f"P90={p90:.1f}",      "dash"),
        (mean_spd, "#333333", f"Mean={mean_spd:.1f}", "dot"),
    ]:
        fig_hist.add_vline(
            x=val, line_color=color, line_width=2, line_dash=dash,
            annotation_text=f"  {text}",
            annotation_font=dict(color=color, size=11),
        )

    fig_hist.update_layout(
        title=f"Speed Distribution — Cluster {int(sel_cluster)}",
        xaxis=dict(title=f"Speed ({spd_unit})", gridcolor="#EEEEEE"),
        yaxis=dict(title="Count",               gridcolor="#EEEEEE"),
        barmode="stack", bargap=0.02, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Box plots across all clusters ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Speed Distribution across All Clusters")
    fig_box = px.box(
        res_df,
        x=res_df["cluster_label"].astype(str),
        y=speed_col,
        color=res_df["cluster_label"].astype(str),
        category_orders={"x": ordered},
        title=f"Speed Box Plots — {sel_key[0]}",
        labels={"x": "Cluster", speed_col: f"Speed ({spd_unit})"},
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
            "lower_band":    f"Lower P10 ({spd_unit})",
            "upper_band":    f"Upper P90 ({spd_unit})",
            "spd_mean":      f"Mean Speed ({spd_unit})",
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
        file_name=f"speed_bands_{sel_key[0]}.csv",
        mime="text/csv",
    )

"""
Tab: Clustering
K-Means speed / volume clustering on feature-engineered parquet files.
Results are stored in st.session_state so the Speed Bands tab can consume them.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .context import (
    AppContext,
    get_pipeline_state,
    csv_stem_from_feature_stem,
    enrich_with_clusters,
    load_feature_metadata,
)


def _cached_discover_features(_cl_mod) -> dict:
    """Return {stem: str(path)} — re-scans on every rerun so new parquets are picked up."""
    return {stem: str(path) for stem, path in _cl_mod.discover_feature_files().items()}


def _run_clustering(cl_mod, stem, path_str, clustering_type, k_override):
    """Run clustering (not cached — result stored in session_state by caller)."""
    df    = pd.read_parquet(path_str)
    fcols = cl_mod.feature_cols_for(df, clustering_type)
    if not fcols:
        return None
    return cl_mod.run_one(df, fcols, stem, clustering_type, k_override=k_override, save=True)


def render(ctx: AppContext) -> None:
    st.header("Clustering")

    ps = get_pipeline_state(ctx)
    if not ps["features_done"]:
        st.warning(
            "⚠️ **Step 2 not complete** — no feature parquet files found in "
            "`clustering/features/`.  "
            "Go to the **Feature Engineering** tab and run it first."
        )
        return

    st.caption(
        "Select a feature file and clustering type, then run clustering. "
        "Speed and Volume are clustered separately. "
        "K is chosen automatically by peak silhouette score."
    )

    if ctx.cl_mod is None:
        st.error("Clustering module not available.")
        return

    feature_files = _cached_discover_features(ctx.cl_mod)

    if not feature_files:
        st.warning(
            "No feature parquet files found in `clustering/features/`. "
            "Go to the **Feature Engineering** tab to generate them first."
        )
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([2.5, 1, 1])
    with fc1:
        sel_stem = st.selectbox("Feature file", list(feature_files.keys()))
    with fc2:
        probe_df    = pd.read_parquet(feature_files[sel_stem])
        avail_types = [
            ct for ct in ("speed", "volume")
            if ctx.cl_mod.feature_cols_for(probe_df, ct)
        ]
        sel_ctype = st.selectbox("Clustering type", avail_types)
    with fc3:
        fcols_display = ctx.cl_mod.feature_cols_for(probe_df, sel_ctype)
        k_override = st.number_input(
            "Override K (0 = auto)", min_value=0, max_value=10, value=0, step=1
        )

    st.caption(f"Features used: `{', '.join(fcols_display)}`")

    run_btn   = st.button("▶  Run Clustering", type="primary")
    cache_key = (sel_stem, sel_ctype, int(k_override) or None)

    if run_btn or cache_key not in st.session_state.cluster_results:
        with st.spinner(f"Running {sel_ctype} clustering on {sel_stem} ..."):
            result = _run_clustering(
                ctx.cl_mod,
                sel_stem,
                feature_files[sel_stem],
                sel_ctype,
                int(k_override) if k_override > 0 else None,
            )
        if result is None:
            st.error(f"No `{sel_ctype.upper()}` column found in this file.")
            return
        st.session_state.cluster_results[cache_key] = result
        st.session_state.cluster_enriched[cache_key] = enrich_with_clusters(
            sel_stem, result.assigned_df, ctx.data_root
        )

    if cache_key not in st.session_state.cluster_results:
        return

    result      = st.session_state.cluster_results[cache_key]
    adf         = result.assigned_df
    cdf         = result.centers_df
    enriched_df = st.session_state.cluster_enriched.get(cache_key)

    sil_str = (
        f"{result.best_sil:.3f}"
        if not np.isnan(result.best_sil)
        else "n/a (K overridden)"
    )
    st.success(f"K = {result.k_opt}  |  Silhouette = {sil_str}")

    # ── K-Sweep diagnostics ───────────────────────────────────────────────────
    if not result.diag_df.empty:
        st.markdown("---")
        st.subheader("K-Sweep Diagnostics")
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(
            x=result.diag_df["k"], y=result.diag_df["silhouette"],
            mode="lines+markers", name="Silhouette ↑",
            line=dict(color="#2196F3", width=2),
        ))
        fig_k.add_trace(go.Scatter(
            x=result.diag_df["k"], y=result.diag_df["davies_bouldin"],
            mode="lines+markers", name="Davies-Bouldin ↓",
            line=dict(color="#F44336", width=2, dash="dash"),
        ))
        fig_k.add_vline(
            x=result.k_opt, line_dash="dot", line_color="green",
            annotation_text=f"  K = {result.k_opt}",
            annotation_font=dict(color="green", size=13),
        )
        fig_k.update_layout(
            title=f"K Sweep — {sel_stem} / {sel_ctype}",
            xaxis=dict(title="K", dtick=1),
            yaxis_title="Score",
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_k, use_container_width=True)

    # ── Cluster sizes ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Cluster Sizes")
    size_df = (
        adf["cluster_label"].value_counts().sort_index()
        .reset_index()
        .rename(columns={"cluster_label": "Cluster", "count": "Records"})
    )
    fig_sz = px.bar(
        size_df, x="Cluster", y="Records",
        color="Records", color_continuous_scale="Blues",
        title="Records per Cluster",
    )
    fig_sz.update_layout(height=280, coloraxis_showscale=False)
    st.plotly_chart(fig_sz, use_container_width=True)

    # ── Cluster patterns — original (un-normalised) values ────────────────────
    st.markdown("---")
    st.subheader("Cluster Patterns — Original Values")
    if enriched_df is not None:
        ec          = ctx.eda_mod._detect_cols(enriched_df) if ctx.eda_mod else {}
        orig_target = ec.get("speed") if sel_ctype == "speed" else ec.get("volume")
        hour_col    = next((c for c in enriched_df.columns if c.lower() == "hour"), None)
        c_order     = sorted(enriched_df["cluster_label"].unique())

        if orig_target:
            esample            = enriched_df.sample(min(8_000, len(enriched_df)), random_state=42).copy()
            esample["Cluster"] = esample["cluster_label"].astype(str)

            sc1, sc2 = st.columns(2)
            with sc1:
                if hour_col:
                    fig_s1 = px.scatter(
                        esample, x=hour_col, y=orig_target,
                        color="Cluster", opacity=0.4,
                        title=f"Hour of Day vs {orig_target}",
                        labels={
                            hour_col:    "Hour (0–23)",
                            orig_target: f"{orig_target} (original)",
                        },
                        category_orders={"Cluster": [str(c) for c in c_order]},
                    )
                    fig_s1.update_traces(marker=dict(size=3))
                    fig_s1.update_layout(height=380, plot_bgcolor="white")
                    st.plotly_chart(fig_s1, use_container_width=True)
                else:
                    st.info("No `hour` column in preprocessed data.")
            with sc2:
                fig_s2 = px.box(
                    enriched_df,
                    x=enriched_df["cluster_label"].astype(str),
                    y=orig_target,
                    color=enriched_df["cluster_label"].astype(str),
                    category_orders={"x": [str(c) for c in c_order]},
                    title=f"{orig_target} Distribution per Cluster",
                    labels={"x": "Cluster", orig_target: f"{orig_target} (original)"},
                    points=False,
                )
                fig_s2.update_layout(showlegend=False, height=380, plot_bgcolor="white")
                st.plotly_chart(fig_s2, use_container_width=True)

            # Heatmap: mean target by hour × cluster
            if hour_col:
                st.markdown("---")
                heat_pivot = (
                    enriched_df.groupby(["cluster_label", hour_col])[orig_target]
                    .mean().unstack("cluster_label").round(1)
                )
                heat_pivot.index.name = "Hour"
                heat_pivot.columns    = [f"Cluster {c}" for c in heat_pivot.columns]
                fig_heat = px.imshow(
                    heat_pivot.T,
                    color_continuous_scale="RdYlGn",
                    title=f"Mean {orig_target} by Hour and Cluster (original values)",
                    text_auto=".1f",
                    aspect="auto",
                )
                fig_heat.update_layout(
                    height=300,
                    xaxis_title="Hour of Day",
                    yaxis_title="Cluster",
                )
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info(f"Could not detect `{sel_ctype.upper()}` column in preprocessed data.")
    else:
        st.warning(
            "Preprocessed data not found — cannot show original-unit charts. "
            "Re-run clustering to rebuild."
        )

    # ── Centroids with inverse-transformed original values ────────────────────
    st.markdown("---")
    st.subheader("Cluster Centroids")
    meta_all    = load_feature_metadata(str(ctx.cl_mod.FEATURES_DIR))
    csv_s       = csv_stem_from_feature_stem(sel_stem)
    fmeta       = meta_all.get(csv_s, {})
    cdf_display = cdf.copy()
    if "SPEED" in cdf_display.columns and "speed_min_raw" in fmeta:
        s_rng = fmeta["speed_max_raw"] - fmeta["speed_min_raw"]
        cdf_display["SPEED (km/h)"] = (
            cdf_display["SPEED"] * s_rng + fmeta["speed_min_raw"]
        ).round(1)
    if "VOLUME" in cdf_display.columns and "volume_min_raw" in fmeta:
        v_rng = fmeta["volume_max_raw"] - fmeta["volume_min_raw"]
        cdf_display["VOLUME (original)"] = (
            cdf_display["VOLUME"] * v_rng + fmeta["volume_min_raw"]
        ).round(1)
    st.dataframe(cdf_display, use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    dl_df   = enriched_df if enriched_df is not None else adf
    dl_label = (
        "preprocessed data + cluster labels"
        if enriched_df is not None
        else "feature data + cluster labels"
    )
    st.download_button(
        f"⬇  Download {dl_label} (CSV)",
        data=dl_df.to_csv(index=False),
        file_name=f"cluster_assignments_{sel_stem}_{sel_ctype}.csv",
        mime="text/csv",
    )

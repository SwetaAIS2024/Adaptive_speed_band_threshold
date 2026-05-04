"""
Adaptive Speed Band Threshold â€” Interactive Dashboard
======================================================
Visualise input traffic data, run K-Means clustering per road category / day
type, and explore how P10/P90 speed bands are derived from each cluster.

Compatible with ANY dataset that matches the 7-column schema below:
    timestamp_hour, LinkID, RoadCategory, hour, day_type, speed, volume

Run:
    streamlit run app/streamlit_app.py
"""

import importlib.util as _ilu
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# â”€â”€ Load clustering module by absolute path (Docker-safe, no sys.path needed) â”€
_CLUSTERING_MODULE_PATH = Path(__file__).resolve().parent.parent / "clustering" / "scripts" / "run_clustering.py"
_cl_spec = _ilu.spec_from_file_location("run_clustering", str(_CLUSTERING_MODULE_PATH))
_cl = _ilu.module_from_spec(_cl_spec)
_cl_spec.loader.exec_module(_cl)

# â”€â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="Adaptive Speed Band Threshold",
    page_icon="ðŸš—",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os

# ─── Module-level paths & EDA module ─────────────────────────────────────────
_DATA_ROOT       = Path(__file__).resolve().parent.parent / "data"
_EDA_REPORT      = _DATA_ROOT / "pre_processed_dataset" / "eda_report.txt"
_EDA_MODULE_PATH = _DATA_ROOT / "data_understanding_EDA.py"

_DATASET_DIRS = {
    "Raw datasets":          str(_DATA_ROOT / "raw_dataset"),
    "Preprocessed datasets": str(_DATA_ROOT / "pre_processed_dataset"),
}

_eda_mod = None
_eda_import_error = None
try:
    _spec = _ilu.spec_from_file_location("data_understanding_EDA", str(_EDA_MODULE_PATH))
    _eda_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_eda_mod)
except Exception as _e:
    _eda_import_error = str(_e)


@st.cache_data(show_spinner=False)
def _load_raw_cached(path_str: str) -> pd.DataFrame:
    return _eda_mod.load_any_file(path_str)


SKIP_FILES = {"preprocessing_summary.csv", "eda_report.txt"}


def _csv_stem_from_feature_stem(feature_stem: str) -> str:
    """Strip 'features_' prefix to get the preprocessed CSV stem."""
    return feature_stem[len("features_"):] if feature_stem.startswith("features_") else feature_stem


@st.cache_data(show_spinner=False)
def _load_feature_metadata() -> dict:
    """Load feature_metadata.json - raw min/max for inverse-transforming centroids."""
    import json as _json
    meta_path = Path(str(_cl.FEATURES_DIR)) / "feature_metadata.json"
    if meta_path.exists():
        return _json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _enrich_with_clusters(feature_stem: str, assigned_df: pd.DataFrame):
    """
    Load the preprocessed CSV for feature_stem and append cluster_label by row-index
    alignment (feature engineering preserves row order, no rows dropped).
    Returns None if the CSV is missing or row counts differ.
    """
    csv_stem = _csv_stem_from_feature_stem(feature_stem)
    csv_path = _DATA_ROOT / "pre_processed_dataset" / f"{csv_stem}.csv"
    if not csv_path.exists():
        return None
    preproc = pd.read_csv(str(csv_path))
    if len(preproc) != len(assigned_df):
        return None
    preproc = preproc.reset_index(drop=True)
    preproc["cluster_label"] = assigned_df["cluster_label"].values
    return preproc


# â”€â”€â”€ Streamlit cache wrappers around clustering module functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@st.cache_data(show_spinner=False)
def cached_discover_features() -> dict:
    """Return {stem: str(path)} â€” cached so it doesn't re-scan on every rerun."""
    return {stem: str(path) for stem, path in _cl.discover_feature_files().items()}


def cached_run_clustering(stem: str, path_str: str, clustering_type: str, k_override: int | None = None):
    """Run clustering. Result is cached in st.session_state.cluster_results by the caller."""
    df = pd.read_parquet(path_str)
    fcols = _cl.feature_cols_for(df, clustering_type)
    if not fcols:
        return None
    return _cl.run_one(df, fcols, stem, clustering_type, k_override=k_override, save=False)


def _discover_preproc_files() -> dict:
    """Return {display_name: abs_path_str} for preprocessed CSVs, skipping summary/report files."""
    if _eda_mod is None:
        return {}
    all_files = _eda_mod.discover_raw_files(str(_DATA_ROOT / "pre_processed_dataset"))
    return {
        name: path
        for name, path in all_files.items()
        if os.path.basename(path) not in SKIP_FILES
    }


# ─── Session state init ──────────────────────────────────────────────────────
for key, default in [("df", None), ("cluster_results", {}), ("cluster_enriched", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 Speed Band Threshold")
    st.caption("Data-driven adaptive speed bands for Singapore roads")
    st.markdown("---")

    # ── Auto-load from preprocessed directory ────────────────────────────
    _preproc_files = _discover_preproc_files()  # {display_name: abs_path_str}

    if _preproc_files:
        st.markdown("**Load preprocessed dataset**")
        _preproc_sel = st.selectbox(
            "Select file", ["(none)"] + list(_preproc_files.keys()),
            key="sidebar_preproc_sel",
        )
        if _preproc_sel != "(none)":
            _ppath = _preproc_files[_preproc_sel]
            if (
                st.session_state.get("_sidebar_loaded") != _ppath
                or st.session_state.df is None
            ):
                st.session_state.df = pd.read_csv(_ppath)
                st.session_state.cluster_results = {}
                st.session_state["_sidebar_loaded"] = _ppath
            st.success(f"Loaded {len(st.session_state.df):,} rows")
        st.markdown("---")

    # ── Or upload any CSV ─────────────────────────────────────────────────
    uploaded = st.file_uploader("Or upload any CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.session_state.df = df_upload
            st.session_state.cluster_results = {}
            st.session_state["_sidebar_loaded"] = None
            st.success(f"Loaded {len(df_upload):,} rows  ·  {len(df_upload.columns)} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if not _preproc_files and st.session_state.df is None:
        st.warning("No preprocessed files found. Run `python data/preprocess.py` first.")

    st.markdown("---")
    st.markdown("**Preprocessed file columns**")
    st.markdown(
        "| Column | Description |\n|--------|-------------|\n"
        "| `DATE_TIME` | Timestamp |\n"
        "| `LINK_ID` / `EQUIP_ID` | Sensor ID |\n"
        "| `SPEED` | Speed, normalised [0,1] |\n"
        "| `VOLUME` | Volume, normalised [0,1] (TIQ only) |\n"
        "| `ROAD_NAME` | Road name (SpeedGraph only) |\n"
        "| `date` | Date part |\n"
        "| `day_type` | Weekday / Weekend |\n"
        "| `hour_sin` | Cyclic hour encoding |\n"
        "| `hour_cos` | Cyclic hour encoding |"
    )

# ─── Gate ────────────────────────────────────────────────────────────────────
# EDA tab always visible; other tabs need a loaded dataset.
# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_eda, tab1, tab2, tab3 = st.tabs(["🔍  Raw Data EDA", "📋  Dataset Explorer", "🔮  Clustering", "🎯  Speed Bands"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB EDA — RAW DATASET EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Raw Dataset EDA")
    st.caption(
        "Exploratory analysis of source files in `data/raw_dataset/` or "
        "`data/pre_processed_dataset/`. Preprocessed files include derived columns: "
        "`hour`, `date`, and `day_type` (Weekday / Weekend from filename). "
        "Select a source and a file to inspect."
    )

    if _eda_import_error:
        st.error(
            f"Could not load EDA module from `{_EDA_MODULE_PATH}`.\n\n"
            f"Error: `{_eda_import_error}`"
        )
    elif _eda_mod is None:
        st.error("EDA module failed to load for an unknown reason.")
    else:
        _source_type = st.radio(
            "Dataset source",
            list(_DATASET_DIRS.keys()),
            horizontal=True,
        )
        _active_dir = _DATASET_DIRS[_source_type]
        _raw_files = _eda_mod.discover_raw_files(_active_dir)

        if not _raw_files:
            st.warning(f"No CSV or Excel files found in `{_active_dir}`. Run the relevant script to populate this folder.")
        else:
            sel_raw = st.selectbox("Select dataset", list(_raw_files.keys()))
            raw_df = _load_raw_cached(_raw_files[sel_raw])

            # ── Section 1: Feature Exploration ───────────────────────────────
            st.markdown("---")
            st.subheader("1. Feature Exploration")
            info = _eda_mod.get_feature_info(raw_df)
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{info['shape'][0]:,}")
            m2.metric("Columns", info['shape'][1])
            m3.metric("Column Names", ", ".join(info['columns']))
            st.markdown("**Data Types**")
            st.dataframe(info["dtypes"], use_container_width=True, hide_index=True)
            st.markdown("**First 10 rows**")
            st.dataframe(info["head"], use_container_width=True, hide_index=True)

            # ── Section 2: Data Quality ───────────────────────────────────────
            st.markdown("---")
            st.subheader("2. Data Quality Checks")
            quality = _eda_mod.get_quality_checks(raw_df)

            q1, q2 = st.columns(2)
            with q1:
                st.markdown("**Null / Missing Values**")
                st.dataframe(quality["null_df"], use_container_width=True, hide_index=True)
            with q2:
                st.markdown("**Duplicate Rows**")
                st.metric("Total Duplicates", f"{quality['dup_count']:,}",
                          delta=f"{quality['dup_pct']}% of rows", delta_color="inverse")
                if not quality["neg_zero"].empty:
                    st.markdown("**Negative / Zero Values (Numeric)**")
                    st.dataframe(quality["neg_zero"], use_container_width=True, hide_index=True)

            if quality["speed_col"] and quality["speed_anomalies"] is not None:
                anom = quality["speed_anomalies"]
                st.markdown(f"**Speed Anomalies (`{quality['speed_col']}` < 0 or > 200):** {len(anom):,} rows")
                if len(anom) > 0:
                    st.dataframe(anom.head(50), use_container_width=True, hide_index=True)

            if quality["date_range"]:
                dr = quality["date_range"]
                st.markdown(
                    f"**Date Range (`{dr['col']}`):** `{dr['min']}` → `{dr['max']}`"
                    f"  |  Unparseable: `{dr['unparseable']}`"
                )

            # ── Section 3: Categorical Analysis ──────────────────────────────
            st.markdown("---")
            st.subheader("3. Categorical Feature Analysis")
            cats = _eda_mod.get_category_analysis(raw_df)

            if not cats:
                st.info("No categorical columns found (excluding date/time column).")
            else:
                for col, data in cats.items():
                    with st.expander(f"{col}  —  {data['unique']} unique values"):
                        vc = data["value_counts"]
                        ca, cb = st.columns([1, 1])
                        with ca:
                            st.dataframe(vc.head(30), use_container_width=True, hide_index=True)
                        with cb:
                            fig_cat = px.bar(vc.head(20), x=col, y="Count", title=f"Top 20 — {col}")
                            fig_cat.update_layout(height=320, xaxis_tickangle=-30)
                            st.plotly_chart(fig_cat, use_container_width=True)

            # ── Section 4: Statistics & Temporal Analysis ─────────────────────
            st.markdown("---")
            st.subheader("4. General Statistics & Temporal Analysis")
            stats = _eda_mod.get_statistics(raw_df)

            if stats["describe"] is not None:
                st.markdown("**Descriptive Statistics**")
                st.dataframe(stats["describe"], use_container_width=True)

            t1, t2 = st.columns(2)
            with t1:
                if stats["hourly_speed"] is not None:
                    hs = stats["hourly_speed"]
                    fig_hs = px.line(hs, x="Hour", y=hs.columns[1], markers=True,
                                     title=f"Mean {stats['speed_col']} by Hour")
                    fig_hs.update_layout(height=320)
                    st.plotly_chart(fig_hs, use_container_width=True)
            with t2:
                if stats["hourly_volume"] is not None:
                    hv = stats["hourly_volume"]
                    fig_hv = px.line(hv, x="Hour", y=hv.columns[1], markers=True,
                                     title=f"Mean {stats['volume_col']} by Hour",
                                     color_discrete_sequence=["#DD8452"])
                    fig_hv.update_layout(height=320)
                    st.plotly_chart(fig_hv, use_container_width=True)
                elif stats["road_speed"] is not None:
                    rs = stats["road_speed"]
                    fig_rs = px.bar(rs, x=stats["road_col"], y=rs.columns[1],
                                    title=f"Mean {stats['speed_col']} by {stats['road_col']}",
                                    color=rs.columns[1], color_continuous_scale="RdYlGn")
                    fig_rs.update_layout(height=320, xaxis_tickangle=-30)
                    st.plotly_chart(fig_rs, use_container_width=True)

            if stats["id_summary"] is not None:
                st.markdown(f"**Top 20 `{stats['id_col']}` by record count**")
                st.dataframe(stats["id_summary"], use_container_width=True, hide_index=True)

            # ── Section 5: Cross-Dataset Comparison ──────────────────────────
            st.markdown("---")
            st.subheader("5. Cross-Dataset Comparison")
            cross_df = _eda_mod.get_cross_comparison({k.split("  [")[0]: v for k, v in _raw_files.items()})
            st.dataframe(cross_df, use_container_width=True, hide_index=True)

            # ── Full EDA Report ───────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Full EDA Report (Text)")
            if _EDA_REPORT.exists():
                report_text = _EDA_REPORT.read_text(encoding="utf-8")
                st.download_button("⬇  Download full EDA report", data=report_text,
                                   file_name="eda_report.txt", mime="text/plain")
                with st.expander("View raw EDA report text"):
                    st.code(report_text, language=None)
            else:
                st.warning("`eda_report.txt` not found. Run `python data/data_understanding_EDA.py` to generate it.")

if st.session_state.get("df") is None:
    with tab1:
        st.info("👈  Select a preprocessed file in the sidebar to get started.")
    with tab2:
        st.info("👈  Select a preprocessed file in the sidebar to get started.")
    with tab3:
        st.info("👈  Select a preprocessed file in the sidebar to get started.")
    st.stop()

df_main: pd.DataFrame = st.session_state.df.copy()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET EXPLORER  (generic — works with any loaded CSV)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset Explorer")

    # Detect columns dynamically
    _dc = _eda_mod._detect_cols(df_main) if _eda_mod else {}
    _dt_col  = _dc.get("dt")
    _spd_col = _dc.get("speed")
    _vol_col = _dc.get("volume")
    _id_col  = _dc.get("id")
    _rd_col  = _dc.get("road")

    # Summary metrics
    _num_cols = df_main.select_dtypes(include="number").columns.tolist()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df_main):,}")
    m2.metric("Columns", len(df_main.columns))
    if _id_col:
        m3.metric(f"Unique {_id_col}", f"{df_main[_id_col].nunique():,}")
    if _dt_col:
        try:
            _dt_parsed = pd.to_datetime(df_main[_dt_col], errors="coerce")
            m4.metric("Date range", f"{_dt_parsed.dt.date.min()} → {_dt_parsed.dt.date.max()}")
        except Exception:
            pass

    st.markdown("---")

    # ── Full data table with optional filters ─────────────────────────────
    _filter_cols = [c for c in [_spd_col, _vol_col, _rd_col] if c]
    if _filter_cols:
        filt_expander = st.expander("Filters", expanded=True)
        with filt_expander:
            fc_cols = st.columns(len(_filter_cols))
            df_filtered = df_main.copy()
            for i, col in enumerate(_filter_cols):
                with fc_cols[i]:
                    if df_main[col].dtype == object:
                        opts = sorted(df_main[col].dropna().unique())
                        sel = st.multiselect(col, opts, default=opts)
                        df_filtered = df_filtered[df_filtered[col].isin(sel)]
                    else:
                        lo, hi = float(df_main[col].min()), float(df_main[col].max())
                        sel_range = st.slider(col, lo, hi, (lo, hi))
                        df_filtered = df_filtered[df_filtered[col].between(*sel_range)]
    else:
        df_filtered = df_main.copy()

    st.caption(f"Showing {len(df_filtered):,} of {len(df_main):,} rows")
    st.dataframe(df_filtered.head(5_000), use_container_width=True, height=300)
    st.download_button(
        "⬇  Download filtered data (CSV)",
        data=df_filtered.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Numeric distributions ──────────────────────────────────────────────
    st.subheader("Numeric Distributions")
    _numeric_cols = df_main.select_dtypes(include="number").columns.tolist()
    if _numeric_cols:
        _dist_col = st.selectbox("Column to plot", _numeric_cols)
        fig_dist = px.histogram(
            df_main.sample(min(50_000, len(df_main)), random_state=42),
            x=_dist_col, nbins=60,
            title=f"Distribution of {_dist_col}",
        )
        fig_dist.update_layout(height=320)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    # ── Speed vs time (if both detected) ─────────────────────────────────
    if _spd_col and _dt_col:
        st.markdown("---")
        st.subheader(f"{_spd_col} over time (sample)")
        _sample = df_main.sample(min(10_000, len(df_main)), random_state=42).copy()
        _sample[_dt_col] = pd.to_datetime(_sample[_dt_col], errors="coerce")
        _sample = _sample.dropna(subset=[_dt_col]).sort_values(_dt_col)
        fig_ts = px.scatter(
            _sample, x=_dt_col, y=_spd_col,
            opacity=0.4, title=f"{_spd_col} over time",
        )
        fig_ts.update_traces(marker=dict(size=3))
        fig_ts.update_layout(height=320)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Road name breakdown (if detected) ────────────────────────────────
    if _rd_col:
        st.markdown("---")
        st.subheader(f"Records by {_rd_col}")
        _rd_counts = df_main[_rd_col].value_counts().reset_index()
        _rd_counts.columns = [_rd_col, "count"]
        fig_rd = px.bar(
            _rd_counts.head(30), x=_rd_col, y="count",
            title=f"Top 30 {_rd_col} by record count",
            color="count", color_continuous_scale="Blues",
        )
        fig_rd.update_layout(height=360, xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig_rd, use_container_width=True)

    # ── Speed vs Volume (if both detected) ───────────────────────────────
    if _spd_col and _vol_col:
        st.markdown("---")
        _sv_sample = df_main.sample(min(5_000, len(df_main)), random_state=42)
        fig_sv = px.scatter(
            _sv_sample, x=_vol_col, y=_spd_col,
            opacity=0.5,
            title=f"{_spd_col} vs {_vol_col} (sample)",
        )
        fig_sv.update_traces(marker=dict(size=4))
        fig_sv.update_layout(height=360)
        st.plotly_chart(fig_sv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Clustering")
    st.caption(
        "Select a feature file and clustering type, then run clustering. "
        "Speed and Volume are clustered separately. K is chosen automatically by peak silhouette score."
    )

    feature_files = cached_discover_features()  # {stem: path_str}

    if not feature_files:
        st.warning(
            "No feature parquet files found in `clustering/features/`. "
            "Run `python clustering/scripts/feature_engineering.py` first."
        )
    else:
        fc1, fc2, fc3 = st.columns([2.5, 1, 1])
        with fc1:
            sel_stem = st.selectbox("Feature file", list(feature_files.keys()))
        with fc2:
            _probe_df = pd.read_parquet(feature_files[sel_stem])
            avail_types = [
                ct for ct in ("speed", "volume")
                if _cl.feature_cols_for(_probe_df, ct)
            ]
            sel_ctype = st.selectbox("Clustering type", avail_types)
        with fc3:
            fcols_display = _cl.feature_cols_for(_probe_df, sel_ctype)
            k_override = st.number_input(
                "Override K (0 = auto)", min_value=0, max_value=10, value=0, step=1
            )

        st.caption(f"Features used: `{', '.join(fcols_display)}`")

        run_btn = st.button("▶  Run Clustering", type="primary")
        cache_key = (sel_stem, sel_ctype, int(k_override) or None)

        if run_btn or cache_key not in st.session_state.cluster_results:
            with st.spinner(f"Running {sel_ctype} clustering on {sel_stem} …"):
                result = cached_run_clustering(
                    sel_stem,
                    feature_files[sel_stem],
                    sel_ctype,
                    int(k_override) if k_override > 0 else None,
                )
            if result is None:
                st.error(f"No `{sel_ctype.upper()}` column found in this file.")
            else:
                st.session_state.cluster_results[cache_key] = result
                st.session_state.cluster_enriched[cache_key] = _enrich_with_clusters(
                    sel_stem, result.assigned_df
                )

        if cache_key in st.session_state.cluster_results:
            result = st.session_state.cluster_results[cache_key]
            adf = result.assigned_df
            cdf = result.centers_df
            enriched_df = st.session_state.cluster_enriched.get(cache_key)

            sil_str = f"{result.best_sil:.3f}" if not np.isnan(result.best_sil) else "n/a (K overridden)"
            st.success(f"K = {result.k_opt}  |  Silhouette = {sil_str}")

            # ── K-sweep diagnostics ────────────────────────────────────────
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

            # ── Cluster size bar ───────────────────────────────────────────
            st.markdown("---")
            st.subheader("Cluster Sizes")
            size_df = (
                adf["cluster_label"].value_counts().sort_index()
                .reset_index().rename(columns={"cluster_label": "Cluster", "count": "Records"})
            )
            fig_sz = px.bar(
                size_df, x="Cluster", y="Records",
                color="Records", color_continuous_scale="Blues",
                title="Records per Cluster",
            )
            fig_sz.update_layout(height=280, coloraxis_showscale=False)
            st.plotly_chart(fig_sz, use_container_width=True)

            # ── Cluster patterns — original (un-normalised) values ─────────
            st.markdown("---")
            st.subheader("Cluster Patterns — Original Values")
            if enriched_df is not None:
                _ec = _eda_mod._detect_cols(enriched_df) if _eda_mod else {}
                _orig_target = _ec.get("speed") if sel_ctype == "speed" else _ec.get("volume")
                _hour_col = next((c for c in enriched_df.columns if c.lower() == "hour"), None)
                _c_order = sorted(enriched_df["cluster_label"].unique())

                if _orig_target:
                    _esample = enriched_df.sample(min(8_000, len(enriched_df)), random_state=42).copy()
                    _esample["Cluster"] = _esample["cluster_label"].astype(str)
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        if _hour_col:
                            fig_s1 = px.scatter(
                                _esample, x=_hour_col, y=_orig_target,
                                color="Cluster", opacity=0.4,
                                title=f"Hour of Day vs {_orig_target}",
                                labels={_hour_col: "Hour (0\u201323)", _orig_target: f"{_orig_target} (original)"},
                                category_orders={"Cluster": [str(c) for c in _c_order]},
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
                            y=_orig_target,
                            color=enriched_df["cluster_label"].astype(str),
                            category_orders={"x": [str(c) for c in _c_order]},
                            title=f"{_orig_target} Distribution per Cluster",
                            labels={"x": "Cluster", _orig_target: f"{_orig_target} (original)"},
                            points=False,
                        )
                        fig_s2.update_layout(showlegend=False, height=380, plot_bgcolor="white")
                        st.plotly_chart(fig_s2, use_container_width=True)

                    # Mean target by hour x cluster heatmap
                    if _hour_col:
                        st.markdown("---")
                        _heat_pivot = (
                            enriched_df.groupby(["cluster_label", _hour_col])[_orig_target]
                            .mean().unstack("cluster_label").round(1)
                        )
                        _heat_pivot.index.name = "Hour"
                        _heat_pivot.columns = [f"Cluster {c}" for c in _heat_pivot.columns]
                        fig_heat = px.imshow(
                            _heat_pivot.T,
                            color_continuous_scale="RdYlGn",
                            title=f"Mean {_orig_target} by Hour and Cluster (original values)",
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
                    "Preprocessed data not found \u2014 cannot show original-unit charts. "
                    "Re-run clustering to rebuild."
                )

            # ── Centroids (with inverse-transformed original values) ────────
            st.markdown("---")
            st.subheader("Cluster Centroids")
            _meta_all = _load_feature_metadata()
            _csv_s = _csv_stem_from_feature_stem(sel_stem)
            _fmeta = _meta_all.get(_csv_s, {})
            cdf_display = cdf.copy()
            if "SPEED" in cdf_display.columns and "speed_min_raw" in _fmeta:
                s_rng = _fmeta["speed_max_raw"] - _fmeta["speed_min_raw"]
                cdf_display["SPEED (km/h)"] = (cdf_display["SPEED"] * s_rng + _fmeta["speed_min_raw"]).round(1)
            if "VOLUME" in cdf_display.columns and "volume_min_raw" in _fmeta:
                v_rng = _fmeta["volume_max_raw"] - _fmeta["volume_min_raw"]
                cdf_display["VOLUME (original)"] = (cdf_display["VOLUME"] * v_rng + _fmeta["volume_min_raw"]).round(1)
            st.dataframe(cdf_display, use_container_width=True, hide_index=True)

            # ── Download (enriched preprocessed data + cluster labels) ─────
            st.markdown("---")
            _dl_df = enriched_df if enriched_df is not None else adf
            _dl_label = "preprocessed data + cluster labels" if enriched_df is not None else "feature data + cluster labels"
            st.download_button(
                f"\u2b07\ufe0f  Download {_dl_label} (CSV)",
                data=_dl_df.to_csv(index=False),
                file_name=f"cluster_assignments_{sel_stem}_{sel_ctype}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SPEED BANDS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Speed Band Derivation")

    speed_results = {
        k: v for k, v in st.session_state.cluster_results.items()
        if k[1] == "speed"
    }

    if not speed_results:
        st.info("Run a **speed** clustering in the **Clustering** tab first, then return here.")
    else:
        sel_key = st.selectbox(
            "View results for",
            list(speed_results.keys()),
            format_func=lambda k: f"{k[0]}",
        )
        result = speed_results[sel_key]

        # Prefer original-unit enriched DF; fall back to normalised feature DF
        enriched_res = st.session_state.cluster_enriched.get(sel_key)
        if enriched_res is not None:
            res_df = enriched_res
            _ec_res = _eda_mod._detect_cols(res_df) if _eda_mod else {}
            speed_col = _ec_res.get("speed") or "SPEED"
            spd_unit = "km/h"
        else:
            res_df = result.assigned_df
            col_map_res = {c.lower(): c for c in res_df.columns}
            speed_col = col_map_res.get("speed", "SPEED")
            spd_unit = "normalised [0-1]"
            st.info(
                "Preprocessed data not available - showing normalised speed values. "
                "Re-run clustering to see original km/h values."
            )

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

        # Speed Band range chart
        st.markdown("---")
        st.subheader("Speed Band Ranges per Cluster")
        st.caption(
            f"Each bar spans **P10 to P90** of the cluster's speed distribution ({spd_unit}).  "
            "The diamond marker shows the mean. Narrow bars = tight, well-defined regime."
        )

        colors = px.colors.qualitative.Safe
        fig_bands = go.Figure()
        for i, row in bands.iterrows():
            c = colors[i % len(colors)]
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
                showlegend=i == 0, name="Mean speed",
                hoverinfo="skip",
            ))

        fig_bands.update_layout(
            title=f"Speed Bands - {sel_key[0]}",
            xaxis=dict(title=f"Speed ({spd_unit})", gridcolor="#EEEEEE"),
            yaxis=dict(title="Cluster", categoryorder="array", categoryarray=ordered),
            barmode="overlay",
            height=max(320, 52 * len(bands) + 80),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_bands, use_container_width=True)

        # Cluster deep-dive
        st.markdown("---")
        st.subheader("Cluster Deep-Dive")
        sel_cluster = st.selectbox("Select cluster", bands["cluster_label"].tolist())
        cluster_speeds = res_df[res_df["cluster_label"] == sel_cluster][speed_col]
        p10 = cluster_speeds.quantile(0.10)
        p90 = cluster_speeds.quantile(0.90)
        mean_spd = cluster_speeds.mean()

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Cluster Size",     f"{len(cluster_speeds):,}")
        mc2.metric("Lower Band (P10)", f"{p10:.1f} {spd_unit}")
        mc3.metric("Upper Band (P90)", f"{p90:.1f} {spd_unit}")
        mc4.metric("Band Width",        f"{p90 - p10:.1f} {spd_unit}")

        _spd_margin = max(1.0, (cluster_speeds.max() - cluster_speeds.min()) * 0.02)
        bins = np.linspace(
            max(0, cluster_speeds.min() - _spd_margin),
            cluster_speeds.max() + _spd_margin, 55
        )
        counts_h, edges = np.histogram(cluster_speeds, bins=bins)
        bar_w = np.diff(edges)

        fig_hist = go.Figure()
        for mask, color, label in [
            (edges[:-1] < p10,                              "rgba(180,180,180,0.55)", "Below P10"),
            ((edges[:-1] >= p10) & (edges[:-1] < p90),     "rgba(56,169,70,0.75)",   "Speed Band P10 to P90"),
            (edges[:-1] >= p90,                             "rgba(180,180,180,0.55)", "Above P90"),
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
            fig_hist.add_vline(x=val, line_color=color, line_width=2, line_dash=dash,
                               annotation_text=f"  {text}",
                               annotation_font=dict(color=color, size=11))

        fig_hist.update_layout(
            title=f"Speed Distribution - Cluster {int(sel_cluster)}",
            xaxis=dict(title=f"Speed ({spd_unit})", gridcolor="#EEEEEE"),
            yaxis=dict(title="Count", gridcolor="#EEEEEE"),
            barmode="stack", bargap=0.02, height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Box plots across all clusters
        st.markdown("---")
        st.subheader("Speed Distribution across All Clusters")
        fig_box = px.box(
            res_df, x=res_df["cluster_label"].astype(str), y=speed_col,
            color=res_df["cluster_label"].astype(str),
            category_orders={"x": ordered},
            title=f"Speed Box Plots - {sel_key[0]}",
            labels={"x": "Cluster", speed_col: f"Speed ({spd_unit})"},
            points=False,
        )
        fig_box.update_layout(showlegend=False, height=380, plot_bgcolor="white")
        st.plotly_chart(fig_box, use_container_width=True)

        # Band summary + download
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
            "Download with bands (CSV)",
            data=export_df.to_csv(index=False),
            file_name=f"speed_bands_{sel_key[0]}.csv",
            mime="text/csv",
        )

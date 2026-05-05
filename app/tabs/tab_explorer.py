"""
Tab: Dataset Explorer
Loads a preprocessed CSV (via selector or upload) and provides a generic viewer.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from .context import AppContext, discover_preproc_files


def render(ctx: AppContext) -> None:
    st.header("Data Explorer")

    # ── File loading ──────────────────────────────────────────────────────────
    preproc_files = discover_preproc_files(ctx)

    col_sel, col_up = st.columns([1, 1])
    with col_sel:
        if preproc_files:
            preproc_sel = st.selectbox(
                "Select preprocessed file",
                ["(none)"] + list(preproc_files.keys()),
                key="explorer_preproc_sel",
            )
            if preproc_sel != "(none)":
                ppath = preproc_files[preproc_sel]
                if (
                    st.session_state.get("_explorer_loaded") != ppath
                    or st.session_state.df is None
                ):
                    st.session_state.df = pd.read_csv(ppath)
                    st.session_state["_explorer_loaded"] = ppath
                st.success(f"Loaded {len(st.session_state.df):,} rows")
        else:
            st.info("No preprocessed files found on disk.")

    with col_up:
        uploaded = st.file_uploader("Or upload any CSV", type=["csv"], key="explorer_upload")
        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                st.session_state.df = df_upload
                st.session_state["_explorer_loaded"] = None
                st.success(f"Loaded {len(df_upload):,} rows  ·  {len(df_upload.columns)} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    st.markdown("---")

    if st.session_state.df is None:
        st.info("Select a preprocessed file or upload a CSV above to get started.")
        return

    df_main: pd.DataFrame = st.session_state.df.copy()

    # Detect columns dynamically
    dc       = ctx.eda_mod._detect_cols(df_main) if ctx.eda_mod else {}
    dt_col   = dc.get("dt")
    spd_col  = dc.get("speed")
    vol_col  = dc.get("volume")
    id_col   = dc.get("id")
    rd_col   = dc.get("road")

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows",    f"{len(df_main):,}")
    m2.metric("Columns", len(df_main.columns))
    if id_col:
        m3.metric(f"Unique {id_col}", f"{df_main[id_col].nunique():,}")
    if dt_col:
        try:
            dt_parsed = pd.to_datetime(df_main[dt_col], errors="coerce")
            m4.metric(
                "Date range",
                f"{dt_parsed.dt.date.min()} → {dt_parsed.dt.date.max()}",
            )
        except Exception:
            pass

    st.markdown("---")

    # ── Filterable data table ─────────────────────────────────────────────────
    filter_cols = [c for c in [spd_col, vol_col, rd_col] if c]
    if filter_cols:
        with st.expander("Filters", expanded=True):
            fc_cols    = st.columns(len(filter_cols))
            df_filtered = df_main.copy()
            for i, col in enumerate(filter_cols):
                with fc_cols[i]:
                    if df_main[col].dtype == object:
                        opts = sorted(df_main[col].dropna().unique())
                        sel  = st.multiselect(col, opts, default=opts)
                        df_filtered = df_filtered[df_filtered[col].isin(sel)]
                    else:
                        lo, hi    = float(df_main[col].min()), float(df_main[col].max())
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

    # ── Numeric distributions ─────────────────────────────────────────────────
    st.subheader("Numeric Distributions")
    numeric_cols = df_main.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        dist_col = st.selectbox("Column to plot", numeric_cols)
        fig_dist = px.histogram(
            df_main.sample(min(50_000, len(df_main)), random_state=42),
            x=dist_col, nbins=60,
            title=f"Distribution of {dist_col}",
        )
        fig_dist.update_layout(height=320)
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    # ── Speed over time ───────────────────────────────────────────────────────
    if spd_col and dt_col:
        st.markdown("---")
        st.subheader(f"{spd_col} over time (sample)")
        sample = df_main.sample(min(10_000, len(df_main)), random_state=42).copy()
        sample[dt_col] = pd.to_datetime(sample[dt_col], errors="coerce")
        sample = sample.dropna(subset=[dt_col]).sort_values(dt_col)
        fig_ts = px.scatter(
            sample, x=dt_col, y=spd_col,
            opacity=0.4, title=f"{spd_col} over time",
        )
        fig_ts.update_traces(marker=dict(size=3))
        fig_ts.update_layout(height=320)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Road name breakdown ───────────────────────────────────────────────────
    if rd_col:
        st.markdown("---")
        st.subheader(f"Records by {rd_col}")
        rd_counts = df_main[rd_col].value_counts().reset_index()
        rd_counts.columns = [rd_col, "count"]
        fig_rd = px.bar(
            rd_counts.head(30), x=rd_col, y="count",
            title=f"Top 30 {rd_col} by record count",
            color="count", color_continuous_scale="Blues",
        )
        fig_rd.update_layout(height=360, xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig_rd, use_container_width=True)

    # ── Speed vs Volume scatter ───────────────────────────────────────────────
    if spd_col and vol_col:
        st.markdown("---")
        sv_sample = df_main.sample(min(5_000, len(df_main)), random_state=42)
        fig_sv = px.scatter(
            sv_sample, x=vol_col, y=spd_col,
            opacity=0.5,
            title=f"{spd_col} vs {vol_col} (sample)",
        )
        fig_sv.update_traces(marker=dict(size=4))
        fig_sv.update_layout(height=360)
        st.plotly_chart(fig_sv, use_container_width=True)

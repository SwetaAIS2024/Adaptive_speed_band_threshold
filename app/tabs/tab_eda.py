"""
Tab: Raw Dataset EDA
Renders the exploratory analysis tab.
"""

import streamlit as st
import plotly.express as px

from .context import AppContext


@st.cache_data(show_spinner=False)
def _load_raw_cached(path_str: str, _eda_mod) -> object:
    return _eda_mod.load_any_file(path_str)


def render(ctx: AppContext) -> None:
    st.header("Data Quality & EDA")
    st.caption(
        "Exploratory analysis of source files in `data/raw_dataset/` or "
        "`data/pre_processed_dataset/`. Preprocessed files include derived columns: "
        "`hour`, `date`, and `day_type` (Weekday / Weekend from filename). "
        "Select a source and a file to inspect."
    )

    if ctx.eda_import_error:
        st.error(
            f"Could not load EDA module from `{ctx.data_root / 'data_understanding_EDA.py'}`.\n\n"
            f"Error: `{ctx.eda_import_error}`"
        )
        return

    if ctx.eda_mod is None:
        st.error("EDA module failed to load for an unknown reason.")
        return

    source_type = st.radio(
        "Dataset source",
        list(ctx.dataset_dirs.keys()),
        horizontal=True,
    )
    active_dir = ctx.dataset_dirs[source_type]
    raw_files  = ctx.eda_mod.discover_raw_files(active_dir)

    if not raw_files:
        st.warning(
            f"No CSV or Excel files found in `{active_dir}`. "
            "Run the relevant script to populate this folder."
        )
        return

    sel_raw = st.selectbox("Select dataset", list(raw_files.keys()))
    raw_df  = _load_raw_cached(raw_files[sel_raw], ctx.eda_mod)

    # ── Preprocess button (raw datasets only) ─────────────────────────────────
    if source_type == "Raw datasets" and ctx.preproc_mod is not None:
        st.markdown("---")
        st.markdown(
            "**Preprocess this file** — cleans the raw CSV and saves it to "
            "`data/pre_processed_dataset/` (adds `hour`, `date`, `day_type`; "
            "drops nulls, duplicates, zero-speed rows; normalises `ROAD_NAME` to uppercase)."
        )
        if st.button("⚙  Run Preprocessing on all raw files", key="eda_run_preproc"):
            with st.spinner("Preprocessing raw files …"):
                try:
                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        ctx.preproc_mod.run_preprocessing()
                    st.success("Preprocessing complete. Preprocessed files are ready in `data/pre_processed_dataset/`.")
                    log_text = buf.getvalue()
                    if log_text:
                        with st.expander("Preprocessing log"):
                            st.text(log_text)
                except SystemExit:
                    st.error("No raw files found in `data/raw_dataset/`. Add raw CSV files and retry.")
                except Exception as exc:
                    st.error(f"Preprocessing failed: {exc}")
        st.markdown("---")
    st.markdown("---")
    st.subheader("1. Feature Exploration")
    info = ctx.eda_mod.get_feature_info(raw_df)
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows",         f"{info['shape'][0]:,}")
    m2.metric("Columns",      info["shape"][1])
    m3.metric("Column Names", ", ".join(info["columns"]))
    st.markdown("**Data Types**")
    st.dataframe(info["dtypes"], use_container_width=True, hide_index=True)
    st.markdown("**First 10 rows**")
    st.dataframe(info["head"],  use_container_width=True, hide_index=True)

    # ── 2. Data Quality ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2. Data Quality Checks")
    quality = ctx.eda_mod.get_quality_checks(raw_df)

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Null / Missing Values**")
        st.dataframe(quality["null_df"], use_container_width=True, hide_index=True)
    with q2:
        st.markdown("**Duplicate Rows**")
        st.metric(
            "Total Duplicates", f"{quality['dup_count']:,}",
            delta=f"{quality['dup_pct']}% of rows", delta_color="inverse",
        )
        if not quality["neg_zero"].empty:
            st.markdown("**Negative / Zero Values (Numeric)**")
            st.dataframe(quality["neg_zero"], use_container_width=True, hide_index=True)

    if quality["speed_col"] and quality["speed_anomalies"] is not None:
        anom = quality["speed_anomalies"]
        st.markdown(
            f"**Speed Anomalies (`{quality['speed_col']}` < 0 or > 200):** {len(anom):,} rows"
        )
        if len(anom) > 0:
            st.dataframe(anom.head(50), use_container_width=True, hide_index=True)

    if quality["date_range"]:
        dr = quality["date_range"]
        st.markdown(
            f"**Date Range (`{dr['col']}`):** `{dr['min']}` → `{dr['max']}`"
            f"  |  Unparseable: `{dr['unparseable']}`"
        )

    # ── 3. Categorical Analysis ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Categorical Feature Analysis")
    cats = ctx.eda_mod.get_category_analysis(raw_df)

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
                    fig_cat = px.bar(
                        vc.head(20), x=col, y="Count",
                        title=f"Top 20 — {col}",
                    )
                    fig_cat.update_layout(height=320, xaxis_tickangle=-30)
                    st.plotly_chart(fig_cat, use_container_width=True)

    # ── 4. Statistics & Temporal Analysis ────────────────────────────────────
    st.markdown("---")
    st.subheader("4. General Statistics & Temporal Analysis")
    stats = ctx.eda_mod.get_statistics(raw_df)

    if stats["describe"] is not None:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(stats["describe"], use_container_width=True)

    t1, t2 = st.columns(2)
    with t1:
        if stats["hourly_speed"] is not None:
            hs = stats["hourly_speed"]
            fig_hs = px.line(
                hs, x="Hour", y=hs.columns[1], markers=True,
                title=f"Mean {stats['speed_col']} by Hour",
            )
            fig_hs.update_layout(height=320)
            st.plotly_chart(fig_hs, use_container_width=True)
    with t2:
        if stats["hourly_volume"] is not None:
            hv = stats["hourly_volume"]
            fig_hv = px.line(
                hv, x="Hour", y=hv.columns[1], markers=True,
                title=f"Mean {stats['volume_col']} by Hour",
                color_discrete_sequence=["#DD8452"],
            )
            fig_hv.update_layout(height=320)
            st.plotly_chart(fig_hv, use_container_width=True)
        elif stats["road_speed"] is not None:
            rs = stats["road_speed"]
            fig_rs = px.bar(
                rs, x=stats["road_col"], y=rs.columns[1],
                title=f"Mean {stats['speed_col']} by {stats['road_col']}",
                color=rs.columns[1], color_continuous_scale="RdYlGn",
            )
            fig_rs.update_layout(height=320, xaxis_tickangle=-30)
            st.plotly_chart(fig_rs, use_container_width=True)

    if stats["id_summary"] is not None:
        st.markdown(f"**Top 20 `{stats['id_col']}` by record count**")
        st.dataframe(stats["id_summary"], use_container_width=True, hide_index=True)

    # ── 5. Cross-Dataset Comparison ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("5. Cross-Dataset Comparison")
    cross_df = ctx.eda_mod.get_cross_comparison(
        {k.split("  [")[0]: v for k, v in raw_files.items()}
    )
    st.dataframe(cross_df, use_container_width=True, hide_index=True)

    # ── Full EDA Report ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Full EDA Report (Text)")

    if st.button("📄  Generate / Refresh EDA Report", key="eda_gen_report"):
        with st.spinner("Running EDA on raw files …"):
            try:
                out_path = ctx.eda_mod.generate_eda_report()
                st.success(f"EDA report saved to `{out_path}`.")
                st.rerun()
            except Exception as exc:
                st.error(f"EDA report generation failed: {exc}")

    if ctx.eda_report_path.exists():
        report_text = ctx.eda_report_path.read_text(encoding="utf-8")
        st.download_button(
            "⬇  Download full EDA report",
            data=report_text,
            file_name="eda_report.txt",
            mime="text/plain",
        )
        with st.expander("View raw EDA report text"):
            st.code(report_text, language=None)
    else:
        st.info(
            "`eda_report.txt` not yet generated. "
            "Click **Generate / Refresh EDA Report** above to create it."
        )

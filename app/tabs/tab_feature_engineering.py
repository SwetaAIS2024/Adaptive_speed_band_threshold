"""
Tab: Feature Engineering
Runs feature_engineering.py on all preprocessed CSVs and shows what was produced.
Must be run before the Clustering tab can work.
"""

import io
import contextlib

import pandas as pd
import streamlit as st

from .context import AppContext, get_pipeline_state


def render(ctx: AppContext) -> None:
    st.header("Feature Engineering")

    ps = get_pipeline_state(ctx)
    if not ps["preproc_done"]:
        st.warning(
            "⚠️ **Step 1 not complete** — no preprocessed files found in "
            "`data/pre_processed_dataset/`.  "
            "Go to the **Data Quality & EDA** tab and run preprocessing first."
        )
        return
    st.caption(
        "Converts preprocessed CSVs in `data/pre_processed_dataset/` into "
        "parquet feature files in `clustering/features/`. "
        "This step is required before running clustering."
    )

    if ctx.feat_eng_mod is None:
        st.error("Feature engineering module could not be loaded.")
        return

    # ── Existing feature files ─────────────────────────────────────────────────
    features_dir = ctx.data_root.parent / "clustering" / "features"
    existing = sorted(features_dir.glob("*.parquet")) if features_dir.exists() else []

    if existing:
        st.success(f"{len(existing)} feature file(s) ready in `clustering/features/`.")
        with st.expander("View existing feature files"):
            rows = []
            for p in existing:
                try:
                    df = pd.read_parquet(p)
                    rows.append({"File": p.name, "Rows": f"{len(df):,}", "Columns": ", ".join(df.columns)})
                except Exception:
                    rows.append({"File": p.name, "Rows": "error", "Columns": ""})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No feature files found. Run feature engineering below to generate them.")

    st.markdown("---")

    # ── Check preprocessed files exist ────────────────────────────────────────
    preproc_dir = ctx.data_root / "pre_processed_dataset"
    skip = ctx.skip_files
    preproc_files = [
        p for p in preproc_dir.glob("*.csv") if p.name not in skip
    ] if preproc_dir.exists() else []

    if not preproc_files:
        st.error(
            "No preprocessed CSVs found in `data/pre_processed_dataset/`. "
            "Go to the **Raw Data EDA** tab and run preprocessing first."
        )
        return

    st.markdown(f"**{len(preproc_files)} preprocessed file(s) found** — will be converted:")
    for p in preproc_files:
        st.markdown(f"- `{p.name}`")

    st.markdown("---")

    if st.button("▶  Run Feature Engineering", type="primary", key="feat_eng_run_btn"):
        log_placeholder = st.empty()
        with st.spinner("Running feature engineering on all preprocessed files …"):
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    ctx.feat_eng_mod.main()
                log_text = buf.getvalue()

                st.success(
                    "Feature engineering complete! "
                    "Switch to the **Clustering** tab to run K-Means."
                )
                if log_text:
                    with st.expander("Feature engineering log", expanded=True):
                        st.text(log_text)

                # Refresh the file list
                existing = sorted(features_dir.glob("*.parquet")) if features_dir.exists() else []
                if existing:
                    st.markdown(f"**{len(existing)} parquet file(s) written:**")
                    for p in existing:
                        st.markdown(f"- `{p.name}`")

            except SystemExit:
                st.error(
                    "No preprocessed files found. "
                    "Run preprocessing first from the **Raw Data EDA** tab."
                )
            except Exception as exc:
                st.error(f"Feature engineering failed: {exc}")

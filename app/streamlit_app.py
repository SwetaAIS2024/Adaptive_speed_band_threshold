"""
Adaptive Speed Band Threshold — Interactive Dashboard
======================================================
Entry point. Each tab is implemented in app/tabs/tab_*.py.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import warnings
from pathlib import Path

import streamlit as st

warnings.filterwarnings("ignore")

# ── Make tabs package importable ──────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from tabs.context import AppContext, build_context, get_pipeline_state
from tabs import (
    tab_eda,
    tab_explorer,
    tab_feature_engineering,
    tab_clustering,
    tab_speed_bands,
    tab_volume_bands,
    tab_threshold_viz,
    tab_volume_threshold_viz,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Speed Band Threshold",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Build shared context once ─────────────────────────────────────────────────
@st.cache_resource
def _get_context() -> AppContext:
    return build_context()

ctx = _get_context()

# ── Session state ─────────────────────────────────────────────────────────────
for _key, _default in [
    ("df",               None),
    ("cluster_results",  {}),
    ("cluster_enriched", {}),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Sidebar — branding + live workflow progress ───────────────────────────────
with st.sidebar:
    st.title("🚗  Adaptive Traffic Volume and Speed Congestion Thresholds Computation")
    st.caption("Data-driven adaptive traffic volume and speed threshold bands computation using K-Means clustering, with an interactive Streamlit dashboard for exploration and visualization.")
    st.markdown("---")

    @st.fragment(run_every=15)
    def _workflow_progress():
        ps = get_pipeline_state(ctx)

        def _icon(done: bool, prereq: bool) -> str:
            if done:   return "✅"
            if prereq: return "▶️"
            return "🔒"

        st.markdown("**Workflow Steps**")
        st.markdown(
            f"{_icon(ps['preproc_done'], True)}  "
            f"**1. Preprocess raw files**"
            + (f"  `{ps['preproc_count']} file(s)`" if ps['preproc_done'] else "  *(not done)*")
        )
        st.markdown(
            f"{_icon(ps['features_done'], ps['preproc_done'])}  "
            f"**2. Feature engineering**"
            + (f"  `{ps['features_count']} parquet(s)`" if ps['features_done'] else "  *(not done)*")
        )
        st.markdown(
            f"{_icon(ps['clustering_done'], ps['features_done'])}  "
            f"**3. Run clustering**"
            + (f"  `{ps['cluster_count']} result(s)`" if ps['clustering_done'] else "  *(not done)*")
        )
        st.markdown(
            f"{_icon(ps['clustering_done'], ps['clustering_done'])}  "
            f"**4. Review speed / volume bands**"
            + ("" if ps['clustering_done'] else "  *(needs step 3)*")
        )
        st.markdown(
            f"{_icon(ps['preproc_done'], ps['preproc_done'])}  "
            f"**5. Threshold visualisation**"
            + ("" if ps['preproc_done'] else "  *(needs step 1)*")
        )
        st.markdown("---")
        st.caption("Icons: ✅ complete · ▶️ ready to run · 🔒 prerequisites not met")

    _workflow_progress()

# ── Page layout ──────────────────────────────────────────────────────────────
PAGES = {
    "🔍  Data Quality & EDA": tab_eda.render,
    "📋  Data Explorer": tab_explorer.render,
    "⚙️  Feature Engineering": tab_feature_engineering.render,
    "🔮  Clustering": tab_clustering.render,
    "🎯  Speed Bands": tab_speed_bands.render,
    "📦  Volume Bands": tab_volume_bands.render,
    "📈  Speed Threshold Visualisation": tab_threshold_viz.render,
    "📊  Volume Threshold Visualisation": tab_volume_threshold_viz.render,
}

page = st.radio(
    "Navigation",
    list(PAGES.keys()),
    horizontal=True,
    label_visibility="collapsed",
    key="active_page",
)

PAGES[page](ctx)

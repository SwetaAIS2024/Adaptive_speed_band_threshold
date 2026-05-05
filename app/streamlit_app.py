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

from tabs.context import AppContext, build_context
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

# ── Sidebar — branding only ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗  Adaptive Traffic Volume and Speed Congestion Thresholds Computation")
    st.caption("Data-driven adaptive traffic volume and speed threshold bands computation using K-Means clustering, with an interactive Streamlit dashboard for exploration and visualization.")
    st.markdown("---")
    st.markdown(
        "**Tabs**\n"
        "- 🔍 **Data Quality & EDA** — explore & preprocess source files\n"
        "- 📋 **Data Explorer** — interactive filter, distributions, upload any CSV\n"
        "- ⚙️ **Feature Engineering** — encode & normalise features for clustering\n"
        "- 🔮 **Clustering** — run K-Means on speed / volume features\n"
        "- 🎯 **Speed Bands** — P10/P90 speed bands from speed clustering\n"
        "- 📦 **Volume Bands** — P10/P90 volume bands from volume clustering\n"
        "- 📈 **Speed Threshold Visualisation** — speed vs hour with adaptive P10 threshold\n"
        "- 📊 **Volume Threshold Visualisation** — volume vs hour with adaptive P10 threshold"
    )

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_eda_ui, tab_explorer_ui, tab_feat_eng_ui, tab_clustering_ui, \
tab_speed_bands_ui, tab_vol_bands_ui, tab_thresh_ui, tab_vol_thresh_ui = st.tabs([
    "🔍  Data Quality & EDA",
    "📋  Data Explorer",
    "⚙️  Feature Engineering",
    "🔮  Clustering",
    "🎯  Speed Bands",
    "📦  Volume Bands",
    "📈  Speed Threshold Visualisation",
    "📊  Volume Threshold Visualisation",
])

with tab_eda_ui:
    tab_eda.render(ctx)

with tab_explorer_ui:
    tab_explorer.render(ctx)

with tab_feat_eng_ui:
    tab_feature_engineering.render(ctx)

with tab_clustering_ui:
    tab_clustering.render(ctx)

with tab_speed_bands_ui:
    tab_speed_bands.render(ctx)

with tab_vol_bands_ui:
    tab_volume_bands.render(ctx)

with tab_thresh_ui:
    tab_threshold_viz.render(ctx)

with tab_vol_thresh_ui:
    tab_volume_threshold_viz.render(ctx)

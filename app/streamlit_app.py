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
from tabs import tab_eda, tab_explorer, tab_feature_engineering, tab_clustering, tab_speed_bands, tab_threshold_viz

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
    st.title("🚗 Speed Band Threshold")
    st.caption("Data-driven adaptive speed bands for Singapore roads")
    st.markdown("---")
    st.markdown(
        "**Tabs**\n"
        "- 🔍 **Data Quality & EDA** — explore & preprocess source files\n"
        "- 📋 **Data Explorer** — interactive filter, distributions, upload any CSV\n"
        "- ⚙️ **Feature Engineering** — encode & normalise features for clustering\n"
        "- 🔮 **Clustering** — run K-Means on speed / volume features\n"
        "- 🎯 **Speed & Volume Bands** — derive P10/P90 bands from cluster results\n"
        "- 📈 **Threshold Visualisation** — speed vs hour with adaptive P10 threshold line"
    )

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_eda_ui, tab_explorer_ui, tab_feat_eng_ui, tab_clustering_ui, tab_bands_ui, tab_thresh_ui = st.tabs([
    "🔍  Data Quality & EDA",
    "📋  Data Explorer",
    "⚙️  Feature Engineering",
    "🔮  Clustering",
    "🎯  Speed & Volume Bands",
    "📈  Threshold Visualisation",
])

with tab_eda_ui:
    tab_eda.render(ctx)

with tab_explorer_ui:
    tab_explorer.render(ctx)

with tab_feat_eng_ui:
    tab_feature_engineering.render(ctx)

with tab_clustering_ui:
    tab_clustering.render(ctx)

with tab_bands_ui:
    tab_speed_bands.render(ctx)

with tab_thresh_ui:
    tab_threshold_viz.render(ctx)

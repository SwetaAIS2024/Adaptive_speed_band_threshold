"""
Shared context for all tab modules.

Each tab module receives a `ctx` object (an instance of AppContext) that holds
all module-level singletons — EDA module, clustering module, paths, helpers —
so tab files never need to do their own importlib loading.

Usage in tab files:
    from app.tabs.context import AppContext
    def render(ctx: AppContext, ...): ...
"""

import importlib.util as _ilu
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Repo-relative paths ───────────────────────────────────────────────────────
_APP_DIR  = Path(__file__).resolve().parent.parent        # app/
_REPO_ROOT = _APP_DIR.parent                              # repo root
DATA_ROOT  = _REPO_ROOT / "data"
EDA_MODULE_PATH      = DATA_ROOT / "data_understanding_EDA.py"
EDA_REPORT_PATH      = DATA_ROOT / "pre_processed_dataset" / "eda_report.txt"
PREPROCESS_MODULE_PATH = DATA_ROOT / "preprocess.py"
CLUSTERING_MODULE_PATH = (
    _REPO_ROOT / "clustering" / "scripts" / "run_clustering.py"
)
FEATURE_ENG_MODULE_PATH = (
    _REPO_ROOT / "clustering" / "scripts" / "feature_engineering.py"
)

SKIP_FILES = {"preprocessing_summary.csv", "eda_report.txt"}

DATASET_DIRS = {
    "Raw datasets":          str(DATA_ROOT / "raw_dataset"),
    "Preprocessed datasets": str(DATA_ROOT / "pre_processed_dataset"),
}


# ── Helper: load a Python module from an absolute path ───────────────────────
def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, str(path))
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Shared context dataclass ──────────────────────────────────────────────────
@dataclass
class AppContext:
    """
    Holds all singleton objects shared across tab modules.
    Instantiated once in streamlit_app.py and passed to every render() call.
    """
    # EDA module (None if it failed to load)
    eda_mod:           object = None
    eda_import_error:  Optional[str] = None

    # Clustering module
    cl_mod:            object = None

    # Preprocessing module
    preproc_mod:       object = None

    # Feature engineering module
    feat_eng_mod:      object = None

    # Convenience paths (same objects as module-level constants above)
    data_root:         Path = field(default_factory=lambda: DATA_ROOT)
    eda_report_path:   Path = field(default_factory=lambda: EDA_REPORT_PATH)
    dataset_dirs:      dict = field(default_factory=lambda: dict(DATASET_DIRS))
    skip_files:        set  = field(default_factory=lambda: set(SKIP_FILES))


def build_context() -> AppContext:
    """Load modules and return a fully populated AppContext."""
    ctx = AppContext()

    # Load EDA module
    try:
        ctx.eda_mod = _load_module("data_understanding_EDA", EDA_MODULE_PATH)
    except Exception as e:
        ctx.eda_import_error = str(e)

    # Load clustering module
    try:
        ctx.cl_mod = _load_module("run_clustering", CLUSTERING_MODULE_PATH)
    except Exception as e:
        st.error(f"Failed to load clustering module: {e}")

    # Load preprocessing module
    try:
        ctx.preproc_mod = _load_module("preprocess", PREPROCESS_MODULE_PATH)
    except Exception as e:
        st.warning(f"Could not load preprocessing module: {e}")

    # Load feature engineering module
    try:
        ctx.feat_eng_mod = _load_module("feature_engineering", FEATURE_ENG_MODULE_PATH)
    except Exception as e:
        st.warning(f"Could not load feature engineering module: {e}")

    return ctx


# ── Shared helper functions (imported by tab files) ───────────────────────────

def csv_stem_from_feature_stem(feature_stem: str) -> str:
    """Strip 'features_' prefix to get the preprocessed CSV stem."""
    return (
        feature_stem[len("features_"):]
        if feature_stem.startswith("features_")
        else feature_stem
    )


@st.cache_data(show_spinner=False)
def load_feature_metadata(features_dir_str: str) -> dict:
    """Load feature_metadata.json — raw min/max for inverse-transforming centroids."""
    meta_path = Path(features_dir_str) / "feature_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def enrich_with_clusters(
    feature_stem: str,
    assigned_df: pd.DataFrame,
    data_root: Path,
) -> Optional[pd.DataFrame]:
    """
    Load the preprocessed CSV for feature_stem and append cluster_label
    by row-index alignment (feature engineering preserves row order).
    Returns None if the CSV is missing or row counts differ.
    """
    csv_stem = csv_stem_from_feature_stem(feature_stem)
    csv_path = data_root / "pre_processed_dataset" / f"{csv_stem}.csv"
    if not csv_path.exists():
        return None
    preproc = pd.read_csv(str(csv_path))
    if len(preproc) != len(assigned_df):
        return None
    preproc = preproc.reset_index(drop=True)
    preproc["cluster_label"] = assigned_df["cluster_label"].values
    return preproc


def enriched_result_path(feature_stem: str, clustering_type: str, data_root: Path) -> Path:
    """Path for persisted enriched clustering output."""
    results_dir = data_root.parent / "clustering" / "results"
    return results_dir / f"enriched_{feature_stem}_{clustering_type}.parquet"


@st.cache_data(show_spinner=False)
def load_enriched_result(path_str: str) -> Optional[pd.DataFrame]:
    """Load a persisted enriched clustering output if present."""
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def persist_enriched_result(
    enriched_df: pd.DataFrame,
    feature_stem: str,
    clustering_type: str,
    data_root: Path,
) -> Path:
    """Persist enriched clustering output as parquet for fast reloads."""
    path = enriched_result_path(feature_stem, clustering_type, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_parquet(path, index=False)
    load_enriched_result.clear()
    return path


def discover_preproc_files(ctx: AppContext) -> dict:
    """Return {display_name: abs_path_str} for preprocessed CSVs, skipping summary files."""
    if ctx.eda_mod is None:
        return {}
    all_files = ctx.eda_mod.discover_raw_files(
        str(ctx.data_root / "pre_processed_dataset")
    )
    return {
        name: path
        for name, path in all_files.items()
        if os.path.basename(path) not in ctx.skip_files
    }


# ── Pipeline state helper ─────────────────────────────────────────────────────

def get_pipeline_state(ctx: AppContext) -> dict:
    """
    Scan the filesystem and return a dict describing which pipeline steps
    are complete. This is intentionally cheap (glob only, no file reads).

    Keys
    ----
    preproc_done   : bool  — at least one *_cleaned.csv in pre_processed_dataset/
    features_done  : bool  — at least one *.parquet in clustering/features/
    clustering_done: bool  — at least one cluster_assignments_*.parquet in clustering/results/
    preproc_count  : int   — number of preprocessed CSVs found
    features_count : int   — number of feature parquets found
    cluster_count  : int   — number of assignment parquets found
    """
    preproc_dir  = ctx.data_root / "pre_processed_dataset"
    features_dir = ctx.data_root.parent / "clustering" / "features"
    results_dir  = ctx.data_root.parent / "clustering" / "results"

    preproc_files  = [
        f for f in preproc_dir.glob("*.csv")
        if f.name not in ctx.skip_files and f.name != "preprocessing_summary.csv"
    ]
    feature_files  = list(features_dir.glob("*.parquet"))
    cluster_files  = list(results_dir.glob("cluster_assignments_*.parquet"))

    # Also count results that exist only in the current session (not yet on disk)
    session_cluster_count = len(st.session_state.get("cluster_results", {}))

    total_cluster_count = len(cluster_files) + session_cluster_count

    return {
        "preproc_done":    len(preproc_files) > 0,
        "features_done":   len(feature_files) > 0,
        "clustering_done": total_cluster_count > 0,
        "preproc_count":   len(preproc_files),
        "features_count":  len(feature_files),
        "cluster_count":   total_cluster_count,
    }

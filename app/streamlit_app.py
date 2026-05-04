"""
Adaptive Speed Band Threshold — Interactive Dashboard
======================================================
Visualise input traffic data, run K-Means clustering per road category / day
type, and explore how P10/P90 speed bands are derived from each cluster.

Compatible with ANY dataset that matches the 7-column schema below:
    timestamp_hour, LinkID, RoadCategory, hour, day_type, speed, volume

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import warnings
from pathlib import Path

# Repo root on sys.path so clustering.pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clustering.pipeline import (
    VALID_CATEGORIES, CATEGORY_PREFIX, DAY_PREFIX, REQUIRED_COLS, FEATURE_COLS,
    engineer_subset, sweep_k, pick_optimal_k, run_kmeans,
    merge_small_clusters, build_cluster_id,
    extract_bands,
)

warnings.filterwarnings("ignore")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Speed Band Threshold",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── App-only constants ───────────────────────────────────────────────────────
REQUIRED_COLS_WITH_TS = REQUIRED_COLS | {"timestamp_hour"}

CATEGORY_LABEL = {
    1: "Expressway", 2: "Major Arterial", 3: "Arterial",
    4: "Minor Arterial", 5: "Local Access", 6: "Minor Access",
}

EXPRESSWAY = {"E1": "BKE", "E2": "KJE", "E3": "SLE", "E4": "TPE", "E5": "AYE", "E6": "CTE", "E7": "ECP", "E8": "PIE"} 

ROAD_DIRECTION = {"D1": "INBOUND", "D2": "OUTBOUND"}

SOURCES = {"S1": "GPS", "S2": "Sensor", "S3": "Other"}

# Actual column names for the optional future-dataset fields
OPTIONAL_COLS = ["expressway", "road_direction", "source"]

# Max rows passed to K-sweep to keep the UI responsive on large datasets
K_SWEEP_SAMPLE = 50_000
SCATTER_SAMPLE = 8_000

# ─── Helpers ─────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> list[str]:
    missing = (REQUIRED_COLS | {"timestamp_hour"}) - set(df.columns)
    errors = []
    if missing:
        errors.append(f"Missing columns: {', '.join(sorted(missing))}")
    return errors


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean the input dataset before clustering.
    Returns (cleaned_df, list_of_warning_strings).
    """
    warnings_out = []
    n0 = len(df)

    # 1. Drop nulls in required columns
    df = df.dropna(subset=["LinkID", "RoadCategory", "hour", "day_type", "speed", "volume"])
    dropped_nulls = n0 - len(df)
    if dropped_nulls:
        warnings_out.append(f"Dropped {dropped_nulls:,} rows with null values in required columns.")

    # 2. Validate RoadCategory
    before = len(df)
    df = df[df["RoadCategory"].isin(VALID_CATEGORIES)]
    dropped_cat = before - len(df)
    if dropped_cat:
        warnings_out.append(f"Dropped {dropped_cat:,} rows with invalid RoadCategory (not in 1–6).")

    # 3. Validate hour range
    before = len(df)
    df = df[df["hour"].between(0, 23)]
    dropped_hour = before - len(df)
    if dropped_hour:
        warnings_out.append(f"Dropped {dropped_hour:,} rows with hour outside 0–23.")

    # 4. Validate day_type
    before = len(df)
    df = df[df["day_type"].isin(["Weekday", "Weekend"])]
    dropped_day = before - len(df)
    if dropped_day:
        warnings_out.append(f"Dropped {dropped_day:,} rows with invalid day_type (expected Weekday / Weekend).")

    # 5. Cap speed at road-category free-flow maximum (sentinel fix)
    FREE_FLOW = {1: 90, 2: 70, 3: 65, 4: 65, 5: 65, 6: 65}
    before_speeds = (df["speed"] > df["RoadCategory"].map(FREE_FLOW)).sum()
    df["speed"] = df.apply(
        lambda r: min(r["speed"], FREE_FLOW.get(int(r["RoadCategory"]), 90)), axis=1
    )
    if before_speeds:
        warnings_out.append(f"Capped {int(before_speeds):,} speed values exceeding the road-category free-flow limit.")

    # 6. Drop non-positive speeds / volumes
    before = len(df)
    df = df[(df["speed"] > 0) & (df["volume"] >= 0)]
    dropped_neg = before - len(df)
    if dropped_neg:
        warnings_out.append(f"Dropped {dropped_neg:,} rows with non-positive speed or negative volume.")

    # 7. Validate optional columns when present (allow NaN, reject unrecognised codes)
    _opt_valid = {
        "expressway": set(EXPRESSWAY),
        "road_direction": set(ROAD_DIRECTION),
        "source": set(SOURCES),
    }
    for _col, _codes in _opt_valid.items():
        if _col in df.columns:
            before = len(df)
            df = df[df[_col].isna() | df[_col].isin(_codes)]
            dropped = before - len(df)
            if dropped:
                warnings_out.append(f"Dropped {dropped:,} rows with unrecognised '{_col}' code.")

    df = df.reset_index(drop=True)
    return df, warnings_out


# ─── Streamlit cache wrappers around pipeline functions ──────────────────────
# These are thin wrappers so Streamlit can cache results between reruns.
# ALL computation logic lives in clustering/pipeline.py.

@st.cache_data(show_spinner=False)
def cached_k_sweep(X_bytes: bytes, n_rows: int, n_cols: int) -> pd.DataFrame:
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(n_rows, n_cols)
    return sweep_k(X)


@st.cache_data(show_spinner=False)
def cached_kmeans(X_bytes: bytes, n_rows: int, n_cols: int, k: int) -> tuple:
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(n_rows, n_cols)
    labels, centers = run_kmeans(X, k)
    return labels, centers


def load_default_dataset() -> pd.DataFrame | None:
    """Load per-category CSVs if available, else single input file."""
    cat_files = sorted(Path("synthetic_dataset/processed").glob("synthetic_hourly_cat*.csv"))
    if cat_files:
        return pd.concat(
            [pd.read_csv(f, parse_dates=["timestamp_hour"]) for f in cat_files],
            ignore_index=True,
        )
    single = Path("data/input/traffic_hourly.csv")
    if single.exists():
        return pd.read_csv(single, parse_dates=["timestamp_hour"])
    return None


# ─── Session state init ──────────────────────────────────────────────────────
for key, default in [("df", None), ("cluster_results", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 Speed Band Threshold")
    st.caption("Data-driven adaptive speed bands for Singapore roads")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload dataset (CSV)",
        type=["csv"],
        help="Required columns: timestamp_hour, LinkID, RoadCategory, hour, day_type, speed, volume",
    )

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded, parse_dates=["timestamp_hour"])
            errors = validate_schema(df_upload)
            if errors:
                st.error("\n".join(errors))
            else:
                df_upload, pp_warns = preprocess(df_upload)
                st.session_state.df = df_upload
                st.session_state.cluster_results = {}
                st.success(f"Loaded {len(df_upload):,} rows")
                for w in pp_warns:
                    st.warning(f"⚠ {w}")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    else:
        if st.session_state.df is None:
            default = load_default_dataset()
            if default is not None:
                default, pp_warns = preprocess(default)
                st.session_state.df = default
                st.info(f"Using default dataset  ({len(default):,} rows)")
                for w in pp_warns:
                    st.warning(f"⚠ {w}")
            else:
                st.warning("No dataset found. Upload a CSV to begin.")

    st.markdown("---")
    st.markdown("**Required schema**")
    st.markdown(
        "| Column | Type |\n|--------|------|\n"
        "| `timestamp_hour` | datetime (SGT) |\n"
        "| `LinkID` | int |\n"
        "| `RoadCategory` | int (1–6) |\n"
        "| `hour` | int (0–23) |\n"
        "| `day_type` | Weekday / Weekend |\n"
        "| `speed` | float (km/h) |\n"
        "| `volume` | int (veh/hr) |"
    )
    st.markdown("**Optional columns**")
    st.markdown(
        "| Column | Codes |\n|--------|-------|\n"
        "| `expressway` | E1\u2013E8 (BKE, KJE \u2026 PIE) |\n"
        "| `road_direction` | D1 = INBOUND, D2 = OUTBOUND |\n"
        "| `source` | S1 = GPS, S2 = Sensor, S3 = Other |"
    )
# ─── Gate ────────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.info("👈  Upload a dataset using the sidebar to get started.")
    st.stop()

df_main: pd.DataFrame = st.session_state.df.copy()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋  Dataset Explorer", "🔮  Clustering", "🎯  Speed Bands"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Input Dataset")

    # Summary cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df_main):,}")
    c2.metric("Unique LinkIDs", f"{df_main['LinkID'].nunique():,}")
    c3.metric("Road Categories", df_main["RoadCategory"].nunique())
    c4.metric("Days Covered", df_main["timestamp_hour"].dt.date.nunique())
    c5.metric("Day Types", " / ".join(sorted(df_main["day_type"].unique())))

    st.markdown("---")

    # Filters
    f1, f2, f3 = st.columns([2, 1, 2])
    with f1:
        sel_cats = st.multiselect(
            "Road Category",
            options=sorted(df_main["RoadCategory"].unique()),
            default=sorted(df_main["RoadCategory"].unique()),
            format_func=lambda x: f"{x} — {CATEGORY_LABEL.get(x, 'Unknown')}",
        )
    with f2:
        sel_days = st.multiselect(
            "Day Type",
            options=sorted(df_main["day_type"].unique()),
            default=sorted(df_main["day_type"].unique()),
        )
    with f3:
        hour_range = st.slider("Hour (SGT)", 0, 23, (0, 23))

    df_filtered = df_main[
        df_main["RoadCategory"].isin(sel_cats)
        & df_main["day_type"].isin(sel_days)
        & df_main["hour"].between(*hour_range)
    ]

    # Optional column filters — rendered only when the columns exist in the loaded dataset
    _opt_present = [c for c in OPTIONAL_COLS if c in df_main.columns]
    if _opt_present:
        _opt_maps   = {"expressway": EXPRESSWAY, "road_direction": ROAD_DIRECTION, "source": SOURCES}
        _opt_labels = {"expressway": "Expressway", "road_direction": "Road Direction", "source": "Source"}
        _opt_cols_ui = st.columns(len(_opt_present))
        for _i, _col in enumerate(_opt_present):
            with _opt_cols_ui[_i]:
                _opts = sorted(df_filtered[_col].dropna().unique())
                _sel  = st.multiselect(
                    _opt_labels[_col], options=_opts, default=_opts,
                    format_func=lambda x, m=_opt_maps[_col]: m.get(x, x),
                )
                df_filtered = df_filtered[df_filtered[_col].isin(_sel)]

    st.caption(f"Showing {len(df_filtered):,} of {len(df_main):,} records")
    st.dataframe(df_filtered.head(5_000), use_container_width=True, height=300)

    st.download_button(
        "⬇  Download filtered data",
        data=df_filtered.to_csv(index=False),
        file_name="filtered_traffic.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Distribution charts
    col_a, col_b = st.columns(2)
    with col_a:
        fig_spd = px.histogram(
            df_filtered, x="speed", color="day_type", nbins=60,
            title="Speed Distribution",
            labels={"speed": "Speed (km/h)"},
            barmode="overlay", opacity=0.72,
            color_discrete_map={"Weekday": "#4C72B0", "Weekend": "#DD8452"},
        )
        fig_spd.update_layout(height=340, legend_title_text="Day type")
        st.plotly_chart(fig_spd, use_container_width=True)

    with col_b:
        fig_vol = px.histogram(
            df_filtered, x="volume", color="day_type", nbins=60,
            title="Volume Distribution",
            labels={"volume": "Volume (veh/hr)"},
            barmode="overlay", opacity=0.72,
            color_discrete_map={"Weekday": "#4C72B0", "Weekend": "#DD8452"},
        )
        fig_vol.update_layout(height=340, legend_title_text="Day type")
        st.plotly_chart(fig_vol, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        cat_counts = (
            df_filtered.groupby("RoadCategory").size()
            .reset_index(name="count")
        )
        cat_counts["label"] = cat_counts["RoadCategory"].map(
            lambda x: f"{x} — {CATEGORY_LABEL.get(x, '?')}"
        )
        fig_cat = px.bar(
            cat_counts, x="label", y="count",
            title="Records by Road Category",
            labels={"label": "Road Category", "count": "Records"},
            color="count", color_continuous_scale="Blues",
        )
        fig_cat.update_layout(height=340, coloraxis_showscale=False, xaxis_tickangle=-20)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_d:
        hour_profile = (
            df_filtered.groupby(["hour", "RoadCategory"])["speed"]
            .median().reset_index()
        )
        hour_profile["Category"] = hour_profile["RoadCategory"].map(
            lambda x: CATEGORY_LABEL.get(x, str(x))
        )
        fig_hspd = px.line(
            hour_profile, x="hour", y="speed", color="Category",
            markers=True,
            title="Median Speed by Hour — per Road Category",
            labels={"hour": "Hour (SGT)", "speed": "Median Speed (km/h)"},
        )
        fig_hspd.update_layout(height=340)
        st.plotly_chart(fig_hspd, use_container_width=True)

    # Speed vs Volume scatter (sample)
    st.markdown("---")
    fig_sv = px.scatter(
        df_filtered.sample(min(5_000, len(df_filtered)), random_state=42),
        x="volume", y="speed",
        color=df_filtered.sample(min(5_000, len(df_filtered)), random_state=42)["RoadCategory"]
              .map(lambda x: CATEGORY_LABEL.get(x, str(x))),
        opacity=0.5,
        title="Speed vs Volume (sample of 5,000)",
        labels={"volume": "Volume (veh/hr)", "speed": "Speed (km/h)", "color": "Road Category"},
    )
    fig_sv.update_traces(marker=dict(size=4))
    fig_sv.update_layout(height=400)
    st.plotly_chart(fig_sv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Clustering")
    st.caption(
        "Select a road category and day type, then run clustering. "
        "K is chosen automatically by peak silhouette score — override it with the slider."
    )

    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 0.8])
    with ctrl1:
        avail_cats = sorted(df_main["RoadCategory"].unique())
        sel_cat = st.selectbox(
            "Road Category", avail_cats,
            format_func=lambda x: f"{x} — {CATEGORY_LABEL.get(x, '?')}",
        )
    with ctrl2:
        avail_days = sorted(df_main["day_type"].unique())
        sel_day = st.selectbox("Day Type", avail_days)
    with ctrl3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Clustering", type="primary", use_container_width=True)

    subset_key = (sel_cat, sel_day)
    subset_df = df_main[
        (df_main["RoadCategory"] == sel_cat) & (df_main["day_type"] == sel_day)
    ].copy()

    if len(subset_df) == 0:
        st.warning(f"No data for Category {sel_cat} — {sel_day}. Choose a different combination.")
    else:
        st.metric(
            f"Subset  ·  Cat {sel_cat} ({CATEGORY_LABEL.get(sel_cat,'?')}) — {sel_day}",
            f"{len(subset_df):,} records",
        )

        need_run = run_btn or subset_key not in st.session_state.cluster_results

        if need_run:
            sub_eng, _, active_feat_cols = engineer_subset(subset_df)
            X_full = sub_eng[active_feat_cols].values.astype(np.float64)
            n_feat_cols = len(active_feat_cols)

            # Sample for K-sweep speed
            if len(X_full) > K_SWEEP_SAMPLE:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_full), K_SWEEP_SAMPLE, replace=False)
                X_sweep = X_full[idx]
            else:
                X_sweep = X_full

            with st.spinner("Running K sweep (K = 2 … 10) …"):
                diag = cached_k_sweep(X_sweep.tobytes(), len(X_sweep), n_feat_cols)

            best_k = pick_optimal_k(diag)

            # K-sweep chart
            fig_k = go.Figure()
            fig_k.add_trace(go.Scatter(
                x=diag["k"], y=diag["silhouette"], mode="lines+markers",
                name="Silhouette ↑ (higher = better)", line=dict(color="#2196F3", width=2),
            ))
            fig_k.add_trace(go.Scatter(
                x=diag["k"], y=diag["davies_bouldin"], mode="lines+markers",
                name="Davies-Bouldin ↓ (lower = better)",
                line=dict(color="#F44336", width=2, dash="dash"),
            ))
            fig_k.add_vline(
                x=best_k, line_dash="dot", line_color="green",
                annotation_text=f"  Optimal K = {best_k}",
                annotation_position="top right",
                annotation_font=dict(color="green", size=13),
            )
            fig_k.update_layout(
                title=f"K Sweep — Cat {sel_cat} ({CATEGORY_LABEL.get(sel_cat,'?')}) {sel_day}",
                xaxis=dict(title="Number of Clusters (K)", dtick=1),
                yaxis_title="Score",
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_k, use_container_width=True)

            k_final = st.slider(
                "Override K (optimal highlighted)", 2, 10, best_k,
                help="The slider starts at the K with the best silhouette score.",
            )

            with st.spinner(f"Fitting K-Means  K = {k_final} on {len(X_full):,} records …"):
                labels, centers = cached_kmeans(X_full.tobytes(), len(X_full), n_feat_cols, k_final)

            labels, centers = merge_small_clusters(labels, centers)

            cat_pfx = CATEGORY_PREFIX.get(sel_cat, f"C{sel_cat}")
            day_pfx = DAY_PREFIX.get(sel_day, sel_day[:2].upper())
            sub_eng["cluster_raw"] = labels
            sub_eng["cluster_id"] = sub_eng["cluster_raw"].apply(
                lambda i: build_cluster_id(cat_pfx, day_pfx, i)
            )
            st.session_state.cluster_results[subset_key] = sub_eng.copy()

        else:
            sub_eng = st.session_state.cluster_results[subset_key]
            k_final = sub_eng["cluster_id"].nunique()

            diag_placeholder = None
            k_final = st.slider("Override K (re-run to apply)", 2, 10, k_final, disabled=True)

        # ── Scatter plots ──────────────────────────────────────────────────
        if subset_key in st.session_state.cluster_results:
            sub_eng = st.session_state.cluster_results[subset_key]
            sample = sub_eng.sample(min(SCATTER_SAMPLE, len(sub_eng)), random_state=42)

            # Sort cluster_ids for consistent legend order
            ordered_clusters = (
                sub_eng.groupby("cluster_id")["speed"].mean()
                .sort_values().index.tolist()
            )

            st.markdown("---")
            s1, s2 = st.columns(2)

            with s1:
                fig_sc1 = px.scatter(
                    sample, x="hour", y="speed", color="cluster_id",
                    category_orders={"cluster_id": ordered_clusters},
                    title="Hour vs Speed — coloured by cluster",
                    labels={"hour": "Hour (SGT)", "speed": "Speed (km/h)", "cluster_id": "Cluster"},
                    hover_data=["LinkID", "volume"],
                    opacity=0.55,
                )
                fig_sc1.update_traces(marker=dict(size=4))
                fig_sc1.update_layout(height=420, plot_bgcolor="white")
                st.plotly_chart(fig_sc1, use_container_width=True)

            with s2:
                fig_sc2 = px.scatter(
                    sample, x="volume", y="speed", color="cluster_id",
                    category_orders={"cluster_id": ordered_clusters},
                    title="Volume vs Speed — coloured by cluster",
                    labels={"volume": "Volume (veh/hr)", "speed": "Speed (km/h)", "cluster_id": "Cluster"},
                    hover_data=["LinkID", "hour"],
                    opacity=0.55,
                )
                fig_sc2.update_traces(marker=dict(size=4))
                fig_sc2.update_layout(height=420, plot_bgcolor="white")
                st.plotly_chart(fig_sc2, use_container_width=True)

            # ── Cluster profile table ──────────────────────────────────────
            st.markdown("---")
            profile = extract_bands(sub_eng).rename(columns={
                "cluster_speed_mean": "mean", "cluster_speed_std": "std",
            })
            profile["band_width"] = (profile["upper_band"] - profile["lower_band"]).round(2)
            profile["vol_mean"] = (
                sub_eng.groupby("cluster_id")["volume"].mean().round(0).reindex(profile["cluster_id"]).values
            )
            profile = profile.sort_values("mean").reset_index(drop=True)

            st.subheader("Cluster Profiles")
            st.dataframe(
                profile.rename(columns={
                    "cluster_id": "Cluster", "cluster_size": "Size",
                    "mean": "Mean Speed", "lower_band": "Lower (P10)",
                    "upper_band": "Upper (P90)", "std": "Std Dev",
                    "band_width": "Band Width", "vol_mean": "Mean Volume",
                }),
                use_container_width=True,
            )

            # Cluster sizes bar
            fig_sizes = px.bar(
                profile, x="cluster_id", y="cluster_size",
                color="mean", color_continuous_scale="RdYlGn",
                title="Cluster Sizes (colour = mean speed)",
                labels={"cluster_id": "Cluster", "cluster_size": "Records", "mean": "Mean Speed"},
                category_orders={"cluster_id": ordered_clusters},
            )
            fig_sizes.update_layout(height=300, coloraxis_showscale=True)
            st.plotly_chart(fig_sizes, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SPEED BANDS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Speed Band Derivation")

    if not st.session_state.cluster_results:
        st.info("Run clustering in the **🔮 Clustering** tab first, then return here.")
    else:
        result_keys = list(st.session_state.cluster_results.keys())
        key_labels = {
            k: f"Cat {k[0]} ({CATEGORY_LABEL.get(k[0], '?')}) — {k[1]}"
            for k in result_keys
        }
        sel_key = st.selectbox(
            "View results for", result_keys,
            format_func=lambda k: key_labels[k],
        )
        res_df = st.session_state.cluster_results[sel_key]

        # Compute bands using the shared pipeline function
        bands = (
            extract_bands(res_df)
            .rename(columns={"cluster_speed_mean": "spd_mean"})
            .sort_values("spd_mean")
            .reset_index(drop=True)
        )
        ordered = bands["cluster_id"].tolist()

        # ── VISUAL 1: Band range chart ─────────────────────────────────────
        st.markdown("---")
        st.subheader("Speed Band Ranges per Cluster")
        st.caption(
            "Each bar spans **P10 → P90** of the cluster's speed distribution.  "
            "The ◆ diamond marks the mean speed.  Narrow bars = tight, well-defined regime."
        )

        colors = px.colors.qualitative.Safe
        fig_bands = go.Figure()
        for i, row in bands.iterrows():
            c = colors[i % len(colors)]
            fig_bands.add_trace(go.Bar(
                name=row["cluster_id"],
                x=[row["upper_band"] - row["lower_band"]],
                y=[row["cluster_id"]],
                base=[row["lower_band"]],
                orientation="h",
                marker_color=c,
                marker_line=dict(color="rgba(0,0,0,0.25)", width=1),
                hovertemplate=(
                    f"<b>{row['cluster_id']}</b><br>"
                    f"Lower Band (P10): {row['lower_band']} km/h<br>"
                    f"Upper Band (P90): {row['upper_band']} km/h<br>"
                    f"Mean Speed: {row['spd_mean']} km/h<br>"
                    f"Band Width: {row['upper_band'] - row['lower_band']:.1f} km/h<br>"
                    f"Cluster Size: {int(row['cluster_size']):,}<extra></extra>"
                ),
                showlegend=False,
            ))
            # Mean marker
            fig_bands.add_trace(go.Scatter(
                x=[row["spd_mean"]], y=[row["cluster_id"]],
                mode="markers",
                marker=dict(color="black", size=11, symbol="diamond"),
                showlegend=i == 0,
                name="Mean speed",
                hoverinfo="skip",
            ))

        fig_bands.update_layout(
            title=f"Speed Bands — {key_labels[sel_key]}",
            xaxis=dict(title="Speed (km/h)", range=[0, 105], gridcolor="#EEEEEE"),
            yaxis=dict(title="Cluster", categoryorder="array", categoryarray=ordered),
            barmode="overlay",
            height=max(320, 52 * len(bands) + 80),
            plot_bgcolor="white",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_bands, use_container_width=True)

        # ── VISUAL 2: P10/P90 deep-dive on a single cluster ───────────────
        st.markdown("---")
        st.subheader("How Speed Bands Are Derived — Cluster Deep-Dive")
        st.caption(
            "Select any cluster to see its full speed distribution. "
            "The **green shaded region** between the two dashed lines is the adaptive speed band."
        )

        sel_cluster = st.selectbox("Select cluster", ordered)
        cluster_speeds = res_df[res_df["cluster_id"] == sel_cluster]["speed"]
        p10 = cluster_speeds.quantile(0.10)
        p90 = cluster_speeds.quantile(0.90)
        mean_spd = cluster_speeds.mean()

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Cluster Size", f"{len(cluster_speeds):,}")
        mc2.metric("Lower Band (P10)", f"{p10:.1f} km/h")
        mc3.metric("Upper Band (P90)", f"{p90:.1f} km/h")
        mc4.metric("Band Width", f"{p90 - p10:.1f} km/h")

        # Histogram with P10–P90 shaded
        bins = np.linspace(
            max(0, cluster_speeds.min() - 3),
            cluster_speeds.max() + 3,
            55,
        )
        counts, edges = np.histogram(cluster_speeds, bins=bins)
        bar_w = np.diff(edges)

        fig_hist = go.Figure()

        mask_low  = edges[:-1] < p10
        mask_band = (edges[:-1] >= p10) & (edges[:-1] < p90)
        mask_high = edges[:-1] >= p90

        for mask, color, label in [
            (mask_low,  "rgba(180,180,180,0.55)", "Outside band (below P10)"),
            (mask_band, "rgba(56,169,70,0.75)",   "Speed Band  P10 → P90"),
            (mask_high, "rgba(180,180,180,0.55)", "Outside band (above P90)"),
        ]:
            if mask.any():
                fig_hist.add_trace(go.Bar(
                    x=edges[:-1][mask],
                    y=counts[mask],
                    width=bar_w[mask],
                    name=label,
                    marker_color=color,
                    marker_line_width=0,
                    offset=0,
                ))

        fig_hist.add_vline(
            x=p10, line_color="#1565C0", line_width=2.5, line_dash="dash",
            annotation_text=f" P10 = {p10:.1f} km/h  →  Lower Band",
            annotation_position="top left",
            annotation_font=dict(color="#1565C0", size=12, family="Arial Bold"),
        )
        fig_hist.add_vline(
            x=p90, line_color="#B71C1C", line_width=2.5, line_dash="dash",
            annotation_text=f" P90 = {p90:.1f} km/h  →  Upper Band",
            annotation_position="top right",
            annotation_font=dict(color="#B71C1C", size=12, family="Arial Bold"),
        )
        fig_hist.add_vline(
            x=mean_spd, line_color="#333333", line_width=1.5, line_dash="dot",
            annotation_text=f"  Mean = {mean_spd:.1f}",
            annotation_position="bottom right",
        )

        fig_hist.update_layout(
            title=f"Speed Distribution — {sel_cluster}",
            xaxis=dict(title="Speed (km/h)", gridcolor="#EEEEEE"),
            yaxis=dict(title="Count", gridcolor="#EEEEEE"),
            barmode="stack",
            bargap=0.02,
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.info(
            "**Why P10 / P90?**  "
            "The 10th and 90th percentiles remove the influence of rare outliers (incidents, sensor noise) "
            "while capturing 80% of the cluster's typical operating-speed envelope. "
            "Using min/max would make the band sensitive to individual extreme readings."
        )

        # ── VISUAL 3: Box plots for all clusters ──────────────────────────
        st.markdown("---")
        st.subheader("Speed Distribution across All Clusters")

        fig_box = px.box(
            res_df, x="cluster_id", y="speed", color="cluster_id",
            category_orders={"cluster_id": ordered},
            title=f"Speed Box Plots — {key_labels[sel_key]}",
            labels={"cluster_id": "Cluster", "speed": "Speed (km/h)"},
            points=False,
        )
        fig_box.update_layout(showlegend=False, height=420, plot_bgcolor="white")
        st.plotly_chart(fig_box, use_container_width=True)

        # ── VISUAL 4: Cluster occurrence heatmap by hour ───────────────────
        st.markdown("---")
        st.subheader("When Does Each Cluster Occur?")
        st.caption("Darker = more records assigned to that cluster at that hour of day.")

        pivot = (
            res_df.groupby(["cluster_id", "hour"]).size()
            .reset_index(name="count")
        )
        pivot_wide = (
            pivot.pivot(index="cluster_id", columns="hour", values="count")
            .fillna(0)
            .reindex(ordered)
        )

        fig_heat = px.imshow(
            pivot_wide,
            labels=dict(x="Hour (SGT)", y="Cluster", color="Records"),
            title=f"Cluster × Hour Heatmap — {key_labels[sel_key]}",
            color_continuous_scale="Blues",
            aspect="auto",
            height=max(320, 44 * len(bands) + 80),
        )
        fig_heat.update_xaxes(dtick=1)
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Export — current subset ───────────────────────────────────────
        st.markdown("---")
        st.subheader("Output Dataset")
        st.caption(
            "The original 7 columns with `cluster_id`, `lower_band` (P10), and `upper_band` (P90) appended. "
            "Every row in the same cluster shares the same band values."
        )

        export_df = res_df.merge(
            bands[["cluster_id", "lower_band", "upper_band"]], on="cluster_id", how="left"
        )
        drop_cols = [c for c in export_df.columns
                     if c in FEATURE_COLS or c.endswith("_enc")
                     or c in ("cluster_raw", "day_bin", "sin_h", "cos_h", "speed_norm", "volume_norm")]
        export_df = export_df.drop(columns=drop_cols)

        # Ensure column order: original 7 + cluster_id, lower_band, upper_band
        base_cols = [c for c in ["timestamp_hour", "LinkID", "RoadCategory", "hour",
                                  "day_type", "speed", "volume"] + OPTIONAL_COLS if c in export_df.columns]
        extra_cols = ["cluster_id", "lower_band", "upper_band"]
        export_df = export_df[base_cols + extra_cols]

        st.dataframe(export_df.head(200), use_container_width=True, height=260)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                f"⬇  Download — {key_labels[sel_key]} (this subset)",
                data=export_df.to_csv(index=False),
                file_name=f"traffic_with_bands_cat{sel_key[0]}_{sel_key[1].lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # ── Export — all clustered subsets combined ───────────────────────
        with dl2:
            all_frames = []
            for k, r in st.session_state.cluster_results.items():
                b = (
                    r.groupby("cluster_id")["speed"]
                    .agg(lower_band=lambda x: round(x.quantile(0.10), 2),
                         upper_band=lambda x: round(x.quantile(0.90), 2))
                    .reset_index()
                )
                f = r.merge(b[["cluster_id", "lower_band", "upper_band"]], on="cluster_id", how="left")
                drop = [c for c in f.columns
                        if c in FEATURE_COLS or c.endswith("_enc")
                        or c in ("cluster_raw", "day_bin", "sin_h", "cos_h", "speed_norm", "volume_norm")]
                f = f.drop(columns=drop)
                f = f[[c for c in base_cols if c in f.columns] + extra_cols]
                all_frames.append(f)

            combined = pd.concat(all_frames, ignore_index=True) if all_frames else export_df

            n_subsets = len(st.session_state.cluster_results)
            st.download_button(
                f"⬇  Download — All {n_subsets} clustered subset(s) combined",
                data=combined.to_csv(index=False),
                file_name="traffic_with_bands_all.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.info(
            f"**{len(combined):,} rows** across **{n_subsets}** subset(s) in combined output.  "
            "Run clustering on additional Road Category / Day Type combinations in the "
            "🔮 Clustering tab to include them here."
        )

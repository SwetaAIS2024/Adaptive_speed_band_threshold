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

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Speed Band Threshold",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───────────────────────────────────────────────────────────────
REQUIRED_COLS = {"timestamp_hour", "LinkID", "RoadCategory", "hour", "day_type", "speed", "volume"}
VALID_CATEGORIES = {
    1: "Expressway", 2: "Major Arterial", 3: "Arterial",
    4: "Minor Arterial", 5: "Local Access", 6: "Minor Access",
}
CAT_PREFIX = {1: "EXP", 2: "MJR", 3: "ART", 4: "MIN", 5: "LOC", 6: "ACC"}
DAY_PREFIX = {"Weekday": "WD", "Weekend": "WE"}
FEATURE_COLS = ["sin_h", "cos_h", "day_bin", "speed_norm", "volume_norm"]

# Max rows passed to K-sweep to keep it fast on large datasets
K_SWEEP_SAMPLE = 50_000
# Max scatter plot points
SCATTER_SAMPLE = 8_000

# ─── Helpers ─────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> list[str]:
    missing = REQUIRED_COLS - set(df.columns)
    errors = []
    if missing:
        errors.append(f"Missing columns: {', '.join(sorted(missing))}")
    return errors


def engineer_features(subset: pd.DataFrame) -> pd.DataFrame:
    df = subset.copy()
    df["sin_h"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_h"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_bin"] = (df["day_type"] == "Weekend").astype(int)
    scaler = MinMaxScaler()
    df[["speed_norm", "volume_norm"]] = scaler.fit_transform(df[["speed", "volume"]])
    return df


@st.cache_data(show_spinner=False)
def cached_k_sweep(X_bytes: bytes, n_rows: int) -> pd.DataFrame:
    """K-sweep cached by numpy array bytes fingerprint."""
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(n_rows, len(FEATURE_COLS))
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sample = min(10_000, len(X))
        sil = silhouette_score(X, labels, sample_size=sample, random_state=42)
        db = davies_bouldin_score(X, labels)
        rows.append({"k": k, "inertia": round(km.inertia_, 2),
                     "silhouette": round(sil, 4), "davies_bouldin": round(db, 4)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_kmeans(X_bytes: bytes, n_rows: int, k: int) -> np.ndarray:
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(n_rows, len(FEATURE_COLS))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X)


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
                st.session_state.df = df_upload
                st.session_state.cluster_results = {}
                st.success(f"Loaded {len(df_upload):,} rows")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    else:
        if st.session_state.df is None:
            default = load_default_dataset()
            if default is not None:
                st.session_state.df = default
                st.info(f"Using default dataset  ({len(default):,} rows)")
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
            format_func=lambda x: f"{x} — {VALID_CATEGORIES.get(x, 'Unknown')}",
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
            lambda x: f"{x} — {VALID_CATEGORIES.get(x, '?')}"
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
            lambda x: VALID_CATEGORIES.get(x, str(x))
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
              .map(lambda x: VALID_CATEGORIES.get(x, str(x))),
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
            format_func=lambda x: f"{x} — {VALID_CATEGORIES.get(x, '?')}",
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
            f"Subset  ·  Cat {sel_cat} ({VALID_CATEGORIES.get(sel_cat,'?')}) — {sel_day}",
            f"{len(subset_df):,} records",
        )

        need_run = run_btn or subset_key not in st.session_state.cluster_results

        if need_run:
            sub_eng = engineer_features(subset_df)
            X_full = sub_eng[FEATURE_COLS].values.astype(np.float64)

            # Sample for K-sweep speed
            if len(X_full) > K_SWEEP_SAMPLE:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_full), K_SWEEP_SAMPLE, replace=False)
                X_sweep = X_full[idx]
            else:
                X_sweep = X_full

            with st.spinner("Running K sweep (K = 2 … 10) …"):
                diag = cached_k_sweep(X_sweep.tobytes(), len(X_sweep))

            best_k = int(diag.loc[diag["silhouette"].idxmax(), "k"])

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
                title=f"K Sweep — Cat {sel_cat} ({VALID_CATEGORIES.get(sel_cat,'?')}) {sel_day}",
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
                labels = cached_kmeans(X_full.tobytes(), len(X_full), k_final)

            cat_pfx = CAT_PREFIX.get(sel_cat, f"C{sel_cat}")
            day_pfx = DAY_PREFIX.get(sel_day, sel_day[:2].upper())
            sub_eng["cluster_raw"] = labels
            sub_eng["cluster_id"] = sub_eng["cluster_raw"].apply(
                lambda i: f"{cat_pfx}_{day_pfx}_{i:02d}"
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
            profile = (
                sub_eng.groupby("cluster_id")["speed"]
                .agg(
                    cluster_size="count",
                    mean=lambda x: round(x.mean(), 2),
                    lower_band=lambda x: round(x.quantile(0.10), 2),
                    upper_band=lambda x: round(x.quantile(0.90), 2),
                    std=lambda x: round(x.std(), 2),
                )
                .reset_index()
            )
            profile["band_width"] = (profile["upper_band"] - profile["lower_band"]).round(2)
            profile["vol_mean"] = (
                sub_eng.groupby("cluster_id")["volume"].mean().round(0).values
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
            k: f"Cat {k[0]} ({VALID_CATEGORIES.get(k[0], '?')}) — {k[1]}"
            for k in result_keys
        }
        sel_key = st.selectbox(
            "View results for", result_keys,
            format_func=lambda k: key_labels[k],
        )
        res_df = st.session_state.cluster_results[sel_key]

        # Compute bands for this subset
        bands = (
            res_df.groupby("cluster_id")["speed"]
            .agg(
                lower_band=lambda x: x.quantile(0.10),
                upper_band=lambda x: x.quantile(0.90),
                spd_mean="mean",
                cluster_size="count",
            )
            .round(2)
            .reset_index()
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

        # ── Export ────────────────────────────────────────────────────────
        st.markdown("---")
        export_df = res_df.merge(
            bands[["cluster_id", "lower_band", "upper_band"]], on="cluster_id", how="left"
        )
        drop_cols = [c for c in FEATURE_COLS + ["cluster_raw"] if c in export_df.columns]
        export_df = export_df.drop(columns=drop_cols)

        st.download_button(
            "⬇  Download dataset with speed bands",
            data=export_df.to_csv(index=False),
            file_name=f"traffic_with_bands_{sel_key[0]}_{sel_key[1].lower()}.csv",
            mime="text/csv",
        )

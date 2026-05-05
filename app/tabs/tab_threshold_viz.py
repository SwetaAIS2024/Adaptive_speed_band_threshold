"""
Tab: Adaptive Threshold Visualisation
Speed (km/h) on the Y-axis, Hour of day (0-23) on the X-axis.

Traces (each individually toggleable via the Plotly legend):
  • Actual speed observations  — scatter points
  • P10 adaptive threshold     — per-hour 10th-percentile line
  • P90 upper band             — per-hour 90th-percentile line  (optional)
  • Hourly mean speed          — per-hour mean line             (optional)

A day-type filter (Weekday / Weekend / All) and a sensor-ID multi-select
are provided so users can drill down without leaving the tab.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .context import AppContext, discover_preproc_files

# ── colour palette ────────────────────────────────────────────────────────────
_COL_POINTS = "rgba(100, 160, 230, 0.35)"   # translucent blue scatter
_COL_P10    = "#E53935"                       # red  — adaptive threshold
_COL_P90    = "#43A047"                       # green — upper band
_COL_MEAN   = "#FB8C00"                       # orange — mean


def _load_data(ctx: AppContext) -> pd.DataFrame | None:
    """Render the file-loading widget; return the loaded DataFrame or None."""
    preproc_files = discover_preproc_files(ctx)

    col_sel, col_up = st.columns([1, 1])
    with col_sel:
        if preproc_files:
            sel = st.selectbox(
                "Select preprocessed file",
                ["(none)"] + list(preproc_files.keys()),
                key="thresh_preproc_sel",
            )
            if sel != "(none)":
                ppath = preproc_files[sel]
                if (
                    st.session_state.get("_thresh_loaded") != ppath
                    or st.session_state.get("_thresh_df") is None
                ):
                    st.session_state["_thresh_df"]     = pd.read_csv(ppath)
                    st.session_state["_thresh_loaded"] = ppath
                st.success(f"Loaded {len(st.session_state['_thresh_df']):,} rows")
        else:
            st.info("No preprocessed files found on disk.")

    with col_up:
        uploaded = st.file_uploader(
            "Or upload any CSV", type=["csv"], key="thresh_upload"
        )
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                st.session_state["_thresh_df"]     = df_up
                st.session_state["_thresh_loaded"] = None
                st.success(
                    f"Loaded {len(df_up):,} rows  ·  {len(df_up.columns)} columns"
                )
            except Exception as e:
                st.error(f"Could not read file: {e}")

    return st.session_state.get("_thresh_df")


def _detect_cols(df: pd.DataFrame, ctx: AppContext) -> dict:
    """Return {speed, id, day_type, hour} column names from the DataFrame."""
    if ctx.eda_mod:
        dc = ctx.eda_mod._detect_cols(df)
    else:
        dc = {}

    # fallback: case-insensitive scan
    col_lower = {c.lower(): c for c in df.columns}
    result = {
        "speed":     dc.get("speed") or col_lower.get("speed"),
        "id":        dc.get("id")    or col_lower.get("link_id") or col_lower.get("equip_id"),
        "day_type":  col_lower.get("day_type"),
        "hour":      col_lower.get("hour"),
        "road_name": col_lower.get("road_name"),
    }
    return result


def render(ctx: AppContext) -> None:
    st.header("Threshold Visualisation")
    st.caption(
        "Speed observations plotted against hour of day.  "
        "The **red line** is the adaptive P10 threshold — it varies by hour "
        "instead of being a single flat value."
    )

    # ── File loading ──────────────────────────────────────────────────────────
    df_raw = _load_data(ctx)
    st.markdown("---")

    if df_raw is None:
        st.info("Select a preprocessed file or upload a CSV above to get started.")
        return

    cols = _detect_cols(df_raw, ctx)
    speed_col     = cols["speed"]
    id_col        = cols["id"]
    day_type_col  = cols["day_type"]
    hour_col      = cols["hour"]
    road_name_col = cols["road_name"]

    if not speed_col or not hour_col:
        st.error(
            "Could not detect **SPEED** or **hour** columns in this file. "
            f"Columns found: {list(df_raw.columns)}"
        )
        return

    df = df_raw.copy()
    df[hour_col]  = pd.to_numeric(df[hour_col],  errors="coerce")
    df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce")
    df = df.dropna(subset=[hour_col, speed_col])

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc3 = st.columns([1, 1])

    with fc1:
        day_opts = ["All"]
        if day_type_col and day_type_col in df.columns:
            day_opts += sorted(df[day_type_col].dropna().unique().tolist())
        day_sel = st.selectbox("Day type", day_opts, key="thresh_day")

    with fc3:
        scatter_sample = st.number_input(
            "Max scatter points", min_value=500, max_value=100_000,
            value=10_000, step=500, key="thresh_scatter_n",
        )

    # ── Road name filter (cascade parent) ────────────────────────────────────
    road_sel = []
    has_road_col = road_name_col and road_name_col in df.columns
    has_id_col   = id_col and id_col in df.columns

    if has_road_col:
        all_roads = sorted(df[road_name_col].dropna().unique().tolist())
        prev_road = st.session_state.get("_thresh_prev_road", [])
        road_sel  = st.multiselect(
            f"Filter by {road_name_col} (leave empty = all)",
            all_roads, default=[], key="thresh_road_names",
        )
        # detect road selection change → clear stale LINK_ID selection
        if road_sel != prev_road:
            st.session_state["_thresh_prev_road"] = road_sel
            st.session_state.pop("thresh_ids", None)   # reset downstream widget

    # ── LINK_ID filter cascades from road selection ───────────────────────────
    id_sel = []
    if has_id_col:
        if has_road_col and road_sel:
            # restrict IDs to those that belong to the selected road(s)
            id_pool = sorted(
                df[df[road_name_col].isin(road_sel)][id_col].dropna().unique().tolist()
            )
            caption = f"{len(id_pool)} {id_col}(s) on selected road(s)"
        else:
            id_pool = sorted(df[id_col].dropna().unique().tolist())
            caption = None

        id_default = [i for i in id_pool[:5] if i in id_pool]
        id_sel = st.multiselect(
            f"Filter by {id_col} (leave empty = all)",
            id_pool,
            default=id_default,
            key="thresh_ids",
        )
        if caption:
            st.caption(caption)

    # ── Apply filters ─────────────────────────────────────────────────────────
    df_filt = df.copy()
    if day_sel != "All" and day_type_col and day_type_col in df_filt.columns:
        df_filt = df_filt[df_filt[day_type_col] == day_sel]

    if road_sel and has_road_col:
        df_filt = df_filt[df_filt[road_name_col].isin(road_sel)]

    if id_sel and has_id_col:
        df_filt = df_filt[df_filt[id_col].isin(id_sel)]

    if df_filt.empty:
        st.warning("No data matches the current filters.")
        return

    # ── Trace toggles ─────────────────────────────────────────────────────────
    st.markdown("**Show / hide traces**")
    t1, t2, t3, t4 = st.columns(4)
    show_points = t1.checkbox("Actual speed points", value=True, key="thresh_t_pts")
    show_p10    = t2.checkbox("P10 adaptive threshold", value=True, key="thresh_t_p10")
    show_p90    = t3.checkbox("P90 upper band",         value=True, key="thresh_t_p90")
    show_mean   = t4.checkbox("Hourly mean",            value=False, key="thresh_t_mean")

    # ── Hourly aggregates ─────────────────────────────────────────────────────
    hourly = (
        df_filt.groupby(hour_col)[speed_col]
        .agg(
            p10=lambda x: x.quantile(0.10),
            p90=lambda x: x.quantile(0.90),
            mean="mean",
            count="count",
        )
        .reset_index()
        .sort_values(hour_col)
    )

    hours      = hourly[hour_col].tolist()
    p10_vals   = hourly["p10"].round(2).tolist()
    p90_vals   = hourly["p90"].round(2).tolist()
    mean_vals  = hourly["mean"].round(2).tolist()
    count_vals = hourly["count"].tolist()

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # 1. Actual speed scatter (sampled)
    if show_points:
        df_scatter = df_filt.sample(
            min(int(scatter_sample), len(df_filt)), random_state=42
        )
        # add slight jitter on hour so overlapping points are visible
        jitter = np.random.default_rng(0).uniform(-0.35, 0.35, len(df_scatter))
        fig.add_trace(go.Scatter(
            x=df_scatter[hour_col].values + jitter,
            y=df_scatter[speed_col].values,
            mode="markers",
            name="Actual speed",
            marker=dict(color=_COL_POINTS, size=4, line=dict(width=0)),
            hovertemplate="Hour: %{x:.1f}<br>Speed: %{y:.1f} km/h<extra></extra>",
        ))

    # 2. P90 upper band
    if show_p90:
        fig.add_trace(go.Scatter(
            x=hours, y=p90_vals,
            mode="lines+markers",
            name="P90 upper band",
            line=dict(color=_COL_P90, width=2.5, dash="dot"),
            marker=dict(size=6),
            hovertemplate=(
                "Hour: %{x}<br>P90: %{y:.1f} km/h"
                "<br>n=%{customdata:,}<extra></extra>"
            ),
            customdata=count_vals,
        ))

    # 3. Hourly mean
    if show_mean:
        fig.add_trace(go.Scatter(
            x=hours, y=mean_vals,
            mode="lines+markers",
            name="Hourly mean",
            line=dict(color=_COL_MEAN, width=2.5, dash="dash"),
            marker=dict(size=6),
            hovertemplate=(
                "Hour: %{x}<br>Mean: %{y:.1f} km/h"
                "<br>n=%{customdata:,}<extra></extra>"
            ),
            customdata=count_vals,
        ))

    # 4. P10 adaptive threshold — rendered last so it sits on top
    if show_p10:
        fig.add_trace(go.Scatter(
            x=hours, y=p10_vals,
            mode="lines+markers",
            name="P10 adaptive threshold",
            line=dict(color=_COL_P10, width=3),
            marker=dict(size=7, symbol="circle"),
            hovertemplate=(
                "Hour: %{x}<br>"
                "<b>P10 threshold: %{y:.1f} km/h</b>"
                "<br>n=%{customdata:,}<extra></extra>"
            ),
            customdata=count_vals,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    n_pts  = len(df_filt)
    title  = "Speed vs Hour of Day"
    if day_sel != "All":
        title += f"  [{day_sel}]"
    if id_sel:
        id_str = ", ".join(str(i) for i in id_sel[:3])
        if len(id_sel) > 3:
            id_str += f" +{len(id_sel) - 3} more"
        title += f"  |  {id_col}: {id_str}"
    if road_sel:
        road_str = ", ".join(str(r) for r in road_sel[:2])
        if len(road_sel) > 2:
            road_str += f" +{len(road_sel) - 2} more"
        title += f"  |  Road: {road_str}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear", tick0=0, dtick=1,
            range=[-0.5, 23.5],
            gridcolor="#EEEEEE",
        ),
        yaxis=dict(
            title="Speed (km/h)",
            gridcolor="#EEEEEE",
            rangemode="tozero",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CCCCCC", borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="x unified",
        margin=dict(t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Scatter: {min(int(scatter_sample), n_pts):,} of {n_pts:,} points shown  ·  "
        f"Hourly aggregates computed from all {n_pts:,} filtered records  ·  "
        "Click a legend item to hide/show it; double-click to isolate."
    )

    # ── Downloads ─────────────────────────────────────────────────────────────
    # Build enriched dataset: every original filtered row with P10/P90/mean
    # joined on by hour so analysts can flag each observation vs. threshold.
    hour_lookup = hourly.set_index(hour_col)[["p10", "p90", "mean"]].rename(
        columns={
            "p10":  "p10_threshold_kmh",
            "p90":  "p90_upper_band_kmh",
            "mean": "hourly_mean_kmh",
        }
    )
    df_enriched = df_filt.copy()
    df_enriched = df_enriched.join(hour_lookup, on=hour_col)
    df_enriched["below_p10_threshold"] = (
        df_enriched[speed_col] < df_enriched["p10_threshold_kmh"]
    )

    with st.expander("Hourly P10 threshold values (summary table)"):
        display_hourly = hourly.rename(columns={
            hour_col: "Hour",
            "p10":    "P10 threshold (km/h)",
            "p90":    "P90 upper band (km/h)",
            "mean":   "Mean speed (km/h)",
            "count":  "Records",
        }).round(2)
        st.dataframe(display_hourly, use_container_width=True, hide_index=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇  Download enriched dataset (CSV)",
            data=df_enriched.to_csv(index=False),
            file_name="speed_data_with_thresholds.csv",
            mime="text/csv",
            help=(
                "Original filtered rows with p10_threshold_kmh, "
                "p90_upper_band_kmh, hourly_mean_kmh and below_p10_threshold "
                "columns appended — one row per speed observation."
            ),
        )
    with dl2:
        st.download_button(
            "⬇  Download hourly summary (CSV)",
            data=display_hourly.to_csv(index=False),
            file_name="hourly_thresholds.csv",
            mime="text/csv",
            help="24-row summary: P10 / P90 / mean / count per hour of day.",
        )

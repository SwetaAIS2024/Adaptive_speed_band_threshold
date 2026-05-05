# Adaptive Speed Band Threshold

An interactive Streamlit dashboard for exploring, clustering, and visualising adaptive speed thresholds on Singapore road segments using historical speed data.

---

## Overview

Traditional speed bands are static values. This project derives **adaptive, hour-of-day thresholds** (P10/P90) directly from historical speed observations, grouped by road name and sensor (LINK_ID). The dashboard lets analysts upload cleaned speed data, explore it, run K-Means clustering, and visualise per-hour congestion thresholds — without writing any code.

---

## Repository Structure

```
Adaptive_speed_band_threshold/
├── app/
│   ├── streamlit_app.py            # Dashboard entry point
│   └── tabs/
│       ├── context.py              # Shared app context & file discovery
│       ├── tab_eda.py              # Raw Data EDA tab
│       ├── tab_explorer.py         # Dataset Explorer tab
│       ├── tab_clustering.py       # Clustering tab
│       ├── tab_speed_bands.py      # Speed Bands tab
│       └── tab_threshold_viz.py    # Adaptive Threshold Visualisation tab
├── clustering/
│   │   ├── pipeline.py             # Shared pure computation functions
│   ├── scripts/
│   │   ├── feature_engineering.py  # Feature preparation for clustering
│   │   ├── run_clustering.py       # K-Means execution
│   │   └── extract_speed_bands.py  # P10/P90 band extraction
│   └── features/
│       └── feature_metadata.json
├── data/
│   ├── preprocess.py               # Cleans raw CSVs → pre_processed_dataset/
│   ├── data_understanding_EDA.py   # EDA utilities used across tabs
│   ├── raw_dataset/                # Place raw input CSVs here
│   └── pre_processed_dataset/      # Cleaned CSVs loaded by the dashboard
├── docs/
│   ├── PRD.md
│   └── pipeline_architecture.drawio
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Input Data

Place raw speed CSV files in `data/raw_dataset/`. The pipeline expects files named with either `weekday` or `weekend` in the filename.

**Expected columns:**

| Column | Type | Description |
|--------|------|-------------|
| `DATE_TIME` | datetime string | Timestamp of the speed observation |
| `LINK_ID` | int | Road segment sensor identifier |
| `SPEED` | float | Observed speed (km/h) |
| `ROAD_NAME` | str | Name of the road (e.g. `AYE`, `CTE TUNNEL`) |

After preprocessing, derived columns `hour` (0–23) and `day_type` (`Weekday` / `Weekend`) are added automatically.

---

## Preprocessing

Run once to clean all raw files and write them to `data/pre_processed_dataset/`:

```bash
python data/preprocess.py
```

What it does:
- Drops duplicate rows and rows with nulls in key columns
- Drops zero-speed records
- Parses `DATE_TIME` → adds `hour` and `date` columns
- Adds `day_type` derived from the filename keyword (`weekday` / `weekend`)
- Strips whitespace from string columns
- Normalises `ROAD_NAME` to uppercase

---

## Running the Dashboard

**Locally:**

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**With Docker:**

```bash
docker compose up -d
```

The dashboard is then available at `http://localhost:8501`.


for resetting the docker:
```
docker compose down  
docker compose build --no-cache
docker compose up -d 
```

---

## Dashboard Tabs

### Raw Data EDA
Automated exploratory analysis of any raw or preprocessed CSV — column types, null rates, speed distributions, and road-name breakdowns.

### Dataset Explorer
Interactive table view and column-level statistics for any loaded dataset.

### Clustering
Run K-Means clustering on the preprocessed data per `(ROAD_NAME × day_type)` subset. Optimal K is selected by silhouette score. Results include cluster assignments, centroids, and elbow/silhouette plots.

### Speed Bands
Visualise the P10/P90 speed bands derived from clustering — per cluster, per road category, and per hour of day.

### Adaptive Threshold Visualisation
The core analysis tab. Plots actual speed observations against hour of day and overlays per-hour adaptive thresholds.

**Filters (cascading):**
1. **Day type** — All / Weekday / Weekend
2. **ROAD_NAME** — multi-select; narrows the LINK_ID list to only sensors on the selected road(s)
3. **LINK_ID** — multi-select; automatically restricted to the IDs belonging to the chosen road(s)
4. **Max scatter points** — cap the number of plotted raw observations

**Visible traces (individually toggleable):**
- Actual speed points (scatter)
- P10 adaptive threshold (red line)
- P90 upper band (green dashed line)
- Hourly mean speed (orange dashed line)

**Downloads:**

| Button | Contents |
|--------|----------|
| Enriched dataset (CSV) | Every filtered row with `p10_threshold_kmh`, `p90_upper_band_kmh`, `hourly_mean_kmh`, and `below_p10_threshold` appended — ready for further analysis |
| Hourly summary (CSV) | 24-row table: P10 / P90 / mean / record count per hour |

The enriched dataset joins the hourly thresholds back onto each original observation, so analysts can immediately filter `below_p10_threshold == True` to identify congestion events.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scikit-learn`, `streamlit`, `plotly`, `pyarrow`, `openpyxl`

---

## References

- [Project Requirements Document](docs/PRD.md)
- [Pipeline Architecture](docs/pipeline_architecture.drawio)

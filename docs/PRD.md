# Project Requirements Document
## Adaptive Speed Band Threshold — Interactive Dashboard

---

## 1. Project Overview

This project derives **adaptive, hour-of-day speed thresholds** for Singapore road segments from historical speed observations and presents them through an interactive Streamlit dashboard.

Unlike static, hand-drawn speed bands, the thresholds produced here are **data-driven**: for each sensor (LINK_ID) or group of sensors on a road, the P10 and P90 of observed speeds are computed per hour of day, forming an empirical congestion envelope that varies with actual traffic conditions.

The dashboard covers the full workflow:
1. **Preprocessing** — clean raw CSVs and derive time features
2. **Exploration** — EDA and dataset inspection
3. **Clustering** — group sensors into traffic regimes via K-Means
4. **Threshold Visualisation** — plot per-hour adaptive thresholds against actual speed observations, with cascading filters and enriched CSV downloads

---

## 2. Input Data

### 2.1 Raw Dataset

Raw speed CSV files are placed in `data/raw_dataset/`. The filename must contain either `weekday` or `weekend` (case-insensitive) to enable automatic `day_type` derivation.

**Expected columns:**

| Column | Type | Description |
|--------|------|-------------|
| `DATE_TIME` | datetime string | Timestamp of the speed observation |
| `LINK_ID` | int | Road segment sensor identifier |
| `SPEED` | float | Observed speed (km/h) |
| `ROAD_NAME` | str | Road name (e.g. `AYE`, `CTE TUNNEL`) |

### 2.2 Preprocessing

`data/preprocess.py` reads all raw files, applies the following fixes, and writes one cleaned CSV per raw file to `data/pre_processed_dataset/`:

| Fix | Rule |
|-----|------|
| Duplicate rows | Dropped |
| Nulls in key columns | Rows dropped |
| Zero-speed records | Dropped |
| `DATE_TIME` parsing | Parsed to datetime; `hour` (0–23) and `date` columns derived |
| `day_type` | Added: `"Weekday"` or `"Weekend"` from filename keyword |
| String whitespace | Stripped from all string columns |
| `ROAD_NAME` casing | Normalised to uppercase |

### 2.3 Preprocessed Schema

| Column | Type | Description |
|--------|------|-------------|
| `DATE_TIME` | datetime | Parsed timestamp |
| `date` | date | Date part of `DATE_TIME` |
| `hour` | int | Hour of day (0–23) |
| `day_type` | str | `"Weekday"` or `"Weekend"` |
| `LINK_ID` | int | Sensor identifier |
| `SPEED` | float | Observed speed (km/h) |
| `ROAD_NAME` | str | Road name (uppercase) |

---

## 3. Dashboard

The dashboard is a multi-tab Streamlit application launched via:

```bash
streamlit run app/streamlit_app.py
# or
docker compose up -d   # available at http://localhost:8501
```

### 3.1 Tabs

| Tab | Purpose |
|-----|---------|
| **Raw Data EDA** | Automated exploratory analysis of any CSV — column types, null rates, speed distributions, road-name breakdowns |
| **Dataset Explorer** | Interactive table view and column-level statistics |
| **Clustering** | Run K-Means per `(ROAD_NAME × day_type)` subset; view elbow/silhouette plots and cluster assignments |
| **Speed Bands** | Visualise P10/P90 speed bands derived from clustering per road and hour |
| **Adaptive Threshold** | Core analysis tab — per-hour adaptive threshold visualisation with filters and downloads |

---

## 4. Adaptive Threshold Visualisation (Core Tab)

### 4.1 Purpose

Plot actual speed observations (scatter) against per-hour P10/P90/mean threshold lines for the selected road(s) and sensor(s). This reveals how the adaptive threshold moves with time of day rather than being a single flat value.

### 4.2 Data Loading

Users either select a preprocessed file from `data/pre_processed_dataset/` or upload any CSV directly. File contents are cached in session state to avoid repeated reads.

### 4.3 Filters

Filters apply in cascade order — selecting a road narrows the LINK_ID options to only those sensors on that road.

| Filter | Type | Behaviour |
|--------|------|-----------|
| **Day type** | Selectbox | All / Weekday / Weekend |
| **ROAD_NAME** | Multiselect | All road names in dataset; leave empty = all roads |
| **LINK_ID** | Multiselect | Restricted to IDs on selected road(s); leave empty = all IDs in pool |
| **Max scatter points** | Number input | Caps plotted raw observations (default 10 000) |

When the ROAD_NAME selection changes, the LINK_ID selection is automatically cleared to prevent cross-road mismatches.

### 4.4 Threshold Computation

Hourly aggregates are computed on the **filtered dataset** (`df_filt`):

$$
\text{P10}_h = P_{10}(\{speed_i : hour_i = h\})
$$
$$
\text{P90}_h = P_{90}(\{speed_i : hour_i = h\})
$$
$$
\text{mean}_h = \text{mean}(\{speed_i : hour_i = h\})
$$

If multiple LINK_IDs are selected, their speed observations are **pooled per hour** before computing the statistics — producing a single set of threshold lines representing the combined corridor. This is corridor-level analysis; for per-link comparison, filter to one LINK_ID at a time.

### 4.5 Chart Traces

Each trace is individually toggleable via the Plotly legend:

| Trace | Style | Key |
|-------|-------|-----|
| Actual speed points | Blue scatter | P10 threshold determines which points are flagged |
| P10 adaptive threshold | Red solid line | Lower operating boundary |
| P90 upper band | Green dashed line | Upper operating boundary |
| Hourly mean speed | Orange dashed line | Central tendency |

### 4.6 Downloads

Two download buttons are provided after the chart:

#### Enriched Dataset (primary download)
Every original filtered row with four new columns joined on by `hour`:

| Column | Description |
|--------|-------------|
| `p10_threshold_kmh` | P10 threshold for that row's hour |
| `p90_upper_band_kmh` | P90 for that row's hour |
| `hourly_mean_kmh` | Mean speed for that row's hour |
| `below_p10_threshold` | `True` if `SPEED < p10_threshold_kmh` |

This is the analytically useful output — analysts can filter `below_p10_threshold == True` to extract all congestion events, or join the enriched file with other datasets using `LINK_ID` + `hour`.

#### Hourly Summary
24-row table (one row per hour) with P10 / P90 / mean / record count. Useful as a quick reference or for reporting.

---

## 5. Clustering

### 5.1 Scope

Clustering is run independently for each `(ROAD_NAME × day_type)` subset within the loaded dataset. Optimal K is selected by silhouette score (sweep K = 2 … 10).

### 5.2 Algorithm

- **Primary**: K-Means (`n_init=10`, `random_state=42`)
- **K selection**: highest silhouette score in sweep
- **Minimum cluster size**: clusters with fewer than 30 records are merged into the nearest centroid
- **Quality flags**: clusters where speed std > 15 km/h or IQR > 20 km/h are flagged for review

### 5.3 Speed Band Extraction

For each cluster $c$, speed bands are derived as:

$$
\text{lower\_band}_c = P_{10}(S_c), \quad \text{upper\_band}_c = P_{90}(S_c)
$$

where $S_c$ is the set of speed observations assigned to cluster $c$.

Cluster results are written to `clustering/results/`.

---

## 6. Directory Structure

```
Adaptive_speed_band_threshold/
├── app/
│   ├── streamlit_app.py
│   └── tabs/
│       ├── context.py
│       ├── tab_eda.py
│       ├── tab_explorer.py
│       ├── tab_clustering.py
│       ├── tab_speed_bands.py
│       └── tab_threshold_viz.py
├── clustering/
│   ├── pipeline.py
│   ├── scripts/
│   │   ├── feature_engineering.py
│   │   ├── run_clustering.py
│   │   └── extract_speed_bands.py
│   └── results/
├── data/
│   ├── preprocess.py
│   ├── data_understanding_EDA.py
│   ├── raw_dataset/
│   └── pre_processed_dataset/
├── docs/
│   ├── PRD.md
│   └── pipeline_architecture.drawio
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 7. Technology Stack

| Component | Library |
|-----------|---------|
| Dashboard | `streamlit >= 1.32` |
| Visualisation | `plotly >= 5.20` |
| Data processing | `pandas >= 2.0`, `numpy >= 1.24` |
| Clustering | `scikit-learn >= 1.3` |
| File I/O | `pyarrow >= 14.0`, `openpyxl >= 3.1`, `xlrd >= 2.0` |
| Containerisation | Docker / docker-compose |

---

## 8. Execution Order

1. Place raw CSV files in `data/raw_dataset/` (filename must contain `weekday` or `weekend`)
2. Run `python data/preprocess.py` to generate cleaned files in `data/pre_processed_dataset/`
3. Launch the dashboard: `streamlit run app/streamlit_app.py` or `docker compose up -d`
4. In the **Adaptive Threshold** tab: select a preprocessed file → apply filters → view chart → download enriched dataset


---

## 1. Project Overview

This project derives **adaptive speed band thresholds** for Singapore road segments using a data-driven clustering approach.

**Starting point assumption:** A single clean, merged, hourly dataset is provided as input. Data from multiple sources has already been collected, processed, and merged. Abnormal records (sensor errors, extreme outliers, missing values) have already been filtered out upstream. This PRD covers everything from that clean dataset onwards.

The pipeline:
1. Takes the clean hourly input dataset
2. Clusters traffic records into interpretable **traffic regimes** per road category and day type — producing **category-level adaptive speed bands** shared across all LinkIDs within that functional road class
3. Extracts per-cluster **speed band thresholds** (lower / upper) as the adaptive output
4. Validates cluster distributions per LinkID to ensure derived bands remain representative across spatial segments within each category

The speed bands produced are adaptive in that they reflect actual operating conditions per functional road class, per hour, and per day type — rather than static, hand-drawn thresholds. This approach follows standard transportation engineering practice where capacity, geometry, signal density, and access control are category-dependent.

---

## 2. Input Dataset

### 2.1 Assumptions
- Data from all sources has been collected, merged, and cleaned prior to this pipeline
- Abnormal records have been filtered out (sensor noise, incidents, missing values)
- The dataset is aggregated to **hourly resolution** per road segment
- Timestamps are in **SGT** (Singapore Time, UTC+8)
- `RoadCategory` defines the functional road class (e.g. Expressway, Arterial). Each category contains multiple `LinkID`s. `LinkID` remains in the dataset but is **not** a clustering boundary — it serves as a post-cluster validation stratifier

### 2.2 Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_hour` | datetime (SGT) | Hour-truncated timestamp in Singapore Time |
| `LinkID` | int | LTA road segment identifier |
| `RoadCategory` | int | LTA road category (1=Expressway … 6=Minor Access) |
| `hour` | int | Hour of day (0–23) |
| `day_type` | str | `"Weekday"` or `"Weekend"` |
| `speed` | float | Average speed (km/h) for the hour |
| `volume` | int | Vehicle count (veh/hr) for the hour |

### 2.3 Input File Location
- `data/input/traffic_hourly.csv`

---

## 3. Feature Set for Clustering

The following features are used to construct the clustering feature vector:

| Feature | Column | Description |
|---------|--------|-------------|
| Time of day | `hour` | Integer hour (0–23) — encoded cyclically |
| Day type | `day_type` | `"Weekday"` or `"Weekend"` |
| Traffic Volume | `volume` | Vehicles per hour |
| Speed | `speed` | Average speed (km/h) |

---

## 4. Clustering Design

### 4.1 Scope

Clustering is run **independently** for each combination of:
- **Day type**: Weekday, Weekend
- **Road category**: 1 (Expressway), 2 (Major Arterial), 3 (Arterial), 4 (Minor Arterial), 5 (Local Access), 6 (Minor Access)

This gives up to **12 independent clustering runs** (2 day types × 6 road categories).

Each run produces **category-level speed band regimes** — thresholds that apply to all LinkIDs within that road category. The derived `lower_band` and `upper_band` are then **appended back to the original dataset**, so every record carries its cluster-assigned speed band. For example:

| Category | Day type | Hour slot | Derived band |
|----------|----------|-----------|-------------|
| Expressway | Weekday | 08:00–09:00 | 45–65 km/h |
| Expressway | Weekday | 22:00–23:00 | 70–90 km/h |
| Major Arterial | Weekend | 12:00–13:00 | 30–50 km/h |

Even though clustering is category-level, **temporal regimes emerge naturally** from the feature vector, since `hour`, `speed`, and `volume` are all included. This allows the algorithm to discover patterns such as morning peak, midday steady flow, and night free-flow automatically.

### 4.2 Feature Engineering for Clustering

The raw columns must be transformed into a numeric feature vector $x_i$ prior to clustering:

| Raw Feature | Encoding Method |
|-------------|----------------|
| `hour` | Cyclic encoding: $[\sin(2\pi h/24),\ \cos(2\pi h/24)]$ |
| `day_type` | Binary: Weekday=0, Weekend=1 |
| `speed` | Min-max normalised **within the current (road\_category × day\_type) clustering subset** |
| `volume` | Min-max normalised **within the current (road\_category × day\_type) clustering subset** |

Final feature vector per record:

$$
x_i = [\sin(h),\ \cos(h),\ \text{day\_bin},\ \text{speed\_norm},\ \text{volume\_norm}]
$$

> Normalisation is applied independently within each `(road_category × day_type)` subset, since clustering already runs per subset. This avoids scale contamination across road classes.

### 4.3 Algorithm

- **Primary**: K-Means (fast, scalable, compatible with medoid extraction)
- **Validation**: Silhouette score, Davies-Bouldin index
- **Number of clusters**: determined by elbow method or silhouette analysis per (road_cat, day_type) group
- **Fallback for non-convex distributions**: DBSCAN or GMM

### 4.4 Representative Point Extraction

For each cluster $c$, find the **medoid** — the actual record closest to the cluster centroid in transformed feature space:

$$
i^* = \arg\min_{i \in c} \|x_i - \mu_c\|_2
$$

The medoid row's raw values are used as the representative point:
- `rep_linkid`, `rep_day`, `rep_hour`, `rep_speed`, `rep_volume`

### 4.5 Speed Band Derivation

For all speeds $S_c = \{speed_i : i \in c\}$:

$$
\text{lower\_band}_c = P_{10}(S_c)
$$
$$
\text{upper\_band}_c = P_{90}(S_c)
$$

Do **not** use min/max — they are sensitive to noise and outliers.

Percentile bands represent the **empirical operating-speed envelope** of each traffic regime and approximate the conditional distribution of speed given temporal context and flow state.

### 4.6 Cluster Quality Criteria

A cluster is considered a valid speed-band regime only if:
- Cluster size $\geq$ 30 records
- Speed standard deviation $\leq$ 15 km/h
- Speed interquartile range (P75 - P25) $\leq$ 20 km/h
- Silhouette score $\geq$ 0.3

Clusters failing the size threshold (size < 30) are **merged with the nearest cluster by centroid distance** in transformed feature space, ensuring stable production behavior. Clusters failing the spread or silhouette thresholds are flagged for investigation (too broad → increase K).

### 4.7 Per-LinkID Post-Cluster Validation

After clustering, cluster speed distributions are evaluated **per LinkID** to ensure that derived category-level bands remain representative across all spatial segments within the category.

For each cluster $c$ and each LinkID $l$ present in $c$, compute:
- Mean and standard deviation of speeds for LinkID $l$ within cluster $c$
- Flag LinkID $l$ if its mean speed deviates from the cluster band midpoint by more than 15 km/h

Flagged links indicate spatial outliers within a cluster — segments where the category-level band may not be a good fit. These should be reviewed but do **not** affect the cluster band definition itself.

This keeps cluster boundaries stable while providing spatial representativeness diagnostics.

---

## 5. Output Dataset Schema

The final output is the **original input dataset with two additional columns appended** — `lower_band` and `upper_band` — derived from the cluster each record was assigned to. Every row in the dataset receives its category-level adaptive speed band.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_hour` | datetime (SGT) | Hour-truncated timestamp in Singapore Time |
| `LinkID` | int | LTA road segment identifier |
| `RoadCategory` | int | LTA road category (1=Expressway … 6=Minor Access) |
| `hour` | int | Hour of day (0–23) |
| `day_type` | str | `"Weekday"` or `"Weekend"` |
| `speed` | float | Average speed (km/h) for the hour |
| `volume` | int | Vehicle count (veh/hr) for the hour |
| `cluster_id` | str | Composite cluster identifier e.g. `"EXP_WD_03"` (category prefix + day type + local index) |
| `lower_band` | float | P10 speed of the assigned cluster (km/h) |
| `upper_band` | float | P90 speed of the assigned cluster (km/h) |

### Example Rows

| timestamp_hour | LinkID | RoadCategory | hour | day_type | speed | volume | cluster_id | lower_band | upper_band |
|---|---|---|---|---|---|---|---|---|---|
| 2026-03-26 08:00:00 | 1045 | 1 | 8 | Weekday | 47.2 | 1350 | EXP_WD_03 | 38.5 | 54.9 |
| 2026-03-26 08:00:00 | 1046 | 1 | 8 | Weekday | 51.0 | 1280 | EXP_WD_03 | 38.5 | 54.9 |
| 2026-03-26 22:00:00 | 1045 | 1 | 22 | Weekday | 82.3 | 410 | EXP_WD_07 | 70.0 | 90.0 |

All records in the same cluster share the same `lower_band` and `upper_band`.

### Output File Location
- `data/output/traffic_hourly_with_bands.csv`

---

## 6. Directory Structure

```
Adaptive_speed_band_threshold/
├── data/
│   ├── input/
│   │   └── traffic_hourly.csv                  # Clean merged hourly input dataset
│   └── output/
│       └── traffic_hourly_with_bands.csv       # Input dataset + cluster_id, lower_band, upper_band
├── clustering/
│   ├── scripts/
│   │   ├── feature_engineering.py              # Feature vector construction
│   │   ├── run_clustering.py                   # K-Means per (road_cat, day_type)
│   │   └── extract_speed_bands.py              # Medoid + P10/P90 band extraction + append to dataset
│   └── results/
│       └── cluster_summary_<cat>_<day>.csv     # One row per cluster (diagnostic)
└── docs/
    └── PRD.md
```

---

## 7. Technology Stack

| Component | Library |
|-----------|--------|
| Data processing | `pandas`, `numpy` |
| Clustering | `scikit-learn` (KMeans, DBSCAN, silhouette_score) |
| Evaluation | `scikit-learn` (silhouette_score, davies_bouldin_score) |
| Visualisation (optional) | `matplotlib`, `seaborn` |

---

## 8. Execution Order

1. Place the clean merged hourly dataset at `data/input/traffic_hourly.csv`
2. Run `clustering/scripts/feature_engineering.py` — build normalised feature vectors
3. Run `clustering/scripts/run_clustering.py` — cluster per (road_cat, day_type); save cluster summaries to `clustering/results/`
4. Run `clustering/scripts/extract_speed_bands.py` — extract P10/P90 bands per cluster and append `cluster_id`, `lower_band`, `upper_band` back to the original dataset; save to `data/output/traffic_hourly_with_bands.csv`

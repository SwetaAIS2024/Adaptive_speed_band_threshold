# Project Requirements Document
## Adaptive Speed Band Threshold — Data-Driven Clustering Pipeline

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

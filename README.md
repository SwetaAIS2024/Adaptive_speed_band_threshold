# Adaptive Speed Band Threshold

Data-driven adaptive speed band thresholds for Singapore road segments.  
Clusters historical hourly traffic records (speed + volume + time-of-day + day type) into interpretable traffic regimes per road category, then derives per-cluster P10/P90 speed bands.

---

## Overview

Traditional speed bands are static and hand-drawn. This pipeline learns them from real traffic data using K-Means clustering, producing thresholds that reflect actual operating conditions per functional road class, hour of day, and day type.

**Pipeline steps:**
1. **Data Collection** — poll LTA APIs for live speed snapshots and historical volume
2. **Feature Engineering** — encode time cyclically, normalise speed/volume per road category subset
3. **Clustering** — K-Means per (RoadCategory × day_type), optimal K selected by silhouette score
4. **Band Extraction** — P10/P90 speed bands, medoid per cluster, per-LinkID spatial validation

---

## Repository Structure

```
Adaptive_speed_band_threshold/
├── data_collection/
│   ├── speed_band/
│   │   └── dtm.py                          # Polls LTA TrafficSpeedBands every 5 min
│   └── traffic_volume/
│       └── collect_traffic_flow.py         # Downloads LTA TrafficFlow monthly JSON
├── clustering/
│   ├── scripts/
│   │   ├── feature_engineering.py          # Cyclic encoding + MinMax normalisation
│   │   ├── run_clustering.py               # K-Means per (road_cat × day_type)
│   │   └── extract_speed_bands.py          # P10/P90 bands + medoid + LinkID validation
│   └── results/                            # Generated — gitignored
│       ├── cluster_summary_<cat>_<day>.csv
│       ├── cluster_centers_<cat>_<day>.csv
│       ├── elbow_silhouette_<cat>_<day>.csv
│       ├── assignments_<cat>_<day>.parquet
│       ├── cluster_medoids.csv
│       └── linkid_validation.csv
├── data/
│   ├── input/
│   │   └── traffic_hourly.csv              # Clean merged hourly input (gitignored)
│   └── output/
│       └── traffic_hourly_with_bands.csv   # Final output (gitignored)
├── docs/
│   ├── PRD.md                              # Full project requirements document
│   └── pipeline_architecture.drawio        # Architecture diagram
└── requirements.txt
```

---

## Input Dataset

Place a clean hourly CSV at `data/input/traffic_hourly.csv` with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_hour` | datetime (SGT) | Hour-truncated timestamp in Singapore Time |
| `LinkID` | int | LTA road segment identifier |
| `RoadCategory` | int | Road category: 1=Expressway … 6=Minor Access |
| `hour` | int | Hour of day (0–23) |
| `day_type` | str | `"Weekday"` or `"Weekend"` |
| `speed` | float | Average speed (km/h) for the hour |
| `volume` | int | Vehicle count (veh/hr) for the hour |

---

## Data Collection

Set your LTA DataMall API key as an environment variable before running either script:

```powershell
$env:LTA_ACCOUNT_KEY = "your_api_key_here"
```

**Speed bands** (polls every 5 minutes for 72 hours, saves a new timestamped CSV per poll):
```bash
python data_collection/speed_band/dtm.py
```
Output: `data_collection/speed_band/raw_datasets/speedbands_YYYYMMDD_HHMMSS.csv`

**Traffic volume** (downloads one month of historical flow data):
```bash
python data_collection/traffic_volume/collect_traffic_flow.py
```
Output: `data_collection/traffic_volume/raw_datasets/traffic_flow_YYYYMMDD_HHMMSS.json`

> Raw data folders are gitignored. Pre-process and merge the raw outputs into `data/input/traffic_hourly.csv` before running the clustering pipeline.

---

## Clustering Pipeline

Run the three scripts in order from the repository root:

### Step 1 — Feature Engineering

```bash
python clustering/scripts/feature_engineering.py
```

For each `(RoadCategory × day_type)` subset:
- Cyclic-encodes `hour`: `sin(2πh/24)`, `cos(2πh/24)`
- Binary-encodes `day_type`: Weekday = 0, Weekend = 1
- Min-Max normalises `speed` and `volume` **within the subset only**
- Feature vector per record: `[sin_h, cos_h, day_bin, speed_norm, volume_norm]`

Reads from `synthetic_dataset/processed/synthetic_hourly_cat*.csv` (if present) or `data/input/traffic_hourly.csv`.  
Outputs: `clustering/features/features_<cat>_<day>.parquet` (×12) + `feature_metadata.json`

---

### Step 2 — Clustering

```bash
python clustering/scripts/run_clustering.py
```

For each subset:
- Sweeps K = 2 … 10, scores silhouette + Davies-Bouldin
- Selects K with the highest silhouette score
- Fits final K-Means (`n_init=10`, `random_state=42`)
- Merges clusters with fewer than 30 records into the nearest centroid
- Flags clusters where speed std > 15 km/h or IQR > 20 km/h
- Assigns composite `cluster_id` e.g. `"EXP_WD_03"` (category prefix + day type + local index)

**Road category prefixes:**

| Category | Prefix | Description |
|----------|--------|-------------|
| 1 | `EXP` | Expressway |
| 2 | `MJR` | Major Arterial |
| 3 | `ART` | Arterial |
| 4 | `MIN` | Minor Arterial |
| 5 | `LOC` | Local Access |
| 6 | `ACC` | Minor Access |

Outputs per subset: `assignments_<cat>_<day>.parquet`, `cluster_centers_<cat>_<day>.csv`, `elbow_silhouette_<cat>_<day>.csv`

---

### Step 3 — Speed Band Extraction

```bash
python clustering/scripts/extract_speed_bands.py
```

For each cluster:
- **`lower_band`** = P10 of cluster speeds (km/h)
- **`upper_band`** = P90 of cluster speeds (km/h)
- **Medoid** = actual record with minimum Euclidean distance to the cluster centroid in feature space
- **Per-LinkID validation** — flags any LinkID whose mean speed deviates > 15 km/h from the cluster band midpoint (spatial outlier diagnostic, does not change bands)
- Appends `cluster_id`, `lower_band`, `upper_band` to every row of the original dataset

Outputs:
- `data/output/traffic_hourly_with_bands.csv` — full dataset with adaptive bands
- `clustering/results/cluster_summary_<cat>_<day>.csv` — per-cluster P10/P90/mean/std/size
- `clustering/results/cluster_medoids.csv` — representative record per cluster
- `clustering/results/linkid_validation.csv` — per-LinkID deviation report

---

## Output Schema

Every row in `traffic_hourly_with_bands.csv` carries:

| Added Column | Type | Description |
|---|---|---|
| `cluster_id` | str | e.g. `"EXP_WD_03"` |
| `lower_band` | float | P10 speed of the assigned cluster (km/h) |
| `upper_band` | float | P90 speed of the assigned cluster (km/h) |

---

## Cluster Quality Criteria

A cluster is considered a valid speed-band regime only if:

| Criterion | Threshold |
|-----------|-----------|
| Cluster size | ≥ 30 records |
| Speed standard deviation | ≤ 15 km/h |
| Speed IQR (P75 − P25) | ≤ 20 km/h |
| Silhouette score | ≥ 0.3 |

Clusters failing the size threshold are automatically merged into the nearest cluster. Clusters failing spread thresholds are flagged in the output for review.

---

## Sample Results — Expressway Weekday (Cat 1)

Optimal K = 10, silhouette = 0.529

| Cluster | Lower (km/h) | Upper (km/h) | Mean | Std | Size | Regime |
|---------|-------------|-------------|------|-----|------|--------|
| EXP_WD_07 | 49.4 | 60.1 | 54.6 | 4.1 | 5,255 | Heavy congestion |
| EXP_WD_06 | 49.4 | 62.1 | 55.6 | 4.9 | 5,179 | Heavy congestion |
| EXP_WD_05 | 55.5 | 65.7 | 60.4 | 3.9 | 7,794 | Moderate congestion |
| EXP_WD_03 | 55.8 | 66.1 | 61.0 | 4.0 | 7,792 | Moderate congestion |
| EXP_WD_04 | 60.5 | 70.8 | 65.6 | 4.0 | 5,244 | Semi-free flow |
| EXP_WD_09 | 70.6 | 83.3 | 76.8 | 4.8 | 5,196 | Near free-flow |
| EXP_WD_01 | 72.5 | 87.4 | 79.7 | 5.7 | 5,215 | Near free-flow |
| EXP_WD_00 | 53.8 | 80.0 | 63.4 | 10.2 | 7,687 | Transition / shoulder |
| EXP_WD_02 | 79.8 | 90.0 | 85.2 | 3.9 | 7,794 | Free-flow |
| EXP_WD_08 | 81.3 | 90.0 | 86.3 | 3.5 | 5,196 | Free-flow |

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scikit-learn`, `pyarrow`

---

## References

- [LTA DataMall API](https://datamall.lta.gov.sg/content/datamall/en/dynamic-data.html)
- [Project Requirements Document](docs/PRD.md)
- [Pipeline Architecture](docs/pipeline_architecture.drawio)




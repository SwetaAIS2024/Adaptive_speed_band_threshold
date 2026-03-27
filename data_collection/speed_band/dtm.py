import requests
import time
import csv
import datetime
import os
import sys

# =========================
# CONFIGURATION
# =========================

ACCOUNT_KEY = os.environ.get("LTA_ACCOUNT_KEY", "")
if not ACCOUNT_KEY:
    print("ERROR: LTA_ACCOUNT_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

API_URL = "https://datamall2.mytransport.sg/ltaodataservice/v4/TrafficSpeedBands"

POLL_INTERVAL = 300         # seconds (5 minutes)
TOTAL_DURATION = 72 * 60 * 60  # 3 days in seconds

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw_datasets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUEST_TIMEOUT = 30        # seconds
MAX_RETRIES = 3             # retries per failed API page


# =========================
# CSV INITIALIZATION
# =========================

HEADERS = [
    "Timestamp",
    "LinkID",
    "RoadName",
    "RoadCategory",
    "SpeedBand",
    "MinimumSpeed",
    "MaximumSpeed",
    "StartLat",
    "StartLon",
    "EndLat",
    "EndLon"
]


# =========================
# DATA FETCH FUNCTION
# =========================

def fetch_all_pages():

    headers = {
        "AccountKey": ACCOUNT_KEY,
        "accept": "application/json"
    }

    all_records = []
    skip = 0

    while True:

        params = {"$skip": skip}
        data = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    API_URL, headers=headers, params=params,
                    timeout=REQUEST_TIMEOUT
                )
                if response.status_code == 200:
                    data = response.json().get("value", [])
                    break
                print(f"  API error (attempt {attempt}/{MAX_RETRIES}): HTTP {response.status_code}")
            except requests.exceptions.RequestException as exc:
                print(f"  Request failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)   # back-off: 5s, 10s

        if data is None:
            print("  Skipping page after all retries failed.")
            break

        if not data:
            break

        all_records.extend(data)
        skip += 500   # pagination step

    return all_records


# =========================
# MAIN COLLECTION LOOP
# =========================

start_time = time.time()

print("Starting 3-day data collection...")

while (time.time() - start_time) < TOTAL_DURATION:

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    timestamp_iso = timestamp.isoformat()
    file_ts = timestamp.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"speedbands_{file_ts}.csv")

    try:

        records = fetch_all_pages()

        print(f"{timestamp_iso} - {len(records)} records collected -> {os.path.basename(output_file)}")

        with open(output_file, "w", newline="", encoding="utf-8") as csv_file:

            writer = csv.writer(csv_file)
            writer.writerow(HEADERS)

            for r in records:

                writer.writerow([
                    timestamp_iso,
                    r.get("LinkID"),
                    r.get("RoadName"),
                    r.get("RoadCategory"),
                    r.get("SpeedBand"),
                    r.get("MinimumSpeed"),
                    r.get("MaximumSpeed"),
                    r.get("StartLat"),
                    r.get("StartLon"),
                    r.get("EndLat"),
                    r.get("EndLon")
                ])

    except Exception as e:
        print(f"Collection error: {e}")

    time.sleep(POLL_INTERVAL)

print("Collection completed successfully.")
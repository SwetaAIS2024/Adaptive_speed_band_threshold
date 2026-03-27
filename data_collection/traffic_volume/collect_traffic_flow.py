import requests
import datetime
import os
import sys

# ======================
# CONFIG
# ======================

ACCOUNT_KEY = os.environ.get("LTA_ACCOUNT_KEY", "")
if not ACCOUNT_KEY:
    print("ERROR: LTA_ACCOUNT_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

API_URL     = "https://datamall2.mytransport.sg/ltaodataservice/TrafficFlow"
SAVE_FOLDER = os.path.join(os.path.dirname(__file__), "raw_datasets")
os.makedirs(SAVE_FOLDER, exist_ok=True)

REQUEST_TIMEOUT = 30    # seconds
MAX_RETRIES     = 3


# ======================
# STEP 1: GET DOWNLOAD LINK
# ======================

def get_download_link() -> str:

    headers = {
        "AccountKey": ACCOUNT_KEY,
        "accept": "application/json"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(API_URL, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                value = response.json().get("value", [])
                if not value or "Link" not in value[0]:
                    raise ValueError("Unexpected response structure — 'Link' key missing.")
                return value[0]["Link"]

            print(f"  API error (attempt {attempt}/{MAX_RETRIES}): HTTP {response.status_code}")

        except (requests.exceptions.RequestException, ValueError) as exc:
            print(f"  Request failed (attempt {attempt}/{MAX_RETRIES}): {exc}")

        if attempt < MAX_RETRIES:
            import time
            time.sleep(5 * attempt)

    raise RuntimeError("Failed to retrieve download link after all retries.")


# ======================
# STEP 2: DOWNLOAD DATASET
# ======================

def download_dataset(download_url: str) -> str:

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(SAVE_FOLDER, f"traffic_flow_{timestamp}.json")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(download_url, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                return filename

            print(f"  Download error (attempt {attempt}/{MAX_RETRIES}): HTTP {response.status_code}")

        except requests.exceptions.RequestException as exc:
            print(f"  Download failed (attempt {attempt}/{MAX_RETRIES}): {exc}")

        if attempt < MAX_RETRIES:
            import time
            time.sleep(5 * attempt)

    raise RuntimeError("Download failed after all retries — link may have expired.")


# ======================
# MAIN EXECUTION
# ======================

try:

    print("Fetching signed download URL...")
    link = get_download_link()

    print(f"  Link: {link}")
    print("Downloading dataset...")
    saved_file = download_dataset(link)

    print(f"Saved to: {saved_file}")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

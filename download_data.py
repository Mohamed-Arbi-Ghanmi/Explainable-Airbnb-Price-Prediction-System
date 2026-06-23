"""
Download the Barcelona Airbnb listings dataset from Inside Airbnb.
Run once before launching the app: python download_data.py

Steps:
  1. Go to https://insideairbnb.com/get-the-data/
  2. Find "Barcelona, Catalonia, Spain" and copy the link for listings.csv.gz
  3. Paste it as the URL variable below, then run this script.
"""

import gzip
import shutil
import urllib.request
from pathlib import Path

# Paste the listings.csv.gz URL from https://insideairbnb.com/get-the-data/
URL = ""

GZ_PATH = Path("listings.csv.gz")
CSV_PATH = Path("listings.csv")


def download():
    if not URL:
        raise SystemExit(
            "Set the URL variable first.\n"
            "Get it from: https://insideairbnb.com/get-the-data/ → Barcelona → listings.csv.gz"
        )

    if CSV_PATH.exists():
        print(f"{CSV_PATH} already exists, skipping download.")
        return

    print(f"Downloading from Inside Airbnb ...")
    urllib.request.urlretrieve(URL, GZ_PATH)

    print("Decompressing ...")
    with gzip.open(GZ_PATH, "rb") as f_in, open(CSV_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    GZ_PATH.unlink()
    print(f"Done — {CSV_PATH} is ready. You can now run: streamlit run app.py")


if __name__ == "__main__":
    download()

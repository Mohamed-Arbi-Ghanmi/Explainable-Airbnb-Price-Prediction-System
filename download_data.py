"""
Download the Barcelona Airbnb listings dataset from Inside Airbnb.
Run once before launching the app: python download_data.py

The URL below points to the March 2026 Barcelona scrape.
The model was originally trained on the September 2024 scrape — column structure
should be identical, but median prices and listing counts will differ.

To use a different scrape date, replace the URL with any listings.csv.gz link
from https://insideairbnb.com/get-the-data/ under Barcelona.
"""

import gzip
import shutil
import urllib.request
from pathlib import Path

URL = "https://data.insideairbnb.com/spain/catalonia/barcelona/2026-03-21/data/listings.csv.gz"

GZ_PATH = Path("listings.csv.gz")
CSV_PATH = Path("listings.csv")


def download():
    if CSV_PATH.exists():
        print(f"{CSV_PATH} already exists, skipping download.")
        return

    print(f"Downloading Barcelona listings from Inside Airbnb ...")
    urllib.request.urlretrieve(URL, GZ_PATH)

    print("Decompressing ...")
    with gzip.open(GZ_PATH, "rb") as f_in, open(CSV_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    GZ_PATH.unlink()
    print(f"Done — {CSV_PATH} is ready. You can now run: streamlit run app.py")


if __name__ == "__main__":
    download()

"""
Download and preprocess the Inside Airbnb Amsterdam dataset.
Run once before starting the app: python scripts/download_data.py
"""

import os, re, urllib.request, gzip, shutil
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(RAW_DIR, exist_ok=True)

URL = "https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2025-09-11/data/listings.csv.gz"
GZ_PATH  = os.path.join(RAW_DIR, "listings.csv.gz")
CSV_PATH = os.path.join(RAW_DIR, "listings.csv")

if not os.path.exists(CSV_PATH):
    print("Downloading listings.csv.gz …")
    urllib.request.urlretrieve(URL, GZ_PATH)
    with gzip.open(GZ_PATH, "rb") as f_in, open(CSV_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(GZ_PATH)
    print(f"Saved to {CSV_PATH}")
else:
    print("listings.csv already present, skipping download.")

LOOKUP_PATH = os.path.join(DATA_DIR, "lookup.csv")
if not os.path.exists(LOOKUP_PATH):
    print("Building lookup.csv …")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    def parse_bathrooms(s):
        if pd.isna(s): return np.nan
        s = str(s).lower()
        if "half" in s: return 0.5
        nums = re.findall(r"\d+\.?\d*", s)
        return float(nums[0]) if nums else np.nan

    def count_amenities(s):
        return len(re.findall(r'"[^"]+"', str(s))) if not pd.isna(s) else 0

    SCRAPE_DATE = pd.Timestamp("2025-09-11")
    df["host_since_dt"] = pd.to_datetime(df["host_since"], errors="coerce")
    df["host_years"] = (SCRAPE_DATE - df["host_since_dt"]).dt.days / 365.25

    lookup = pd.DataFrame({
        "id":                   df["id"],
        "listing_url":          df["listing_url"],
        "name":                 df["name"],
        "price_numeric":        df["price"].str.replace(r"[$,]", "", regex=True).astype(float),
        "latitude":             df["latitude"],
        "longitude":            df["longitude"],
        "neighbourhood_cleansed": df["neighbourhood_cleansed"],
        "room_type":            df["room_type"],
        "property_type":        df["property_type"],
        "accommodates":         df["accommodates"],
        "bedrooms":             df["bedrooms"],
        "beds":                 df["beds"],
        "bathrooms":            df["bathrooms_text"].apply(parse_bathrooms),
        "minimum_nights":       df["minimum_nights"],
        "maximum_nights":       df["maximum_nights"],
        "number_of_reviews":    df["number_of_reviews"],
        "number_of_reviews_ltm":df["number_of_reviews_ltm"],
        "reviews_per_month":    df["reviews_per_month"].fillna(0),
        "review_scores_rating":      df["review_scores_rating"],
        "review_scores_cleanliness": df["review_scores_cleanliness"],
        "review_scores_location":    df["review_scores_location"],
        "host_years":           df["host_years"],
        "host_response_rate":   df["host_response_rate"].str.replace("%","").astype(float),
        "host_acceptance_rate": df["host_acceptance_rate"].str.replace("%","").astype(float),
        "host_response_time":   df["host_response_time"],
        "host_is_superhost":    df["host_is_superhost"].map({"t":1,"f":0}),
        "host_identity_verified": df["host_identity_verified"].map({"t":1,"f":0}),
        "instant_bookable":     df["instant_bookable"].map({"t":1,"f":0}),
        "calculated_host_listings_count": df["calculated_host_listings_count"],
        "amenity_count":        df["amenities"].apply(count_amenities),
        "availability_365":     df["availability_365"],
        "low_availability":     (df["availability_365"] <= 90).astype(int),
    })
    lookup.to_csv(LOOKUP_PATH, index=False)
    print(f"lookup.csv saved ({len(lookup):,} rows)")
else:
    print("lookup.csv already present, skipping build.")

print("Done.")

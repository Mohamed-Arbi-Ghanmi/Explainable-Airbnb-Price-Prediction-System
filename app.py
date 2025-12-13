import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import re

st.set_page_config(page_title="Airbnb Barcelona Price Predictor", layout="wide")


# -------------------------
# Utils (same logic as notebook)
# -------------------------
def percent_to_float(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace("%", "", regex=False)
         .replace("nan", np.nan)
         .astype(float)
    )

def parse_bathrooms_text(x) -> float:
    if pd.isna(x):
        return np.nan
    x = str(x).lower()
    m = re.search(r"(\d+(\.\d+)?)", x)
    return float(m.group(1)) if m else np.nan

def amenities_count(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x)
    return s.count('"')

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

CATALUNYA = (41.3870, 2.1700)
BARCELONETA = (41.3780, 2.1920)

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "bathrooms_text" in df.columns:
        df["bathrooms_num"] = df["bathrooms_text"].apply(parse_bathrooms_text)
    if "amenities" in df.columns:
        df["amenities_count"] = df["amenities"].apply(amenities_count)

    if "host_response_rate" in df.columns:
        df["host_response_rate_num"] = percent_to_float(df["host_response_rate"])
    if "host_acceptance_rate" in df.columns:
        df["host_acceptance_rate_num"] = percent_to_float(df["host_acceptance_rate"])

    if "host_since" in df.columns:
        hs = pd.to_datetime(df["host_since"], errors="coerce")
        ref = pd.Timestamp.today(tz=None).normalize()
        df["host_tenure_days"] = (ref - hs).dt.days

    if "latitude" in df.columns and "longitude" in df.columns:
        df["dist_to_center_km"] = haversine_km(df["latitude"], df["longitude"], *CATALUNYA)
        df["dist_to_beach_km"]  = haversine_km(df["latitude"], df["longitude"], *BARCELONETA)

    return df


# -------------------------
# Load assets
# -------------------------
@st.cache_data
def load_data(path="listings.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path="airbnb_barcelona_price_model.joblib"):
    return joblib.load(path)

@st.cache_data
def load_meta(path="airbnb_barcelona_model_metadata.json"):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_importance(path="permutation_importance.csv"):
    try:
        imp = pd.read_csv(path, index_col=0)
        return imp
    except Exception:
        return None


st.title("Airbnb Barcelona — Nightly Price Predictor")
st.caption("Predict nightly price from listing features + show market patterns on a map.")

df = load_data("listings.csv")  # change to listings.csv.gz if needed
meta = load_meta()
model = load_model()
imp_df = load_importance()

df_feat = engineer_features(df)

# Build neighborhood -> representative coordinates (median)
nb_centers = (
    df_feat.dropna(subset=["neighbourhood_cleansed","latitude","longitude"])
          .groupby("neighbourhood_cleansed")[["latitude","longitude"]]
          .median()
)

# Options
neigh_options = sorted(nb_centers.index.tolist())
room_options = sorted(df_feat["room_type"].dropna().unique().tolist()) if "room_type" in df_feat.columns else []
prop_options = sorted(df_feat["property_type"].dropna().unique().tolist()) if "property_type" in df_feat.columns else []
resp_time_options = sorted(df_feat["host_response_time"].dropna().unique().tolist()) if "host_response_time" in df_feat.columns else []


# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("Listing inputs")

neigh = st.sidebar.selectbox("Neighbourhood", neigh_options)
room_type = st.sidebar.selectbox("Room type", room_options)
property_type = st.sidebar.selectbox("Property type", prop_options)

accommodates = st.sidebar.slider("Accommodates", 1, 16, 2)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 1)
beds = st.sidebar.slider("Beds", 0, 16, 1)
bathrooms_num = st.sidebar.slider("Bathrooms", 0.0, 6.0, 1.0, step=0.5)

minimum_nights = st.sidebar.slider("Minimum nights", 1, 365, 2)
maximum_nights = st.sidebar.slider("Maximum nights", 1, 365, 30)

amenities_count_val = st.sidebar.slider("Amenities count (approx.)", 0, 150, 40)
review_score = st.sidebar.slider("Review score rating", 0.0, 100.0, 90.0)
num_reviews = st.sidebar.slider("Number of reviews", 0, 2000, 20)
reviews_per_month = st.sidebar.slider("Reviews per month", 0.0, 30.0, 1.0)

availability_30 = st.sidebar.slider("Availability (30 days)", 0, 30, 10)
availability_60 = st.sidebar.slider("Availability (60 days)", 0, 60, 20)
availability_90 = st.sidebar.slider("Availability (90 days)", 0, 90, 30)
availability_365 = st.sidebar.slider("Availability (365 days)", 0, 365, 180)

host_listings_count = st.sidebar.slider("Host listings count", 1, 50, 1)
host_tenure_days = st.sidebar.slider("Host tenure (days)", 0, 6000, 800)

host_response_rate_num = st.sidebar.slider("Host response rate (%)", 0, 100, 90)
host_acceptance_rate_num = st.sidebar.slider("Host acceptance rate (%)", 0, 100, 90)

instant_bookable = st.sidebar.selectbox("Instant bookable", ["t", "f"])
has_availability = st.sidebar.selectbox("Has availability", ["t", "f"])
host_is_superhost = st.sidebar.selectbox("Host is superhost", ["t", "f"])
host_identity_verified = st.sidebar.selectbox("Host identity verified", ["t", "f"])
host_has_profile_pic = st.sidebar.selectbox("Host has profile pic", ["t", "f"])

host_response_time = st.sidebar.selectbox("Host response time", resp_time_options) if resp_time_options else None


# Use neighborhood median coordinates
lat, lon = nb_centers.loc[neigh, ["latitude","longitude"]].tolist()

# Create single-row dataframe for prediction (must match FEATURES used during training)
row = {
    "neighbourhood_cleansed": neigh,
    "room_type": room_type,
    "property_type": property_type,
    "latitude": lat,
    "longitude": lon,

    "accommodates": accommodates,
    "bedrooms": bedrooms,
    "beds": beds,
    "bathrooms_num": bathrooms_num,

    "minimum_nights": minimum_nights,
    "maximum_nights": maximum_nights,

    "availability_30": availability_30,
    "availability_60": availability_60,
    "availability_90": availability_90,
    "availability_365": availability_365,

    "number_of_reviews": num_reviews,
    "reviews_per_month": reviews_per_month,
    "review_scores_rating": review_score,

    "amenities_count": amenities_count_val,

    "host_listings_count": host_listings_count,
    "host_tenure_days": host_tenure_days,
    "host_response_rate_num": host_response_rate_num,
    "host_acceptance_rate_num": host_acceptance_rate_num,

    "instant_bookable": instant_bookable,
    "has_availability": has_availability,
    "host_is_superhost": host_is_superhost,
    "host_identity_verified": host_identity_verified,
    "host_has_profile_pic": host_has_profile_pic,
}
if host_response_time is not None:
    row["host_response_time"] = host_response_time

X_one = pd.DataFrame([row])

# Some models expect engineered distance features: add them
X_one = engineer_features(X_one)

# Keep only expected features (from metadata)
X_one = X_one[meta["features"]]


# -------------------------
# Predict
# -------------------------
pred_log = model.predict(X_one)[0]
pred_price = np.expm1(pred_log)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")
    st.metric("Predicted nightly price", f"€{pred_price:,.2f}")

    actual_price = st.number_input("Optional: enter actual host price (€) to compare", min_value=0.0, value=0.0, step=5.0)
    if actual_price > 0:
        pct_diff = 100 * (actual_price - pred_price) / max(pred_price, 1e-6)
        if pct_diff > 30:
            st.error(f"Overpriced vs model (+{pct_diff:.1f}%)")
        elif pct_diff < -30:
            st.warning(f"Underpriced vs model ({pct_diff:.1f}%)")
        else:
            st.success(f"Fairly priced ({pct_diff:.1f}%)")

with col2:
    st.subheader("Top global drivers (from your model)")
    if imp_df is not None and "importance_pct" in imp_df.columns:
        st.dataframe(imp_df.sort_values("importance_pct", ascending=False).head(10))
    else:
        st.info("To show drivers here, export permutation_importance.csv from the notebook.")


# -------------------------
# Maps
# -------------------------
st.subheader("Barcelona price map (sample of listings)")

# sample for performance
map_df = df_feat.dropna(subset=["latitude","longitude","price"]).copy()
map_df["price"] = pd.to_numeric(map_df["price"].astype(str).str.replace(r"[\€, \$]", "", regex=True).str.replace(",", "", regex=False), errors="coerce")
map_df = map_df.dropna(subset=["price"])
map_df = map_df.sample(min(2500, len(map_df)), random_state=42)

m = folium.Map(location=[map_df["latitude"].mean(), map_df["longitude"].mean()], zoom_start=12, tiles="OpenStreetMap")

# Clip for visualization
map_df["price_clip"] = map_df["price"].clip(upper=300)

for lat_i, lon_i, p_i in zip(map_df["latitude"], map_df["longitude"], map_df["price_clip"]):
    folium.CircleMarker(
        location=[lat_i, lon_i],
        radius=3,
        fill=True,
        fill_opacity=0.5,
        popup=f"€{p_i:.0f}/night (clipped)",
    ).add_to(m)

# Add predicted point
folium.Marker(
    [lat, lon],
    popup=f"Your listing prediction: €{pred_price:.0f}/night",
    icon=folium.Icon(color="red")
).add_to(m)

st_folium(m, width=1000, height=550)

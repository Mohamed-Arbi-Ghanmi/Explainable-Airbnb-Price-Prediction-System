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
# Utils
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

def clean_price_to_float(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[\€, \$]", "", regex=True)
         .str.replace(",", "", regex=False)
         .replace("nan", np.nan)
    )

def safe_median(series: pd.Series, fallback: float):
    try:
        v = pd.to_numeric(series, errors="coerce").dropna()
        return float(v.median()) if len(v) else float(fallback)
    except Exception:
        return float(fallback)

# -------------------------
# Local explanation fallback (sensitivity)
# -------------------------
def local_sensitivity(model, X_one: pd.DataFrame, features: list) -> pd.DataFrame:
    base_log = float(model.predict(X_one)[0])
    base_price = float(np.expm1(base_log))

    effects = []
    for f in features:
        if f not in X_one.columns:
            continue
        if pd.api.types.is_numeric_dtype(X_one[f]):
            x2 = X_one.copy()
            x = float(x2[f].iloc[0]) if pd.notna(x2[f].iloc[0]) else 0.0
            delta = max(abs(x) * 0.10, 1.0)  # +10% or +1
            x2[f] = x + delta
            p2 = float(np.expm1(model.predict(x2)[0]))
            effects.append((f, p2 - base_price))

    out = pd.DataFrame(effects, columns=["feature", "delta_price_eur"])
    if out.empty:
        return out

    out["abs_delta"] = out["delta_price_eur"].abs()
    out = out.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"])
    total = out["delta_price_eur"].abs().sum()
    out["contribution_pct"] = 100 * out["delta_price_eur"].abs() / (total if total != 0 else 1.0)
    return out

# -------------------------
# SHAP local explanation (preferred)
# -------------------------
@st.cache_resource
def get_shap_explainer(pipeline_model, background_df: pd.DataFrame):
    """
    Cache a SHAP explainer. If SHAP fails, this will raise and we'll fallback.
    Uses shap.Explainer (auto) on pipeline_model.predict.
    """
    import shap  # only import if available
    # shap.Explainer can work with callable predict + background
    explainer = shap.Explainer(pipeline_model.predict, background_df)
    return explainer

def try_shap_explain(model, X_one: pd.DataFrame, background_df: pd.DataFrame):
    """
    Returns (ok, fig_or_none, error_or_none)
    """
    try:
        import shap
        explainer = get_shap_explainer(model, background_df)
        sv = explainer(X_one)

        # Waterfall plot -> matplotlib figure
        fig = shap.plots.waterfall(sv[0], show=False)
        return True, fig, None
    except Exception as e:
        return False, None, str(e)

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

# -------------------------
# Header
# -------------------------
st.title("Airbnb Barcelona — Nightly Price Predictor")
st.caption("Predict nightly price, explore city price patterns, and explain each prediction.")

# -------------------------
# Load everything
# -------------------------
df = load_data("listings.csv")  # adjust if gz
meta = load_meta()
model = load_model()
imp_df = load_importance()

df_feat = engineer_features(df)

# Parse price for map/analysis
if "price" in df_feat.columns:
    df_feat["price_num"] = pd.to_numeric(clean_price_to_float(df_feat["price"]), errors="coerce")

# Neighborhood centers
nb_centers = (
    df_feat.dropna(subset=["neighbourhood_cleansed", "latitude", "longitude"])
          .groupby("neighbourhood_cleansed")[["latitude", "longitude"]]
          .median()
)

neigh_options = sorted(nb_centers.index.tolist())
room_options = sorted(df_feat["room_type"].dropna().unique().tolist()) if "room_type" in df_feat.columns else []
prop_options = sorted(df_feat["property_type"].dropna().unique().tolist()) if "property_type" in df_feat.columns else []
resp_time_options = sorted(df_feat["host_response_time"].dropna().unique().tolist()) if "host_response_time" in df_feat.columns else []

# Tabs
tab_predict, tab_map, tab_analysis = st.tabs(["Predict", "Map", "Analysis"])

# -------------------------
# Sidebar: Mode & Inputs
# -------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode", ["New listing (host)", "Existing listing (from dataset)"])

X_one = None
selected_lat, selected_lon = None, None
pred_price = None

if mode == "Existing listing (from dataset)":
    st.sidebar.header("Select existing listing")

    neigh = st.sidebar.selectbox("Neighbourhood", neigh_options)
    subset = df_feat[df_feat["neighbourhood_cleansed"] == neigh].copy()

    if "room_type" in subset.columns and len(room_options) > 0:
        rt = st.sidebar.selectbox("Room type filter", sorted(subset["room_type"].dropna().unique()))
        subset = subset[subset["room_type"] == rt]

    if "property_type" in subset.columns and len(prop_options) > 0:
        pt = st.sidebar.selectbox("Property type filter", sorted(subset["property_type"].dropna().unique()))
        subset = subset[subset["property_type"] == pt]

    subset = subset.dropna(subset=["latitude", "longitude"])
    candidates = subset.index.tolist()

    if len(candidates) == 0:
        st.sidebar.error("No listings found for this selection.")
    else:
        idx = st.sidebar.selectbox("Pick a listing (index)", candidates[:500])
        base_row = subset.loc[idx].to_dict()

        selected_lat = float(base_row.get("latitude", nb_centers.loc[neigh, "latitude"]))
        selected_lon = float(base_row.get("longitude", nb_centers.loc[neigh, "longitude"]))

        X_one = pd.DataFrame([base_row])
        X_one = engineer_features(X_one)
        X_one = X_one.reindex(columns=meta["features"], fill_value=np.nan)

else:
    st.sidebar.header("New listing inputs")

    neigh = st.sidebar.selectbox("Neighbourhood", neigh_options)
    selected_lat, selected_lon = nb_centers.loc[neigh, ["latitude", "longitude"]].tolist()

    room_type = st.sidebar.selectbox("Room type", room_options)
    property_type = st.sidebar.selectbox("Property type", prop_options)

    accommodates = st.sidebar.slider("Accommodates", 1, 16, 2)
    bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 1)
    beds = st.sidebar.slider("Beds", 0, 16, 1)
    bathrooms_num = st.sidebar.slider("Bathrooms", 0.0, 6.0, 1.0, step=0.5)
    minimum_nights = st.sidebar.slider("Minimum nights", 1, 365, 2)
    maximum_nights = st.sidebar.slider("Maximum nights", 1, 365, 30)
    amenities_count_val = st.sidebar.slider("Amenities count (approx.)", 0, 150, 40)

    nb_df = df_feat[df_feat["neighbourhood_cleansed"] == neigh].copy()

    defaults = {
        "number_of_reviews": safe_median(nb_df.get("number_of_reviews", pd.Series(dtype=float)), 0),
        "reviews_per_month": safe_median(nb_df.get("reviews_per_month", pd.Series(dtype=float)), 0),
        "review_scores_rating": safe_median(nb_df.get("review_scores_rating", pd.Series(dtype=float)), 90),

        "availability_30": safe_median(nb_df.get("availability_30", pd.Series(dtype=float)), 10),
        "availability_60": safe_median(nb_df.get("availability_60", pd.Series(dtype=float)), 20),
        "availability_90": safe_median(nb_df.get("availability_90", pd.Series(dtype=float)), 30),
        "availability_365": safe_median(nb_df.get("availability_365", pd.Series(dtype=float)), 180),

        "host_listings_count": safe_median(nb_df.get("host_listings_count", pd.Series(dtype=float)), 1),
        "host_tenure_days": safe_median(nb_df.get("host_tenure_days", pd.Series(dtype=float)), 800),
        "host_response_rate_num": safe_median(nb_df.get("host_response_rate_num", pd.Series(dtype=float)), 90),
        "host_acceptance_rate_num": safe_median(nb_df.get("host_acceptance_rate_num", pd.Series(dtype=float)), 90),
    }

    with st.sidebar.expander("Advanced (optional)", expanded=False):
        instant_bookable = st.sidebar.selectbox("Instant bookable", ["t", "f"])
        host_is_superhost = st.sidebar.selectbox("Host is superhost", ["t", "f"])
        host_identity_verified = st.sidebar.selectbox("Host identity verified", ["t", "f"])
        host_has_profile_pic = st.sidebar.selectbox("Host has profile pic", ["t", "f"])
        has_availability = st.sidebar.selectbox("Has availability", ["t", "f"])
        host_response_time = st.sidebar.selectbox("Host response time", resp_time_options) if resp_time_options else None

    # If expander not opened, define defaults
    if "instant_bookable" not in locals():
        instant_bookable = "t"
        host_is_superhost = "f"
        host_identity_verified = "t"
        host_has_profile_pic = "t"
        has_availability = "t"
        host_response_time = None

    row = {
        "neighbourhood_cleansed": neigh,
        "room_type": room_type,
        "property_type": property_type,
        "latitude": float(selected_lat),
        "longitude": float(selected_lon),

        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "bathrooms_num": bathrooms_num,
        "minimum_nights": minimum_nights,
        "maximum_nights": maximum_nights,
        "amenities_count": amenities_count_val,

        # auto-filled (not user-entered)
        "number_of_reviews": defaults["number_of_reviews"],
        "reviews_per_month": defaults["reviews_per_month"],
        "review_scores_rating": defaults["review_scores_rating"],

        "availability_30": defaults["availability_30"],
        "availability_60": defaults["availability_60"],
        "availability_90": defaults["availability_90"],
        "availability_365": defaults["availability_365"],

        "host_listings_count": defaults["host_listings_count"],
        "host_tenure_days": defaults["host_tenure_days"],
        "host_response_rate_num": defaults["host_response_rate_num"],
        "host_acceptance_rate_num": defaults["host_acceptance_rate_num"],

        "instant_bookable": instant_bookable,
        "has_availability": has_availability,
        "host_is_superhost": host_is_superhost,
        "host_identity_verified": host_identity_verified,
        "host_has_profile_pic": host_has_profile_pic,
    }
    if host_response_time is not None:
        row["host_response_time"] = host_response_time

    X_one = pd.DataFrame([row])
    X_one = engineer_features(X_one)
    X_one = X_one.reindex(columns=meta["features"], fill_value=np.nan)

# -------------------------
# Predict tab
# -------------------------
with tab_predict:
    if X_one is None or X_one.empty:
        st.warning("Select inputs in the sidebar to get a prediction.")
    else:
        pred_log = float(model.predict(X_one)[0])
        pred_price = float(np.expm1(pred_log))

        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Prediction")
            st.metric("Predicted nightly price", f"€{pred_price:,.2f}")

            actual_price = st.number_input(
                "Optional: enter an actual host price (€) to compare",
                min_value=0.0, value=0.0, step=5.0
            )
            if actual_price > 0:
                pct_diff = 100 * (actual_price - pred_price) / max(pred_price, 1e-6)
                if pct_diff > 30:
                    st.error(f"Overpriced vs model (+{pct_diff:.1f}%)")
                elif pct_diff < -30:
                    st.warning(f"Underpriced vs model ({pct_diff:.1f}%)")
                else:
                    st.success(f"Fairly priced ({pct_diff:.1f}%)")

        with c2:
            st.subheader("Why this price? (local explanation)")

            # Build background for SHAP (small & cached-like)
            bg = df_feat.copy()
            bg = bg.dropna(subset=["latitude", "longitude"])
            bg = bg.sample(min(150, len(bg)), random_state=42)
            bg = engineer_features(bg)
            bg = bg.reindex(columns=meta["features"], fill_value=np.nan)

            # Try SHAP waterfall, fallback to sensitivity
            ok, fig, err = try_shap_explain(model, X_one, bg)

            if ok:
                st.pyplot(fig)
                st.caption("SHAP waterfall shows how features push the prediction up/down from a baseline.")
            else:
                expl = local_sensitivity(model, X_one, meta["features"]).head(12)
                if expl.empty:
                    st.info("No numeric features found for local explanation.")
                else:
                    show = expl.copy()
                    show["delta_price_eur"] = show["delta_price_eur"].map(lambda x: f"{x:+.2f}")
                    show["contribution_pct"] = show["contribution_pct"].map(lambda x: f"{x:.1f}%")
                    st.dataframe(show, use_container_width=True)

                    chart = expl.sort_values("delta_price_eur").set_index("feature")["delta_price_eur"]
                    st.bar_chart(chart)

                with st.expander("Why not SHAP here?"):
                    st.write("SHAP failed in this environment for the pipeline model. Fallback explanation is a sensitivity-style approximation.")
                    st.code(err if err else "Unknown SHAP error")

# -------------------------
# Map tab
# -------------------------
with tab_map:
    st.subheader("Barcelona price map (sample of listings)")

    max_price = st.slider("Clip prices on map at (€)", 100, 600, 300, step=25)
    sample_n = st.slider("Number of points", 500, 5000, 2500, step=250)

    map_df = df_feat.dropna(subset=["latitude", "longitude"]).copy()
    map_df = map_df.dropna(subset=["price_num"]) if "price_num" in map_df.columns else map_df
    map_df = map_df.sample(min(sample_n, len(map_df)), random_state=42)

    if "price_num" in map_df.columns:
        map_df["price_clip"] = map_df["price_num"].clip(upper=max_price)
    else:
        map_df["price_clip"] = 0

    m = folium.Map(
        location=[map_df["latitude"].mean(), map_df["longitude"].mean()],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    for lat_i, lon_i, p_i in zip(map_df["latitude"], map_df["longitude"], map_df["price_clip"]):
        folium.CircleMarker(
            location=[lat_i, lon_i],
            radius=3,
            fill=True,
            fill_opacity=0.45,
            popup=f"€{p_i:.0f}/night (clipped)",
        ).add_to(m)

    if selected_lat is not None and selected_lon is not None and pred_price is not None:
        folium.Marker(
            [float(selected_lat), float(selected_lon)],
            popup=f"Your listing prediction: €{pred_price:.0f}/night",
            icon=folium.Icon(color="red")
        ).add_to(m)

    # IMPORTANT: stable key fixes "blank map in tabs" issues
    st_folium(m, width=1100, height=600, key="map_main")

# -------------------------
# Analysis tab
# -------------------------
with tab_analysis:
    st.subheader("Global model insights")

    # 1) Global drivers as chart (not just table)
    if imp_df is not None and "importance_pct" in imp_df.columns:
        top = imp_df.sort_values("importance_pct", ascending=False).head(15)
        st.markdown("### Top global drivers (Permutation Importance %)")
        st.bar_chart(top["importance_pct"])

        with st.expander("Show raw importance table"):
            st.dataframe(top, use_container_width=True)
    else:
        st.info("Export `permutation_importance.csv` from the notebook to show global drivers here.")

    # 2) Neighborhood median prices as chart
    st.markdown("### Neighborhood median prices (from dataset)")
    if "price_num" in df_feat.columns and "neighbourhood_cleansed" in df_feat.columns:
        nb = (
            df_feat.dropna(subset=["price_num"])
                  .groupby("neighbourhood_cleansed")["price_num"]
                  .median()
                  .sort_values(ascending=False)
        )
        nb_top = nb.head(15)
        st.bar_chart(nb_top)

        with st.expander("Show raw neighbourhood table"):
            st.dataframe(nb_top.to_frame("median_price_eur"), use_container_width=True)
    else:
        st.info("Neighborhood analysis requires `price` to be parsed correctly.")

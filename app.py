import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import folium
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
import re
import altair as alt
import shap

st.set_page_config(page_title="Airbnb Barcelona Price Predictor", layout="wide")

st.markdown(
    """
    <style>
      /* Make tabs bigger */
      button[data-baseweb="tab"] {
        font-size: 20px;
        padding: 12px 18px;
      }
      /* Make tab list a bit taller */
      div[data-baseweb="tab-list"] {
        gap: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

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
# SHAP explanation
# -------------------------
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model[-1])

def shap_explanation(model, X_one: pd.DataFrame, meta: dict) -> pd.DataFrame:
    preprocessor = model[:-1]
    X_proc = preprocessor.transform(X_one)

    try:
        raw_names = preprocessor.get_feature_names_out()
    except Exception:
        raw_names = [f"f{i}" for i in range(X_proc.shape[1])]

    explainer = get_shap_explainer(model)
    shap_vals = explainer.shap_values(X_proc)[0]

    # Sum OHE dummy columns back to their original feature
    agg = {}
    for raw_name, sv in zip(raw_names, shap_vals):
        stripped = raw_name.split("__", 1)[-1] if "__" in raw_name else raw_name
        orig = next(
            (f for f in meta["features"] if stripped == f or stripped.startswith(f + "_")),
            stripped,
        )
        agg[orig] = agg.get(orig, 0.0) + float(sv)

    base_price = float(np.expm1(float(explainer.expected_value)))
    rows = [
        {
            "feature": feat,
            "delta_price_eur": base_price * (np.exp(sv) - 1),
            "direction": "Increase ↑" if sv >= 0 else "Decrease ↓",
            "contribution_pct": abs(sv),
        }
        for feat, sv in agg.items()
    ]

    df = pd.DataFrame(rows)
    total = df["contribution_pct"].sum() or 1.0
    df["contribution_pct"] = 100 * df["contribution_pct"] / total
    df = df.sort_values("contribution_pct", ascending=False)
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

# -------------------------
# Header
# -------------------------
st.title("Airbnb Barcelona — Nightly Price Predictor")
st.caption("Predict nightly price, explore city price patterns, and explain each prediction.")

# -------------------------
# Load everything
# -------------------------
df = load_data("listings.csv")
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
        instant_bookable = st.selectbox("Instant bookable", ["t", "f"])
        host_is_superhost = st.selectbox("Host is superhost", ["t", "f"])
        host_identity_verified = st.selectbox("Host identity verified", ["t", "f"])
        host_has_profile_pic = st.selectbox("Host has profile pic", ["t", "f"])
        has_availability = st.selectbox("Has availability", ["t", "f"])
        host_response_time = st.selectbox("Host response time", resp_time_options) if resp_time_options else None

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

        # auto-filled
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

        # compute explanation ONCE
        expl = shap_explanation(model, X_one, meta).head(12)

        # -------- ROW 1: Prediction | Donut --------
        r1c1, r1c2 = st.columns([1, 1])

        with r1c1:
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

        with r1c2:
            st.subheader("Contribution share (%)")
            if expl.empty:
                st.info("No numeric features available.")
            else:
                donut_df = expl.copy()
                donut_df["contribution_pct"] = donut_df["contribution_pct"].round(1)

                donut = alt.Chart(donut_df).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta("contribution_pct:Q"),
                    color=alt.Color("feature:N", legend=alt.Legend(title="Feature")),
                    tooltip=[
                        alt.Tooltip("feature:N"),
                        alt.Tooltip("contribution_pct:Q", format=".1f", title="Contribution (%)"),
                        alt.Tooltip("delta_price_eur:Q", format="+.2f", title="Δ price (€)")
                    ]
                ).properties(height=320)

                st.altair_chart(donut, use_container_width=True)

        st.divider()

        # -------- ROW 2: Bar | Table --------
        r2c1, r2c2 = st.columns([1.25, 1])

        with r2c1:
            st.subheader("Impact on predicted price (€)")
            if expl.empty:
                st.info("No numeric features available.")
            else:
                bar_df = expl.copy()
                bar_df["impact_sign"] = np.where(bar_df["delta_price_eur"] >= 0, "Push ↑", "Push ↓")
                bar_df = bar_df.sort_values("delta_price_eur")

                bar = alt.Chart(bar_df).mark_bar().encode(
                    x=alt.X("delta_price_eur:Q", title="Δ Predicted price (€) when feature is increased"),
                    y=alt.Y("feature:N", sort=None, title=""),
                    color=alt.Color("impact_sign:N", legend=alt.Legend(title="Effect")),
                    tooltip=[
                        alt.Tooltip("feature:N"),
                        alt.Tooltip("delta_price_eur:Q", format="+.2f", title="Δ price (€)"),
                        alt.Tooltip("contribution_pct:Q", format=".1f", title="Contribution (%)")
                    ]
                ).properties(height=360)

                zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule().encode(x="x:Q")
                st.altair_chart(bar + zero_line, use_container_width=True)

        with r2c2:
            st.subheader("Top local drivers")
            if expl.empty:
                st.info("No numeric features available.")
            else:
                table_df = expl.copy()
                table_df["delta_price_eur"] = table_df["delta_price_eur"].map(lambda x: f"{x:+.2f} €")
                table_df["contribution_pct"] = table_df["contribution_pct"].map(lambda x: f"{x:.1f}%")

                st.dataframe(
                    table_df[["feature", "direction", "delta_price_eur", "contribution_pct"]],
                    use_container_width=True,
                    hide_index=True
                )

        st.caption(
            "Local explanation via SHAP (TreeExplainer): values show each feature's contribution to the prediction relative to the dataset average."
        )



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

    min_p = float(map_df["price_clip"].min())
    max_p = float(map_df["price_clip"].max())
    colormap = LinearColormap(
        ["green", "yellow", "red"], vmin=min_p, vmax=max_p, caption="Nightly price (€)"
    )
    map_df = map_df.copy()
    map_df["color"] = map_df["price_clip"].apply(colormap)

    m = folium.Map(
        location=[map_df["latitude"].mean(), map_df["longitude"].mean()],
        zoom_start=12,
        tiles="OpenStreetMap"
    )
    colormap.add_to(m)

    geo_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row.longitude, row.latitude]},
                "properties": {"price": int(row.price_clip), "color": row.color},
            }
            for row in map_df.itertuples()
        ],
    }

    folium.GeoJson(
        geo_data,
        marker=folium.CircleMarker(radius=4, fill=True),
        style_function=lambda f: {
            "fillColor": f["properties"]["color"],
            "color": f["properties"]["color"],
            "fillOpacity": 0.6,
            "weight": 0,
        },
        tooltip=folium.GeoJsonTooltip(fields=["price"], aliases=["€/night"]),
    ).add_to(m)

    if selected_lat is not None and selected_lon is not None and pred_price is not None:
        folium.Marker(
            [float(selected_lat), float(selected_lon)],
            popup=f"Your listing prediction: €{pred_price:.0f}/night",
            icon=folium.Icon(color="red")
        ).add_to(m)

    st_folium(m, width=1100, height=600, key="map_main")

# -------------------------
# Analysis tab (VARIED charts + neighborhood chart restored)
# -------------------------
with tab_analysis:
    st.subheader("Market & Model Analysis")

    # ---- 1) Price distribution (Histogram) ----
    st.markdown("### Price distribution (nightly €)")
    price_plot = df_feat.dropna(subset=["price_num"]).copy()
    price_plot = price_plot[price_plot["price_num"] > 0]
    price_plot["price_clip"] = price_plot["price_num"].clip(upper=500)

    hist = alt.Chart(price_plot).mark_bar().encode(
        x=alt.X("price_clip:Q", bin=alt.Bin(maxbins=60), title="Nightly price (€) (clipped at 500)"),
        y=alt.Y("count()", title="Listings")
    ).properties(height=300)
    st.altair_chart(hist, use_container_width=True)

    # ---- 2) Boxplot by room type ----
    if "room_type" in df_feat.columns:
        st.markdown("### Price by room type (boxplot)")
        box_df = df_feat.dropna(subset=["price_num", "room_type"]).copy()
        box_df["price_clip"] = box_df["price_num"].clip(upper=500)

        box = alt.Chart(box_df).mark_boxplot().encode(
            x=alt.X("room_type:N", title="Room type"),
            y=alt.Y("price_clip:Q", title="Nightly price (€) (clipped at 500)")
        ).properties(height=320)
        st.altair_chart(box, use_container_width=True)

    # ---- 3) Scatter: distance to center vs price ----
    if "dist_to_center_km" in df_feat.columns:
        st.markdown("### Price vs distance to city center")
        sc = df_feat.dropna(subset=["price_num", "dist_to_center_km"]).copy()
        sc["price_clip"] = sc["price_num"].clip(upper=500)
        sc = sc.sample(min(4000, len(sc)), random_state=42)

        scatter = alt.Chart(sc).mark_circle(opacity=0.25).encode(
            x=alt.X("dist_to_center_km:Q", title="Distance to center (km)"),
            y=alt.Y("price_clip:Q", title="Nightly price (€) (clipped at 500)"),
            tooltip=["neighbourhood_cleansed", "room_type", "price_num", "dist_to_center_km"]
        ).properties(height=320)
        st.altair_chart(scatter, use_container_width=True)

    # ---- 4) Correlation heatmap (numeric features) ----
    st.markdown("### Correlation heatmap (numeric features)")
    num_cols = [c for c in df_feat.columns if pd.api.types.is_numeric_dtype(df_feat[c])]
    keep = [c for c in [
        "price_num", "accommodates", "bedrooms", "beds", "bathrooms_num",
        "amenities_count", "dist_to_center_km", "dist_to_beach_km",
        "availability_30", "availability_90", "number_of_reviews", "reviews_per_month",
        "review_scores_rating"
    ] if c in num_cols]

    if len(keep) >= 4:
        corr = df_feat[keep].corr().stack().reset_index()
        corr.columns = ["x", "y", "corr"]

        heat = alt.Chart(corr).mark_rect().encode(
            x=alt.X("x:N", title=""),
            y=alt.Y("y:N", title=""),
            color=alt.Color("corr:Q", title="corr", scale=alt.Scale(domain=[-1, 1])),
            tooltip=["x", "y", "corr"]
        ).properties(height=420)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Not enough numeric features available for correlation heatmap.")

    # ---- 5) ✅ Neighborhood median prices (RESTORED) ----
    st.markdown("### Neighborhood median prices (from dataset)")
    if "price_num" in df_feat.columns and "neighbourhood_cleansed" in df_feat.columns:
        nb = (
            df_feat.dropna(subset=["price_num"])
                  .groupby("neighbourhood_cleansed")["price_num"]
                  .median()
                  .sort_values(ascending=False)
        )
        nb_top = nb.head(15).reset_index()
        nb_top.columns = ["neighbourhood_cleansed", "median_price_eur"]

        nb_chart = alt.Chart(nb_top).mark_bar().encode(
            x=alt.X("median_price_eur:Q", title="Median nightly price (€)"),
            y=alt.Y("neighbourhood_cleansed:N", sort="-x", title="Neighbourhood"),
            tooltip=["neighbourhood_cleansed", "median_price_eur"]
        ).properties(height=420)

        st.altair_chart(nb_chart, use_container_width=True)
    else:
        st.info("Neighborhood analysis requires parsed `price` and `neighbourhood_cleansed`.")

    # ---- 6) Global drivers (Permutation Importance %) ----
    st.markdown("### Global drivers (Permutation Importance %)")
    if imp_df is not None and "importance_pct" in imp_df.columns:
        top = imp_df.sort_values("importance_pct", ascending=False).head(15).copy()
        top = top.reset_index().rename(columns={"index": "feature"})

        imp_chart = alt.Chart(top).mark_bar().encode(
            x=alt.X("importance_pct:Q", title="Importance (%)"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", "importance_pct"]
        ).properties(height=420)

        st.altair_chart(imp_chart, use_container_width=True)

        with st.expander("Show raw importance table"):
            st.dataframe(top, use_container_width=True)
    else:
        st.info("Export `permutation_importance.csv` from the notebook to show global drivers here.")

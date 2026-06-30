# Explainable Airbnb Price Prediction System

A machine learning web application that predicts nightly Airbnb prices in Barcelona and explains each prediction using SHAP values.

**[Live Demo →](https://explainable-airbnb-price-prediction-system-khdygpyjggnd3dt5xtn.streamlit.app/)**

---

## What it does

- **Predicts** the nightly price of any Airbnb listing in Barcelona
- **Explains** why the model gave that price using real SHAP values — showing each feature's contribution relative to the average listing
- **Visualizes** price patterns across the city on an interactive map
- **Analyzes** market trends: price distributions, neighbourhood rankings, feature correlations, and global feature importance

---

## Features

| Tab | Description |
|-----|-------------|
| **Predict** | Enter listing details (or pick an existing one) → get a price prediction + local SHAP explanation |
| **Map** | Interactive Barcelona map with listings colored green→red by price |
| **Analysis** | Market charts: price distribution, room type boxplots, neighbourhood medians, correlation heatmap, global feature importance |

---

## Tech stack

- **ML:** scikit-learn (`HistGradientBoostingRegressor`), SHAP (`TreeExplainer`)
- **App:** Streamlit, Altair, Folium
- **Data:** [Inside Airbnb](https://insideairbnb.com/get-the-data/) — Barcelona listings

---

## Model

| Metric | Value |
|--------|-------|
| Algorithm | HistGradientBoostingRegressor |
| Target | `log1p(price)` |
| Test R² | 0.770 |
| Median absolute error | €20.86 |
| Features | 29 (20 numeric + 9 categorical) |

Top predictors (permutation importance): `minimum_nights` (28.8%), `accommodates` (27.5%), `property_type` (10.8%)

---

## Run locally

**1. Clone the repo**
```bash
git clone https://github.com/Mohamed-Arbi-Ghanmi/Explainable-Airbnb-Price-Prediction-System.git
cd Explainable-Airbnb-Price-Prediction-System
```

**2. Set up environment**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**
```bash
python download_data.py
```

This downloads the March 2026 Barcelona scrape from Inside Airbnb (~8MB). To use a different scrape date, update the `URL` variable in `download_data.py` before running.

> **Note on dataset versions:** The model was trained on the September 2024 scrape (19,410 listings, 79 columns). The pre-configured URL points to the March 2026 scrape which may have a different number of listings and slightly different price distributions. The column structure is expected to be the same, so predictions and feature engineering should work correctly — but market statistics in the Analysis tab will reflect current data rather than training data.

**4. Launch the app**
```bash
streamlit run app.py
```

> The model (`airbnb_barcelona_price_model.joblib`) is included in the repo. The full dataset is not — the app falls back to a 3,000-listing sample when `listings.csv` is not present.

---

## Project structure

```
├── app.py                               # Streamlit application
├── training_pipeline.ipynb              # ML pipeline: EDA → training → serialization
├── airbnb_barcelona_price_model.joblib  # Trained sklearn pipeline (preprocessing + model)
├── airbnb_barcelona_model_metadata.json # Feature names and target variable info
├── permutation_importance.csv           # Global feature importance scores
├── listings_sample.csv                  # 3,000-row sample used in cloud deployment
├── download_data.py                     # Script to download the full dataset locally
└── requirements.txt
```

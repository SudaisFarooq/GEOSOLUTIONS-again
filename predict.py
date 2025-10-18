import os
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta

# === Load trained model once ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "flood_rf_model_usn.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# === Functions ===
def fetch_rainfall(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch daily rainfall data from NASA POWER API.
    Returns pd.DataFrame with columns ['date', 'rainfall_mm'].
    """
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}"
        "&parameters=PRECTOTCORR&community=AG&format=JSON"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rainfall = data["properties"]["parameter"]["PRECTOTCORR"]
        df = pd.DataFrame({
            "date": pd.to_datetime(list(rainfall.keys()), format='%Y%m%d'),
            "rainfall_mm": list(rainfall.values())
        })
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch rainfall data: {e}")

def predict_flood(start_date_str: str, end_date_str: str, lat: float, lon: float) -> dict:
    """
    Generate flood probability predictions for a date range and location.
    Returns a dict compatible with the Flask API.
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    # Fetch rainfall with lag days
    rain_df = fetch_rainfall(lat, lon, start_date - timedelta(days=14), end_date)
    df = rain_df.copy()

    # === Create lag features ===
    for lag in [1, 2, 3, 7, 14]:
        df[f"rain_lag_{lag}"] = df["rainfall_mm"].shift(lag)

    # === Cumulative rainfall features ===
    for w in [3, 7, 14]:
        df[f"rain_sum_{w}"] = df["rainfall_mm"].rolling(window=w, min_periods=1).sum()

    df = df.dropna()

    # Filter dates within requested range
    pred_df = df[df["date"].between(start_date, end_date)].copy()
    if pred_df.empty:
        raise ValueError("No data available for the selected date range after feature generation")

    # Prepare features
    X = pred_df.drop(columns=["date"])

    # Predict probabilities
    pred_probs = model.predict_proba(X)[:, 1] * 100

    return {
        "prediction": {
            "date": pred_df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "percentage": pred_probs.round(2).tolist()
        }
    }

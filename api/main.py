"""
main.py
FastAPI service for energy demand forecasting.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.features import build_features, FEATURE_COLUMNS
from src.models.ml_models import load_model

app = FastAPI(title="EnergyForecast API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in real production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_energy_forecast.pkl")
model = load_model(MODEL_PATH)


class HistoryPoint(BaseModel):
    timestamp: str  # ISO format, e.g. "2018-08-01T14:00:00"
    demand_mw: float


class ForecastRequest(BaseModel):
    history: List[HistoryPoint]


class ForecastResponse(BaseModel):
    forecast_timestamp: str
    predicted_demand_mw: float

@app.get("/demo-history")
def demo_history(end_date: str):
    """
    Returns 169 hours of actual historical demand ending at end_date (inclusive),
    for demo purposes. end_date format: 'YYYY-MM-DD HH:00:00'
    """
    from src.data_loader import load_pjm_data
    df = load_pjm_data(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "PJME_hourly.csv"))

    end_ts = pd.to_datetime(end_date)
    if end_ts not in df.index:
        raise HTTPException(status_code=400, detail="end_date not found in dataset range (2002-01-01 to 2018-08-03).")

    window = df.loc[:end_ts].iloc[-169:]
    return {
        "history": [
            {"timestamp": str(idx), "demand_mw": float(row["PJME_MW"])}
            for idx, row in window.iterrows()
        ]
    }


@app.get("/")
def root():
    return {"status": "EnergyForecast API is running"}


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    if len(request.history) < 169:
        raise HTTPException(
            status_code=400,
            detail="At least 169 hours of history required (168 for lag_168, plus 1 to compute it against)."
        )



    # Build a DataFrame from the incoming history
    df = pd.DataFrame([{"Datetime": h.timestamp, "PJME_MW": h.demand_mw} for h in request.history])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()

    # Compute the same features used in training
    df_feat = build_features(df).dropna()

    if len(df_feat) == 0:
        raise HTTPException(status_code=400, detail="Not enough valid history after feature computation.")

    # Use the most recent row's features to predict the NEXT hour
    latest_features = df_feat[FEATURE_COLUMNS].iloc[[-1]]
    prediction = model.predict(latest_features)[0]

    next_timestamp = df_feat.index[-1] + pd.Timedelta(hours=1)

    return ForecastResponse(
        forecast_timestamp=str(next_timestamp),
        predicted_demand_mw=float(prediction)
    )
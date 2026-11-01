"""
EnergyForecast — single-service Streamlit application.

The app downloads the public PJME hourly-demand dataset, recreates the
repository's calendar/lag/rolling features, trains the project's XGBoost
recipe once per Streamlit process, and serves one-hour-ahead forecasts.
"""

from __future__ import annotations

from datetime import datetime, time
from io import BytesIO
from typing import Tuple

import pandas as pd
import requests
import streamlit as st

from src.features import FEATURE_COLUMNS, build_features
from src.models.ml_models import train_xgboost


DATA_URL = (
    "https://raw.githubusercontent.com/archd3sai/"
    "Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv"
)
MIN_HISTORY = 169
REQUEST_TIMEOUT = 60


st.set_page_config(
    page_title="EnergyForecast | Electricity Demand",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    """Download and clean the PJME dataset using the project's original rules."""
    response = requests.get(DATA_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    raw = pd.read_csv(BytesIO(response.content))
    required = {"Datetime", "PJME_MW"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    raw["Datetime"] = pd.to_datetime(raw["Datetime"], errors="coerce")
    raw["PJME_MW"] = pd.to_numeric(raw["PJME_MW"], errors="coerce")
    raw = raw.dropna(subset=["Datetime", "PJME_MW"])

    if raw.empty:
        raise ValueError("The downloaded dataset contains no valid observations.")

    frame = raw.set_index("Datetime")[["PJME_MW"]].sort_index()
    # Match src/data_loader.py: average duplicate timestamps, create an hourly
    # index, then linearly interpolate any missing observations.
    frame = frame.groupby(level=0).mean()
    hourly_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    frame = frame.reindex(hourly_index)
    frame.index.name = "Datetime"
    frame["PJME_MW"] = frame["PJME_MW"].interpolate(method="linear")

    if frame["PJME_MW"].isna().any():
        raise ValueError("Could not fill all missing hourly demand observations.")
    if len(frame) < MIN_HISTORY:
        raise ValueError("The source dataset does not contain enough hourly history.")

    return frame


@st.cache_resource(show_spinner="Training the XGBoost forecasting model…")
def get_model():
    """
    Rebuild the project's saved-model recipe from the same PJME dataset.

    The model artifact is DVC-tracked but its remote is not configured in the
    GitHub repository, so this reproducible training path avoids a missing
    binary artifact at deployment time.
    """
    data = load_demo_data()
    training = build_features(data).dropna()

    x_train = training[FEATURE_COLUMNS]
    y_train = training["PJME_MW"]

    # Keep the repository's model hyperparameters; cap parallel workers for
    # predictable resource use on Streamlit Community Cloud.
    return train_xgboost(x_train, y_train, n_jobs=2)


def prepare_uploaded_history(uploaded: pd.DataFrame) -> pd.DataFrame:
    """Validate a user CSV with timestamp and demand_mw columns."""
    required = {"timestamp", "demand_mw"}
    missing = required.difference(uploaded.columns)
    if missing:
        raise ValueError(
            "CSV must contain exactly these required columns: "
            "timestamp and demand_mw."
        )

    frame = uploaded[["timestamp", "demand_mw"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["demand_mw"] = pd.to_numeric(frame["demand_mw"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "demand_mw"])

    if frame.empty:
        raise ValueError("The uploaded CSV contains no valid observations.")

    frame = frame.groupby("timestamp", as_index=True)["demand_mw"].mean().to_frame()
    frame = frame.sort_index()
    frame.index.name = "Datetime"
    frame = frame.rename(columns={"demand_mw": "PJME_MW"})

    if len(frame) < MIN_HISTORY:
        raise ValueError(
            f"At least {MIN_HISTORY} hourly observations are required. "
            f"Your file contains {len(frame)} valid observations."
        )

    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    missing_hours = expected.difference(frame.index)
    if len(missing_hours):
        examples = ", ".join(str(value) for value in missing_hours[:3])
        raise ValueError(
            "The uploaded time series must have continuous hourly timestamps. "
            f"Missing hour(s) include: {examples}"
        )

    return frame


def forecast_next_hour(history: pd.DataFrame, model) -> Tuple[pd.Timestamp, float]:
    """Generate a one-hour-ahead forecast from the latest available history."""
    if len(history) < MIN_HISTORY:
        raise ValueError(f"At least {MIN_HISTORY} hourly observations are required.")

    features = build_features(history).dropna()
    if features.empty:
        raise ValueError("Not enough valid history to calculate the lag features.")

    latest = features[FEATURE_COLUMNS].iloc[[-1]]
    prediction = float(model.predict(latest)[0])
    forecast_timestamp = pd.Timestamp(history.index[-1]) + pd.Timedelta(hours=1)

    if not pd.notna(prediction):
        raise ValueError("The model returned an invalid forecast.")

    return forecast_timestamp, prediction


def show_forecast(history: pd.DataFrame) -> None:
    """Train/load the model, predict, and display the result and recent history."""
    with st.spinner("Preparing the forecasting model and generating your forecast…"):
        model = get_model()
        forecast_timestamp, prediction = forecast_next_hour(history, model)

    left, right = st.columns([1, 2])
    with left:
        st.metric("Next-hour electricity demand", f"{prediction:,.0f} MW")
        st.caption(f"Forecast timestamp: {forecast_timestamp:%Y-%m-%d %H:%M}")
        st.success("Forecast generated successfully.")
    with right:
        observed = history["PJME_MW"].tail(48).rename("Observed demand")
        forecast_point = pd.Series(
            [prediction],
            index=[forecast_timestamp],
            name="Next-hour forecast",
        )
        chart = pd.concat([observed, forecast_point], axis=1)
        st.line_chart(chart, height=300)

    with st.expander("View the latest input observations"):
        st.dataframe(
            history.tail(10).rename(columns={"PJME_MW": "Demand (MW)"}),
            use_container_width=True,
        )


st.title("⚡ EnergyForecast")
st.markdown(
    "### Short-term electricity demand forecasting\n"
    "Forecast the next hour of electricity demand with an XGBoost regression "
    "model using calendar, lag, and rolling-window features."
)

with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** XGBoost Regressor")
    st.write("**Forecast horizon:** Next hour")
    st.write("**Features:** Calendar, lagged demand, 24-hour rolling statistics")
    st.divider()
    st.caption(
        "The demonstration dataset contains historical PJM East (PJME) "
        "demand observations from 2002 to 2018. This is not a live grid feed."
    )

mode = st.radio(
    "Choose your input method",
    ("Explore the PJME demo dataset", "Upload your own hourly data"),
    horizontal=True,
)

if mode == "Explore the PJME demo dataset":
    try:
        with st.spinner("Loading the public PJME hourly-demand dataset…"):
            demo = load_demo_data()

        default_end = pd.Timestamp(demo.index[-2])
        min_date = pd.Timestamp(demo.index[MIN_HISTORY - 1]).date()
        max_date = pd.Timestamp(demo.index[-1]).date()

        st.info(
            f"Dataset available from **{demo.index[0]:%Y-%m-%d %H:%M}** "
            f"to **{demo.index[-1]:%Y-%m-%d %H:%M}**. "
            "Choose the final observation in your input history; the model "
            "will forecast the following hour."
        )

        col_date, col_hour = st.columns(2)
        with col_date:
            selected_date = st.date_input(
                "History end date",
                value=default_end.date(),
                min_value=min_date,
                max_value=max_date,
            )
        with col_hour:
            selected_hour = st.slider("History end hour", 0, 23, default_end.hour)

        selected_end = pd.Timestamp(
            datetime.combine(selected_date, time(hour=selected_hour))
        )

        if selected_end not in demo.index:
            st.warning(
                "That exact date and hour is not present in the hourly dataset. "
                "Choose another hour or date."
            )
        else:
            history = demo.loc[:selected_end].tail(MIN_HISTORY)
            st.caption(
                f"Using {len(history)} hourly observations ending at "
                f"{history.index[-1]:%Y-%m-%d %H:%M}."
            )
            if st.button("Generate next-hour forecast", type="primary"):
                show_forecast(history)

    except requests.RequestException as exc:
        st.error(
            "The public PJME dataset could not be downloaded. "
            "Please retry later or use the upload option with a working "
            "dataset connection."
        )
        st.caption(f"Connection detail: {exc}")
    except Exception as exc:
        st.error(f"Could not prepare the demonstration dataset: {exc}")

else:
    st.markdown(
        "Upload a CSV containing **timestamp** and **demand_mw** columns. "
        "Provide at least 169 consecutive hourly observations."
    )
    st.download_button(
        "Download CSV template",
        data=(
            "timestamp,demand_mw\n"
            "2024-01-01 00:00:00,1200\n"
            "2024-01-01 01:00:00,1180\n"
        ),
        file_name="energy_history_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Upload hourly demand CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded = pd.read_csv(uploaded_file)
            history = prepare_uploaded_history(uploaded)
            st.success(
                f"Validated {len(history):,} continuous hourly observations "
                f"from {history.index[0]:%Y-%m-%d %H:%M} to "
                f"{history.index[-1]:%Y-%m-%d %H:%M}."
            )
            if st.button("Generate next-hour forecast", type="primary"):
                show_forecast(history)
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption(
    "For educational and research demonstration only. Forecasts are estimates, "
    "not operational instructions for electricity-grid control."
)

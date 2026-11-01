"""
EnergyForecast — Streamlit application for recursive multi-hour forecasting.

The app downloads the public PJME hourly-demand dataset, recreates the
repository's feature engineering, trains the XGBoost recipe once per app
process, and forecasts a user-selected horizon from 1 to 168 hours.
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
MAX_HORIZON = 168
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
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )

    raw["Datetime"] = pd.to_datetime(raw["Datetime"], errors="coerce")
    raw["PJME_MW"] = pd.to_numeric(raw["PJME_MW"], errors="coerce")
    raw = raw.dropna(subset=["Datetime", "PJME_MW"])
    if raw.empty:
        raise ValueError("The downloaded dataset contains no valid observations.")

    frame = raw.set_index("Datetime")[["PJME_MW"]].sort_index()
    # Match src/data_loader.py: average duplicate timestamps, create a complete
    # hourly index, and linearly interpolate missing observations.
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

    The trained binary is DVC-tracked but its remote is not configured in this
    repository, so this reproducible training path avoids a missing artifact.
    """
    data = load_demo_data()
    training = build_features(data).dropna()
    x_train = training[FEATURE_COLUMNS]
    y_train = training["PJME_MW"]

    # Preserve the repository's XGBoost hyperparameters; cap workers for hosting.
    return train_xgboost(x_train, y_train, n_jobs=2)


def prepare_uploaded_history(uploaded: pd.DataFrame) -> pd.DataFrame:
    """Validate a CSV with timestamp and demand_mw columns."""
    required = {"timestamp", "demand_mw"}
    missing = required.difference(uploaded.columns)
    if missing:
        raise ValueError(
            "CSV must contain these columns: timestamp and demand_mw."
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


def forecast_hours(
    history: pd.DataFrame, model, horizon: int
) -> pd.DataFrame:
    """
    Recursively forecast the requested number of future hours.

    For each future timestamp, append a placeholder target row, calculate
    features at that future timestamp, predict it, and append the prediction
    to the history used by the next step. This also aligns calendar features
    with the timestamp being forecast.
    """
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"Choose a forecast horizon between 1 and {MAX_HORIZON} hours.")
    if len(history) < MIN_HISTORY:
        raise ValueError(f"At least {MIN_HISTORY} hourly observations are required.")

    working = history[["PJME_MW"]].copy().sort_index()
    if working.index.has_duplicates:
        raise ValueError("Historical timestamps must be unique.")
    expected = pd.date_range(working.index[0], working.index[-1], freq="h")
    if not expected.equals(working.index):
        raise ValueError("Historical observations must be continuous and hourly.")

    predictions = []
    for step in range(1, horizon + 1):
        next_timestamp = pd.Timestamp(working.index[-1]) + pd.Timedelta(hours=1)

        # Add an unknown target at the future time so features are calculated
        # for that future timestamp using only observations available so far.
        future_placeholder = pd.DataFrame(
            {"PJME_MW": [float("nan")]},
            index=pd.DatetimeIndex([next_timestamp], name=working.index.name),
        )
        feature_input = pd.concat([working, future_placeholder])
        feature_frame = build_features(feature_input)
        latest_features = feature_frame.loc[[next_timestamp], FEATURE_COLUMNS]

        if latest_features.isna().any().any():
            raise ValueError(
                "Not enough valid history to calculate the features for "
                f"forecast step {step}."
            )

        prediction = float(model.predict(latest_features)[0])
        if not pd.notna(prediction):
            raise ValueError(f"The model returned an invalid value at step {step}.")

        predictions.append(
            {
                "timestamp": next_timestamp,
                "forecast_demand_mw": prediction,
                "hours_ahead": step,
            }
        )
        # Feed the forecast back as an input to the next recursive step.
        working.loc[next_timestamp, "PJME_MW"] = prediction

    result = pd.DataFrame(predictions).set_index("timestamp")
    result.index.name = "Datetime"
    return result


def show_forecast(history: pd.DataFrame, horizon: int) -> None:
    """Load/train the model, forecast the selected horizon, and display results."""
    with st.spinner(
        f"Preparing the model and generating a {horizon}-hour forecast…"
    ):
        model = get_model()
        forecasts = forecast_hours(history, model, horizon)

    first_value = float(forecasts["forecast_demand_mw"].iloc[0])
    last_value = float(forecasts["forecast_demand_mw"].iloc[-1])
    final_timestamp = forecasts.index[-1]

    st.subheader("Forecast results")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("First hour ahead", f"{first_value:,.0f} MW")
    metric_2.metric(f"Demand after {horizon} hours", f"{last_value:,.0f} MW")
    metric_3.metric("Forecast horizon", f"{horizon} hours")
    st.caption(
        f"Forecast period: {forecasts.index[0]:%Y-%m-%d %H:%M} "
        f"to {final_timestamp:%Y-%m-%d %H:%M}"
    )

    if horizon > 24:
        st.warning(
            "Longer recursive forecasts are increasingly uncertain: each "
            "prediction becomes input to later predictions. Treat horizons "
            "beyond 24 hours as exploratory, not operational forecasts."
        )

    observed = history["PJME_MW"].tail(48).rename("Observed demand")
    predicted = forecasts["forecast_demand_mw"].rename("Forecast demand")
    chart = pd.concat([observed, predicted], axis=1)
    st.markdown("#### Historical demand and future forecast")
    st.line_chart(chart, height=380)

    st.markdown("#### Hour-by-hour forecast")
    display_forecasts = forecasts.rename(
        columns={
            "forecast_demand_mw": "Predicted demand (MW)",
            "hours_ahead": "Hours ahead",
        }
    )
    st.dataframe(
        display_forecasts.style.format({"Predicted demand (MW)": "{:,.2f}"}),
        use_container_width=True,
    )

    csv_data = display_forecasts.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download forecast as CSV",
        data=csv_data,
        file_name=f"energy_forecast_{horizon}h.csv",
        mime="text/csv",
    )

    with st.expander("View the latest input observations"):
        st.dataframe(
            history.tail(10).rename(columns={"PJME_MW": "Demand (MW)"}),
            use_container_width=True,
        )


st.title("⚡ EnergyForecast")
st.markdown(
    "### Short-term electricity demand forecasting\n"
    "Forecast electricity demand from **1 hour up to 7 days ahead** with "
    "an XGBoost regression model and time-series features."
)

with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** XGBoost Regressor")
    st.write("**Forecast horizon:** 1–168 hours")
    st.write("**Features:** Calendar, lagged demand, 24-hour rolling statistics")
    st.divider()
    st.caption(
        "The demo uses historical PJM East (PJME) demand data from 2002–2018. "
        "It is not a live grid feed."
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
            "Choose the final observation in your input history; forecasts "
            "will start in the following hour."
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
        horizon = st.slider(
            "How many hours ahead should we forecast?",
            min_value=1,
            max_value=MAX_HORIZON,
            value=24,
            step=1,
            help="24 hours = one day; 168 hours = seven days.",
        )

        if horizon > 24:
            st.info(
                "You selected more than 24 hours. This version uses recursive "
                "forecasting, so uncertainty can grow with each future step."
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
                f"{history.index[-1]:%Y-%m-%d %H:%M}. "
                f"The forecast will cover the next {horizon} hours."
            )
            if st.button("Generate electricity demand forecast", type="primary"):
                show_forecast(history, horizon)

    except requests.RequestException as exc:
        st.error(
            "The public PJME dataset could not be downloaded. "
            "Please retry later or use the upload option."
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
        "Download CSV format example",
        data=(
            "timestamp,demand_mw\n"
            "2024-01-01 00:00:00,1200\n"
            "2024-01-01 01:00:00,1180\n"
            "... add at least 169 consecutive hourly rows ...\n"
        ),
        file_name="energy_history_format_example.csv",
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
            horizon = st.slider(
                "How many hours ahead should we forecast?",
                min_value=1,
                max_value=MAX_HORIZON,
                value=24,
                step=1,
                help="24 hours = one day; 168 hours = seven days.",
                key="upload_horizon",
            )
            if horizon > 24:
                st.info(
                    "Longer recursive forecasts are increasingly uncertain. "
                    "Treat results beyond 24 hours as exploratory."
                )
            if st.button("Generate electricity demand forecast", type="primary"):
                show_forecast(history, horizon)
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption(
    "For educational and research demonstration only. Forecasts are estimates, "
    "not operational instructions for electricity-grid control."
)

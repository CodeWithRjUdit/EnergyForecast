"""
app.py
Streamlit frontend for EnergyForecast.
"""

import streamlit as st
import requests
import pandas as pd

import os
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="EnergyForecast", layout="centered")
st.title("EnergyForecast")
st.write("Forecast next-hour electricity demand using an XGBoost model.")

mode = st.radio("Choose input mode:", ["Demo (sample dataset)", "Upload CSV"])

history = None

if mode == "Demo (sample dataset)":
    demo_date = st.text_input(
        "End date/time (format: YYYY-MM-DD HH:00:00, range 2002-01-08 to 2018-08-03)",
        value="2018-08-02 23:00:00"
    )
    if st.button("Get Forecast", key="demo_btn"):
        try:
            resp = requests.get(f"{API_BASE}/demo-history", params={"end_date": demo_date})
            resp.raise_for_status()
            history = resp.json()["history"]
        except requests.exceptions.HTTPError:
            st.error(resp.json().get("detail", "Error fetching demo data."))
        except Exception as e:
            st.error(f"Could not reach API: {e}")

else:
    uploaded_file = st.file_uploader("Upload CSV (columns: timestamp, demand_mw)", type="csv")
    if uploaded_file is not None and st.button("Get Forecast", key="upload_btn"):
        df_upload = pd.read_csv(uploaded_file)
        history = [
            {"timestamp": str(row["timestamp"]), "demand_mw": float(row["demand_mw"])}
            for _, row in df_upload.iterrows()
        ]

if history is not None:
    try:
        predict_resp = requests.post(f"{API_BASE}/predict", json={"history": history})
        predict_resp.raise_for_status()
        result = predict_resp.json()

        st.success(f"Forecast for **{result['forecast_timestamp']}**: "
                   f"**{result['predicted_demand_mw']:.0f} MW**")

        # Build chart: last 48 actual hours + forecast point
        recent = pd.DataFrame(history[-48:])
        recent["timestamp"] = pd.to_datetime(recent["timestamp"])
        recent = recent.set_index("timestamp")["demand_mw"]

        forecast_point = pd.Series(
            [result["predicted_demand_mw"]],
            index=[pd.to_datetime(result["forecast_timestamp"])]
        )

        chart_df = pd.DataFrame({
            "Actual": recent,
            "Forecast": forecast_point
        })

        st.line_chart(chart_df)

    except requests.exceptions.HTTPError:
        st.error(predict_resp.json().get("detail", "Prediction failed."))
    except Exception as e:
        st.error(f"Could not reach API: {e}")
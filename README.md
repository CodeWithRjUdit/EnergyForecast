# ⚡ EnergyForecast

**Short-term electricity demand forecasting with XGBoost, time-series feature engineering, and Streamlit.**

EnergyForecast predicts electricity demand one hour ahead using historical hourly demand. The application includes an interactive PJM East (PJME) demonstration dataset and a CSV upload option for custom hourly demand data.

## Live application

Add the public Streamlit Community Cloud URL here after deploying the app.

## Features

- One-hour-ahead electricity-demand forecasting.
- XGBoost regression using the project's original training recipe.
- Calendar features: hour, day of week, month, and weekend indicator.
- Lag features: 1, 2, 24, and 168 hours.
- Rolling 24-hour mean and standard deviation, shifted to avoid target leakage.
- Interactive historical demo using the PJME hourly-demand dataset.
- CSV upload for custom hourly demand histories.
- Forecast metric and recent-history chart.
- Single-service Streamlit deployment; no separate FastAPI server is required.

## Application architecture

```text
EnergyForecast/
├── app.py                     # Streamlit application and inference workflow
├── requirements.txt           # Minimal deployment dependencies
├── runtime.txt                # Python runtime for Streamlit Cloud
├── src/
│   ├── features.py             # Shared feature engineering
│   └── models/
│       └── ml_models.py        # XGBoost training and model utilities
├── api/                        # Original FastAPI service (kept for local/API use)
├── frontend/                   # Original Streamlit frontend
├── models/                     # DVC pointer for the trained model
└── data/                       # DVC pointer for the raw PJME dataset
```

The deployed Streamlit app calls the existing feature-engineering and XGBoost training code directly. This avoids the need to deploy and expose a second web service.

## Model and data availability

The original repository tracks its trained model and raw dataset through DVC pointers. The actual model binary and CSV are not present in the GitHub working tree, and the committed `.dvc/config` does not specify a remote.

To make the public app reproducible without unavailable DVC credentials, the Streamlit app downloads the public PJME hourly dataset from a public GitHub mirror and trains the repository's XGBoost recipe once per running app process. Streamlit caches the dataset and trained model to avoid repeating this work on each interaction.

The demonstration dataset is historical PJM East demand data (2002–2018), not a live electricity-grid feed. The dataset is associated with the [Hourly Energy Consumption dataset on Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

## Input format

For custom data, upload a CSV with these columns:

| Column | Description |
|---|---|
| `timestamp` | Hourly timestamp, e.g. `2024-01-01 00:00:00` |
| `demand_mw` | Observed electricity demand in MW |

Requirements:

- At least 169 observations.
- Timestamps must be continuous and exactly one hour apart.
- Demand values must be numeric.
- Duplicate timestamps are consolidated by taking their mean.

A CSV template is available inside the app.

## Run locally

Python 3.11 is recommended.

```bash
git clone https://github.com/CodeWithRjUdit/EnergyForecast.git
cd EnergyForecast

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first forecast requires downloading the PJME dataset and training the XGBoost model. Subsequent interactions reuse Streamlit's cache while the app process remains active.

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with GitHub and authorize access to this repository.
3. Select **New app**.
4. Choose repository `CodeWithRjUdit/EnergyForecast`, branch `main`, and main file path `app.py`.
5. Deploy the app. Streamlit will assign the public `streamlit.app` URL.

No API keys or Streamlit secrets are required.

## Project limitations

- The PJME demo source is historical and ends in 2018; it must not be interpreted as current or real-time electricity demand.
- The model is retrained when the Streamlit process starts without a warm cache. Initial model training can take longer than later forecasts.
- Custom CSV inputs should represent a consistent hourly demand series comparable to the model's training data.
- Predictions are for educational and research demonstration, not grid operation or safety-critical decisions.

## Technologies

Python · Pandas · NumPy · XGBoost · Scikit-learn · Streamlit · Requests

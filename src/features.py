"""
features.py
Feature engineering for the energy demand forecasting model.
"""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a clean, datetime-indexed DataFrame with a PJME_MW column,
    add calendar, lag, and rolling-window features.

    Returns a DataFrame with all original + new feature columns.
    Rows without enough history for the longest lag (168h) will
    contain NaNs and should be dropped by the caller if needed.
    """
    df = df.copy()

    # Calendar features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Lag features
    df["lag_1"] = df["PJME_MW"].shift(1)
    df["lag_2"] = df["PJME_MW"].shift(2)
    df["lag_24"] = df["PJME_MW"].shift(24)
    df["lag_168"] = df["PJME_MW"].shift(168)

    # Rolling window features (shifted first to avoid leakage)
    df["rolling_mean_24"] = df["PJME_MW"].shift(1).rolling(window=24).mean()
    df["rolling_std_24"] = df["PJME_MW"].shift(1).rolling(window=24).std()

    return df


FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend",
    "lag_1", "lag_2", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24",
]
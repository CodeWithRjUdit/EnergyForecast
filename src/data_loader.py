"""
data_loader.py
Loads and cleans the PJM hourly energy consumption data.
"""

import pandas as pd


def load_pjm_data(filepath: str) -> pd.DataFrame:
    """
    Load PJM hourly energy consumption CSV, clean it, and return a
    complete, gap-free, duplicate-free, datetime-indexed DataFrame.
    """
    df = pd.read_csv(filepath)

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    df = df.sort_index()

    # Fix duplicate timestamps (DST fall-back) by averaging
    df = df.groupby(df.index).mean()

    # Fix gaps by reindexing onto a complete hourly range, then interpolating
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_range)
    df["PJME_MW"] = df["PJME_MW"].interpolate(method="linear")

    return df
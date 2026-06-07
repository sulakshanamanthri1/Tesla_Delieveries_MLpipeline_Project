import pandas as pd


def create_features(df):

    df["Date"] = pd.to_datetime(
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str)
        + "-01"
    )

    df["Quarter"] = df["Date"].dt.quarter

    df["Price_Per_KM"] = (
        df["Avg_Price_USD"]
        / df["Range_km"]
    )

    return df
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_data


df = load_data()

monthly = (

    df.groupby(
        ["Year","Month"]
    )["Estimated_Deliveries"]

    .sum()

    .reset_index()

)

monthly["Date"] = pd.to_datetime(

    monthly["Year"].astype(str)
    + "-"
    + monthly["Month"].astype(str)
    + "-01"

)

monthly = monthly.sort_values(
    "Date"
)

monthly["Forecast"] = (

    monthly["Estimated_Deliveries"]

    .rolling(window=3)

    .mean()

)

plt.figure(figsize=(12,6))

plt.plot(

    monthly["Date"],
    monthly["Estimated_Deliveries"],
    label="Actual Deliveries"

)

plt.plot(

    monthly["Date"],
    monthly["Forecast"],
    label="Forecast"

)

plt.title(
    "Tesla Deliveries Forecast"
)

plt.xlabel("Date")

plt.ylabel("Deliveries")

plt.legend()

plt.show()
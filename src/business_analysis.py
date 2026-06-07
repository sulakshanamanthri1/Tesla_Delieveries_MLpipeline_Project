import matplotlib.pyplot as plt

from preprocessing import load_data

df = load_data()

region_sales = (
    df.groupby("Region")
      ["Estimated_Deliveries"]
      .sum()
      .sort_values(
          ascending=False
      )
)

plt.figure(figsize=(10,5))

region_sales.plot(
    kind="bar"
)

plt.title(
    "Region Wise Deliveries"
)

plt.ylabel(
    "Deliveries"
)

plt.show()

model_sales = (
    df.groupby("Model")
      ["Estimated_Deliveries"]
      .sum()
      .sort_values(
          ascending=False
      )
)

plt.figure(figsize=(10,5))

model_sales.plot(
    kind="bar"
)

plt.title(
    "Model Wise Deliveries"
)

plt.ylabel(
    "Deliveries"
)

plt.show()
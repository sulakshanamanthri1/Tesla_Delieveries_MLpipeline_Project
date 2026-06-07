import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data

df = load_data()

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

plt.figure(figsize=(10,5))

sns.histplot(
    df["Estimated_Deliveries"],
    kde=True
)

plt.title(
    "Distribution of Estimated Deliveries"
)

plt.show()

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["Estimated_Deliveries"]
)

plt.title(
    "Outlier Detection"
)

plt.show()

numeric_cols = df.select_dtypes(
    include=["number"]
)

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_cols.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.show()
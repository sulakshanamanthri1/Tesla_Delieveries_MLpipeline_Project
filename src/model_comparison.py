import pandas as pd
import matplotlib.pyplot as plt

results = {

    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest"
    ],

    "R2": [
        0.91,
        0.91,
        0.91,
        0.96
    ]
}

df = pd.DataFrame(results)

plt.figure(figsize=(8,5))

plt.bar(
    df["Model"],
    df["R2"]
)

plt.title(
    "Model Comparison"
)

plt.ylabel(
    "R² Score"
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()
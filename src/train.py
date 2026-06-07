import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import load_data
from feature_engineering import create_features
from config import MODEL_PATH

df = load_data()
df = create_features(df)

X = df.drop(
    columns=[
        "Estimated_Deliveries",
        "Date"
    ]
)

y = df["Estimated_Deliveries"]

cat_cols = [
    "Region",
    "Model",
    "Source_Type"
]

num_cols = [
    col
    for col in X.columns
    if col not in cat_cols
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            cat_cols
        )
    ]
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(alpha=1.0),

    "Lasso Regression":
        Lasso(alpha=0.1),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
}

results = []

best_model = None
best_model_name = None
best_r2 = -999

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = pipeline.predict(
        X_test
    )

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = (
        mean_squared_error(
            y_test,
            predictions
        )
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        [
            name,
            round(mae, 2),
            round(rmse, 2),
            round(r2, 4)
        ]
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    if r2 > best_r2:

        best_r2 = r2
        best_model = pipeline
        best_model_name = name

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)
joblib.dump(
    best_model,
    MODEL_PATH
)

print("\n")
print(f"Best Model : {best_model_name}")
print(f"Best R²    : {best_r2:.4f}")
print("Model Saved Successfully")
print("=" * 60)
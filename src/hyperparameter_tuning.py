import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
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
from config import TUNED_MODEL_PATH


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

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42
            )
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

param_grid = {

    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__max_depth": [
        5,
        10,
        15,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]
}

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    cv=5,

    scoring="r2",

    n_jobs=-1,

    verbose=2
)

print("\nStarting Hyperparameter Tuning...\n")

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

predictions = best_model.predict(
    X_test
)

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

print("\n" + "="*60)

print("BEST PARAMETERS")

print("="*60)

print(
    grid_search.best_params_
)

print("\nBest CV Score:")
print(
    round(
        grid_search.best_score_,
        4
    )
)

print("\nTest Metrics")

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"R²   : {r2:.4f}"
)

joblib.dump(
    best_model,
    TUNED_MODEL_PATH
)

print("\nBest Tuned Model Saved!")
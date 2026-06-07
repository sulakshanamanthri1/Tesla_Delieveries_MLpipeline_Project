from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "tesla_deliveries_dataset_2015_2025.csv"

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
TUNED_MODEL_PATH = BASE_DIR / "models" / "best_tuned_model.pkl"
DASHBOARD_TITLE = "Tesla Deliveries Intelligence Platform"
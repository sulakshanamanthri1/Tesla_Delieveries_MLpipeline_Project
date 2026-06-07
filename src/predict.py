import joblib
import pandas as pd

from config import MODEL_PATH

# ==========================
# Load Saved Model
# ==========================

model = joblib.load(MODEL_PATH)

# ==========================
# New Input Data
# ==========================

input_data = pd.DataFrame({

    "Year": [2026],

    "Month": [1],

    "Region": ["North America"],

    "Model": ["Model Y"],

    "Production_Units": [8000],

    "Avg_Price_USD": [55000],

    "Battery_Capacity_kWh": [75],

    "Range_km": [500],

    "CO2_Saved_tons": [3000],

    "Source_Type": ["Official"],

    "Charging_Stations": [18000],

    "Quarter": [1],

    "Price_Per_KM": [110]

})

# ==========================
# Prediction
# ==========================

prediction = model.predict(input_data)

print("\nPredicted Deliveries:")
print(round(prediction[0]))
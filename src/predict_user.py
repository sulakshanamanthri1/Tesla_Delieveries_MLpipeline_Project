import joblib
import pandas as pd

from config import MODEL_PATH

model = joblib.load(MODEL_PATH)

year = int(input("Enter Year: "))
month = int(input("Enter Month: "))
production = int(input("Production Units: "))
price = float(input("Average Price USD: "))
battery = float(input("Battery Capacity kWh: "))
range_km = float(input("Range KM: "))
co2 = float(input("CO2 Saved Tons: "))
charging = int(input("Charging Stations: "))

region = input("Region: ")
model_name = input("Model: ")
source = input("Source Type: ")

quarter = ((month - 1) // 3) + 1

price_per_km = price / range_km

input_df = pd.DataFrame({

    "Year": [year],
    "Month": [month],
    "Region": [region],
    "Model": [model_name],
    "Production_Units": [production],
    "Avg_Price_USD": [price],
    "Battery_Capacity_kWh": [battery],
    "Range_km": [range_km],
    "CO2_Saved_tons": [co2],
    "Source_Type": [source],
    "Charging_Stations": [charging],
    "Quarter": [quarter],
    "Price_Per_KM": [price_per_km]

})

prediction = model.predict(input_df)

print("\nPredicted Deliveries:")
print(round(prediction[0]))
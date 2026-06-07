from preprocessing import load_data
from feature_engineering import create_features

df = load_data()

df = create_features(df)

print(df.head())

print(df.columns)
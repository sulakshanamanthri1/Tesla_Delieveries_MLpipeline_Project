## Tesla Deliveries Intelligence Platform

An end-to-end Machine Learning project that analyzes Tesla vehicle deliveries, predicts future deliveries using regression models, performs hyperparameter tuning, and visualizes insights through an interactive Streamlit dashboard.

## Project Overview

This project demonstrates the complete Machine Learning lifecycle:

Data Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Regression Modeling
Hyperparameter Tuning
Forecasting
Interactive Dashboard Development

The objective is to predict Estimated Deliveries using Tesla production, pricing, battery, charging infrastructure, and regional information.

## SDataset Information
Features
Feature	Description
Year	Delivery Year
Month	Delivery Month
Region	Sales Region
Model	Tesla Vehicle Model
Production_Units	Units Produced
Avg_Price_USD	Average Vehicle Price
Battery_Capacity_kWh	Battery Capacity
Range_km	Vehicle Range
CO2_Saved_tons	Environmental Impact
Source_Type	Data Source
Charging_Stations	Charging Infrastructure
Target Variable
Estimated_Deliveries
## Project Structure

Tesla_Delivery_Pipeline/
│
├── data/
│   └── tesla_deliveries.csv
│
├── models/
│   ├── best_model.pkl
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── business_analysis.py
│   ├── train.py
│   ├── predict.py
│   ├── forecasting.py
│   └── hyperparameter_tuning.py
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
│
└── README.md

## Machine Learning Pipeline
1. Data Preprocessing
Data loading
Duplicate removal
Data validation
2. Feature Engineering
Created additional features:
Quarter
Price_Per_KM
3. Exploratory Data Analysis
Performed:
Distribution Analysis
Correlation Analysis
Outlier Detection
Business Insights
4. Model Development
Implemented:
Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regression
5. Hyperparameter Tuning
Used:
GridSearchCV
Optimized:
Number of Trees
Maximum Depth
Minimum Samples Split
Minimum Samples Leaf
6. Forecasting
Generated delivery forecasts using moving average techniques and time-series trend analysis.

## Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Plotly
Scikit-Learn
Joblib
Streamlit
Installation

Clone the repository:

git clone <https://github.com/sulakshanamanthri1/Tesla_Delieveries_MLpipeline_Project.git>

Navigate to the project folder:

cd Tesla_Delivery_Pipeline

Install dependencies:

pip install -r requirements.txt
Run Model Training
python src/train.py
Run Hyperparameter Tuning
python src/hyperparameter_tuning.py
Run Prediction Module
python src/predict.py
Run Forecasting Module
python src/forecasting.py
Launch Dashboard
streamlit run dashboard/app.py
Dashboard Modules
Dataset Overview
Dataset Shape
Dataset Preview
Summary Statistics
Business Analytics
Region-wise Deliveries
Model-wise Deliveries
Business Insights
Delivery Prediction

Predict future Tesla deliveries using trained machine learning models.

 ## Forecasting

Visualize historical and forecasted delivery trends.

## Model Evaluation Metrics

The models were evaluated using:

MAE

Mean Absolute Error

RMSE

Root Mean Squared Error

R² Score

Coefficient of Determination

## Key Business Insights
Production units significantly influence deliveries.
Charging infrastructure impacts vehicle adoption.
Regional demand varies considerably across markets.
Vehicle pricing and range affect delivery volumes.
Future Enhancements
XGBoost Integration
SHAP Explainability
MLflow Experiment Tracking
Advanced Time-Series Forecasting
Docker Deployment
Cloud Deployment
Learning Outcomes

Through this project, the following concepts were implemented:

End-to-End Machine Learning Pipeline
Feature Engineering
Model Comparison
Hyperparameter Optimization
Time Series Analysis
Business Analytics
Dashboard Development
Model Deployment Concepts

https://tesladelieveriesmlpipelineproject-5qg37sdxjvss2xdtw3mtrl.streamlit.app/
This is the link to check this project after deployement
by
M.Sulakshana

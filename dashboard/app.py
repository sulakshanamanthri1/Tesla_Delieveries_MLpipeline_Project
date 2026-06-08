import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "tesla_deliveries_dataset_2015_2025.csv"

MODEL_PATH = BASE_DIR / "models" / "best_tuned_model.pkl"

st.set_page_config(
    page_title="Tesla Dashboard",
    layout="wide"
)

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

st.title(
    "Tesla Deliveries Intelligence Platform"
)

st.info("Select any button below to navigate through the dashboard.")

col1, col2, col3, col4 = st.columns(4)

if "page" not in st.session_state:
    st.session_state.page = "Prediction"

with col1:
    if st.button(
        " Prediction",
        use_container_width=True
    ):
        st.session_state.page = "Prediction"

with col2:
    if st.button(
        " Dataset Overview",
        use_container_width=True
    ):
        st.session_state.page = "Dataset Overview"

with col3:
    if st.button(
        " Business Analytics",
        use_container_width=True
    ):
        st.session_state.page = "Business Analytics"

with col4:
    if st.button(
        " Forecasting",
        use_container_width=True
    ):
        st.session_state.page = "Forecasting"

page = st.session_state.page

if page == "Dataset Overview":

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Regions",
        df["Region"].nunique()
    )

    st.dataframe(df.head())

elif page == "Business Analytics":

    st.header("Business Analytics")

    region_data = (

        df.groupby("Region")

        ["Estimated_Deliveries"]

        .sum()

        .reset_index()

    )

    fig1 = px.bar(

        region_data,

        x="Region",

        y="Estimated_Deliveries",

        title="Region Wise Deliveries"

    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    model_data = (

        df.groupby("Model")

        ["Estimated_Deliveries"]

        .sum()

        .reset_index()

    )

    fig2 = px.pie(

        model_data,

        names="Model",

        values="Estimated_Deliveries",

        title="Model Contribution"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

elif page == "Prediction":

    st.header("Predict Deliveries of Tesla Vehicles")

    year = st.number_input(
        "Year",
        value=2026
    )

    month = st.slider(
        "Month",
        1,
        12,
        1
    )

    production = st.number_input(
        "Production Units",
        value=8000
    )

    price = st.number_input(
        "Average Price USD",
        value=55000
    )

    battery = st.number_input(
        "Battery Capacity",
        value=75
    )

    range_km = st.number_input(
        "Range KM",
        value=500
    )

    charging = st.number_input(
        "Charging Stations",
        value=18000
    )

    region = st.selectbox(
        "Region",
        sorted(df["Region"].unique())
    )

    model_name = st.selectbox(
        "Model",
        sorted(df["Model"].unique())
    )

    source = st.selectbox(
        "Source Type",
        sorted(df["Source_Type"].unique())
    )

    if st.button(
        "Predict Deliveries"
    ):

        quarter = (
            (month - 1) // 3
        ) + 1

        input_df = pd.DataFrame({

            "Year":[year],

            "Month":[month],

            "Region":[region],

            "Model":[model_name],

            "Production_Units":[production],

            "Avg_Price_USD":[price],

            "Battery_Capacity_kWh":[battery],

            "Range_km":[range_km],

            "CO2_Saved_tons":[3000],

            "Source_Type":[source],

            "Charging_Stations":[charging],

            "Quarter":[quarter],

            "Price_Per_KM":[
                price/range_km
            ]
        })

        prediction = model.predict(
            input_df
        )

        st.success(
            f"Predicted Deliveries: {int(prediction[0]):,}"
        )

elif page == "Forecasting":

    st.header("Forecasting")

    monthly = (

        df.groupby(
            ["Year","Month"]
        )

        ["Estimated_Deliveries"]

        .sum()

        .reset_index()

    )

    monthly["Date"] = pd.to_datetime(

        monthly["Year"].astype(str)

        + "-"

        + monthly["Month"].astype(str)

        + "-01"
    )

    monthly["Forecast"] = (

        monthly[
            "Estimated_Deliveries"
        ]

        .rolling(3)

        .mean()
    )

    fig = px.line(

        monthly,

        x="Date",

        y=[
            "Estimated_Deliveries",
            "Forecast"
        ],

        title="Delivery Forecast"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
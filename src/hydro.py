import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------
# Load Model + Encoders
# ---------------------------------------------------
model = pickle.load(open("models/lightgbm_model.pkl", "rb"))
encoders = pickle.load(open("models/label_encoders.pkl", "rb"))

st.set_page_config(page_title="Hydraulic Sales Forecasting", layout="wide")

# ---------------------------------------------------
# Helper encoding function
# ---------------------------------------------------
def encode(col, value):
    return encoders[col].transform([value])[0]

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("🔧 Hydraulic Sales Forecasting App")
st.markdown("Use product, market, and operational conditions to predict **Units Sold**.")
st.write("---")

# ---------------------------------------------------
# PRODUCT INFORMATION
# ---------------------------------------------------
st.header("📦 Product Information")

col1, col2 = st.columns(2)

with col1:
    product_type = st.selectbox("Product Type", encoders['product_type'].classes_)
    product_code = st.selectbox("Product Code", encoders['product_code'].classes_)
    variant = st.selectbox("Variant", encoders['variant'].classes_)
    oil_type = st.selectbox("Oil Type", encoders['oil_type'].classes_)

with col2:
    region = st.selectbox("Region", encoders['region'].classes_)
    country = st.selectbox("Country", encoders['country'].classes_)
    customer_segment = st.selectbox("Customer Segment", encoders['customer_segment'].classes_)
    channel = st.selectbox("Sales Channel", encoders['channel'].classes_)
    application_area = st.selectbox("Application Area", encoders['application_area'].classes_)

st.write("---")

# ---------------------------------------------------
# MARKET & OPERATIONAL FACTORS
# ---------------------------------------------------
st.header("📊 Market & Operational Factors")

col3, col4, col5 = st.columns(3)

with col3:
    revenue = st.number_input(
        "Revenue (₹2.5L – ₹18L)", 
        min_value=250000, max_value=1800000, 
        value=800000, step=5000,
        help="Typical revenue range: 6.5L–14L"
    )

    marketing_spend = st.number_input(
        "Marketing Spend (₹10k – ₹5L)", 
        min_value=10000, max_value=500000, 
        value=30000, step=5000,
        help="Most companies spend 20k–40k"
    )

    discount_percent = st.number_input(
        "Discount Percent (3% – 14%)", 
        min_value=3, max_value=14, 
        value=8, step=1
    )

with col4:
    stock_available = st.number_input(
        "Stock Available (100 – 500 units)",
        min_value=100, max_value=500,
        value=300, step=10
    )

    lead_time_days = st.number_input(
        "Lead Time (Days) (5 – 19)",
        min_value=5, max_value=19,
        value=12, step=1
    )

    competitor_activity = st.number_input(
        "Competitor Activity Index (0 – 9)",
        min_value=0, max_value=9,
        value=4, step=1
    )

with col5:
    seasonality_index = st.number_input(
        "Seasonality Index (0.85 – 1.25)",
        min_value=0.85, max_value=1.25,
        value=1.05, step=0.01
    )

    economic_indicator = st.number_input(
        "Economic Indicator (5.5 – 8.5)",
        min_value=5.5, max_value=8.5,
        value=6.80, step=0.1
    )

    failure_rate_pct = st.number_input(
        "Failure Rate (%) (0.1 – 2.5)",
        min_value=0.1, max_value=2.5,
        value=1.20, step=0.1
    )

return_units = st.number_input("Return Units (0–4)", min_value=0, max_value=4, value=1, step=1)

st.write("---")

# ---------------------------------------------------
# TECHNICAL FIELDS
# ---------------------------------------------------
st.header("⚙ Technical Specifications")

col6, col7 = st.columns(2)

with col6:
    pressure_rating_bar = st.selectbox("Pressure Rating (Bar)", [160, 200, 250, 315, 350])
with col7:
    temperature_rating_c = st.selectbox("Temperature Rating (°C)", [80, 90, 100, 110])

st.write("---")

# ---------------------------------------------------
# DATE FEATURES
# ---------------------------------------------------
st.header("🗓 Date Features")

col8, col9 = st.columns(2)

with col8:
    year = st.number_input("Year", min_value=2000, max_value=2050, value=2020)

with col9:
    month = st.number_input("Month (1-12) ", min_value=1, max_value=12, value=1)

quarter = (month - 1) // 3 + 1
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

st.write("---")

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------
if st.button("🔮 Predict Units Sold"):
    
    row = np.array([
        encode('product_type', product_type),
        encode('product_code', product_code),
        encode('variant', variant),
        encode('region', region),
        encode('country', country),
        encode('customer_segment', customer_segment),
        encode('channel', channel),
        encode('application_area', application_area),
        revenue,
        marketing_spend,
        discount_percent,
        stock_available,
        lead_time_days,
        competitor_activity,
        seasonality_index,
        economic_indicator,
        0,  # new_product_launch not used in your UI
        failure_rate_pct,
        return_units,
        pressure_rating_bar,
        temperature_rating_c,
        encode('oil_type', oil_type),
        year,
        month,
        quarter,
        month_sin,
        month_cos
    ]).reshape(1, -1)

    prediction = model.predict(row)[0]

    st.success(f"### 📦 Predicted Units Sold: **{int(prediction)}**")

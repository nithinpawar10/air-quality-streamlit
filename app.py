import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load("logistic_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please make sure 'logistic_model.pkl' and 'scaler.pkl' exist.")
        return None, None

model, scaler = load_model()

# UI Header
st.title("🌍 Air Quality Classifier")
st.markdown("Predict Pollution Level based on air quality sensor data using a trained Logistic Regression model.")

# Input form only if model is loaded
if model is not None and scaler is not None:
    st.subheader("📥 Enter Sensor Readings:")

    features = [
        'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
        'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]
    user_input = {}

    with st.form("input_form"):
        cols = st.columns(3)
        for idx, feature in enumerate(features):
            with cols[idx % 3]:
                user_input[feature] = st.number_input(f"{feature}", step=0.01, format="%.2f")

        submitted = st.form_submit_button("🚀 Predict Pollution Level")

    # Run prediction
    if submitted:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        label_map = {0: "Good", 1: "Moderate", 2: "Unhealthy", 3: "Hazardous"}
        prediction = label_map.get(pred, "Unknown")

        st.success(f"✅ Predicted Pollution Level: **{prediction}**")

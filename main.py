# python -m streamlit run main.py (for run streamlit app in browser)
# pip install scikit-learn
# pip install streamlit
# pip install joblib

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = joblib.load("model.pkl")
scaler = joblib.load("x_scaler.pkl")

condition_encoder = joblib.load("x_Condition.pkl")
garage_encoder = joblib.load("x_Garage.pkl")
location_encoder = joblib.load("x_Location.pkl")

if not hasattr(condition_encoder, "classes_"):
    condition_encoder.fit(["Excellent", "Good", "Fare", "Poor"])
if not hasattr(garage_encoder, "classes_"):
    garage_encoder.fit(["Yes", "No"])
if not hasattr(location_encoder, "classes_"):
    location_encoder.fit(["Downtown", "Suburban"])

st.title("House Price Prediction")
x_Condition = st.selectbox(
    "condition",
    ["Excellent", "Good", "Fare", "Poor"]
)
x_Garage = st.selectbox(
    "Garage",
    ["Yes", "No"]
)
x_Location = st.selectbox(
    "Location",
    ["Downtown", "Suburban"]
)
Bedroom = st.number_input("Bedroom", min_value=1)
area = st.number_input("area", min_value=501)
Bathroom = st.number_input("Bathroom", min_value=1)
Floors = st.number_input("floors", min_value=1)
yearbuilt = st.number_input("yearbuilt", min_value=1950)

if st.button("Predict"):

    Condition = condition_encoder.transform([x_Condition])[0]
    Garage = garage_encoder.transform([x_Garage])[0]
    Location = location_encoder.transform([x_Location])[0]

    features = [[Condition, Garage, Location, Bedroom, area, Bathroom, Floors, yearbuilt]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.success(f"Predicted house price: {prediction[0]:.2f}")

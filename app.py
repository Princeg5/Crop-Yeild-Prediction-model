import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load("crop_yield_pre.pkl")  # Ensure the model file exists
scaler = joblib.load("scaler_crop.pkl")  # Ensure the scaler file exists

# Streamlit UI
st.title("Crop Yield Prediction App 🌾")
st.markdown("### Enter the feature values below:")

# Create input fields for features (update based on dataset)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)
feature5 = st.number_input("Feature 5", value=0.0)
feature6 = st.number_input("Feature 6", value=0.0)
feature7 = st.number_input("Feature 7", value=0.0)

# Predict button
if st.button("Predict Crop Yield"):
    # Prepare input for model
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6 , feature7]])
    input_scaled = scaler.transform(input_data)  # Apply same scaling used in training

    # Predict
    prediction = model.predict(input_scaled)

    # Display result
    st.success(f"Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")


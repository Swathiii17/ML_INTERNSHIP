import streamlit as st
import pickle
import numpy as np

# Load model
with open("crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾")

st.title("🌾 Crop Yield Prediction System")
st.write("Enter soil and climate details to predict crop yield")

# Input fields
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
fertilizer = st.number_input("Fertilizer (kg/acre)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
nitrogen = st.number_input("Nitrogen (N)")
phosphorus = st.number_input("Phosphorus (P)")
potassium = st.number_input("Potassium (K)")

if st.button("Predict Yield"):
    input_data = np.array([[rainfall, fertilizer, temperature,
                             nitrogen, phosphorus, potassium]])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"🌾 Predicted Crop Yield: {prediction:.2f} Q/acre")

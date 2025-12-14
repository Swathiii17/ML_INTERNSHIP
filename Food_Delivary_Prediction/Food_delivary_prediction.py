import streamlit as st
import numpy as np
import pickle

# ----------------------------------
# Load saved objects
# ----------------------------------
with open("Food_Delivary_Prediction/delivary_detect.pkl","rb") as f:
    model = pickle.load(f)

with open("Food_Delivary_Prediction/scaler_deliv.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("Food_Delivary_Prediction/encoders_deliv.pkl", "rb") as f:
    encoder = pickle.load(f)

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Food Delivery Time Prediction", layout="centered")
st.title("🍔 Food Delivery Time Prediction")

st.write("Predict delivery time based on order details")

# ----------------------------------
# User Inputs
# ----------------------------------
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)

traffic_text = st.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

weather_text = st.selectbox(
    "Weather Condition",
    ["Clear", "Rainy", "Foggy"]
)

prep_time = st.number_input(
    "Preparation Time (minutes)",
    min_value=0,
    step=1
)

# ----------------------------------
# Encode categorical inputs
# ----------------------------------
traffic_encoded = encoder["Traffic"].transform([traffic_text])[0]
weather_encoded = encoder["Weather"].transform([weather_text])[0]

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("Predict Delivery Time"):
    user_input = np.array([[distance, traffic_encoded, weather_encoded, prep_time]])

    # Scale numeric inputs
    user_input_scaled = scaler.transform(user_input)

    prediction = model.predict(user_input_scaled)[0]

    st.success(f"⏱ Estimated Delivery Time: **{prediction:.2f} minutes**")

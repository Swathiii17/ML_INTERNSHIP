import streamlit as st
import pickle
import pandas as pd

# -------------------- Load saved objects --------------------
with open("Food_Delivery_Pridiction/deliv_predict", "rb") as f:
    model = pickle.load(f)

with open("Food_Delivery_Pridiction/deliv_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)   # dict of LabelEncoders

with open("Food_Delivery_Pridiction/deliv_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Delivery Time Predictor", layout="centered")
st.title("🚚 Food Delivery Time Prediction")

st.write("Enter order details to predict delivery time")

# Inputs
distance = st.number_input("Distance (km)", min_value=0.0, step=0.1)
prep_time = st.number_input("Preparation Time (min)", min_value=0.0, step=1.0)

weather = st.selectbox(
    "Weather",
    ["Clear", "Windy", "Foggy", "Rainy", "Snowy"]
)

traffic = st.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

# -------------------- Prediction --------------------
if st.button("Predict Delivery Time"):
    try:
        # Encode categorical values
        weather_encoded = encoders["Weather"].transform([weather])[0]
        traffic_encoded = encoders["Traffic_Level"].transform([traffic])[0]

        # Create input dataframe (MUST match training columns)
        input_df = pd.DataFrame(
            [[distance, prep_time, weather_encoded, traffic_encoded]],
            columns=[
                "Distance_km",
                "Preparation_Time_min",
                "Weather",
                "Traffic_Level"
            ]
        )

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        st.success(f"⏱️ Estimated Delivery Time: **{prediction:.2f} minutes**")

    except Exception as e:
        st.error(f"Error: {e}")

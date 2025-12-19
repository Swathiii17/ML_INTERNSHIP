import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("💼 Salary Prediction App")
st.caption("Predict salary based on years of experience using a trained ML model")

# -----------------------------
# Load Pickle Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("Salary_prediction/salary_predict_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -----------------------------
# Sidebar Information
# -----------------------------
st.sidebar.header("ℹ️ Model Details")
st.sidebar.write("Algorithm: Linear Regression")
st.sidebar.write("Input: Years of Experience")
st.sidebar.write("Output: Predicted Salary")
st.sidebar.write("Model Type: Pickle (.pkl)")

# -----------------------------
# User Input
# -----------------------------
st.subheader("📝 Enter Years of Experience")

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("💰 Predict Salary"):
    input_data = np.array([[experience]])
    salary = model.predict(input_data)[0]

    st.success(f"Estimated Salary: ₹ {salary:,.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit and Machine Learning")

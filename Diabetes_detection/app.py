import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Diabetes Prediction App")
st.caption("Logistic Regression model for diabetes detection")

# -----------------------------
# Load Pickle Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("Diabetes_detection/diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

Logr = load_model()

# -----------------------------
# Sidebar - Model Info
# -----------------------------
st.sidebar.header("ℹ️ Model Information")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Inputs:")
st.sidebar.write("• Age")
st.sidebar.write("• Body Mass")
st.sidebar.write("• Insulin Level")
st.sidebar.write("• Plasma Level")
st.sidebar.write("Output:")
st.sidebar.write("• Diabetes Status")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📝 Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, step=1)
mass = st.number_input("Body Mass", min_value=10, max_value=200, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
plasma = st.number_input("Plasma Level", min_value=0, max_value=300, step=1)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Diabetes"):
    input_data = np.array([[age, mass, insulin, plasma]])
    prediction = Logr.predict(input_data)[0]

    if prediction == "tested_positive":
        st.error("⚠️ Result: Tested Positive for Diabetes")
    else:
        st.success("✅ Result: Tested Negative for Diabetes")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")

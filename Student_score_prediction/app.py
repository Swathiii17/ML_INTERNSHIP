import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Title
# -----------------------------
st.title("📘 Student Score Prediction (PKL Model)")

# -----------------------------
# Load Pickle Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("Student_score_prediction/stu_score.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -----------------------------
# User Input
# -----------------------------
hours = st.number_input(
    "Enter study hours",
    min_value=0.0,
    max_value=24.0,
    step=0.5
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Score"):
    input_data = np.array([[hours]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Score: {prediction[0]:.2f}")

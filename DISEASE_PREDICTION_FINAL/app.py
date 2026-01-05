import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Load model, scaler, encoder & symptoms
# -----------------------------
output_dir = "/content/drive/MyDrive/extra_need/h5_pkl/disease_final/"

with open(os.path.join(output_dir, "disease_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(output_dir, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(output_dir, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(output_dir, "all_symptoms.pkl"), "rb") as f:
    all_symptoms = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🩺 Disease Prediction Based on Symptoms")

st.write("Select your symptoms from the list below:")

# Multi-select symptoms
selected_symptoms = st.multiselect(
    "Symptoms", options=all_symptoms
)

# Predict button
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom!")
    else:
        # Prepare input vector
        input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
        for symptom in selected_symptoms:
            input_vector.at[0, symptom] = 1
        
        # Scale
        input_scaled = scaler.transform(input_vector)
        
        # Predict
        pred_enc = model.predict(input_scaled)[0]
        pred_disease = le.inverse_transform([pred_enc])[0]
        
        st.success(f"Predicted Disease: **{pred_disease}**")

import pandas as pd
import streamlit as st
import pickle
import os

# -----------------------------
# Paths
# -----------------------------
output_dir = "/content/drive/MyDrive/extra_need/h5_pkl/disease_final/"
dataset_path = "/content/drive/MyDrive/extra_need/Datas/disease_final/dataset.csv"

# -----------------------------
# -----------------------------
# Load model, scaler, encoder
# -----------------------------
with open("DISEASE_PREDICTION_FINAL/disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("DISEASE_PREDICTION_FINAL/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("DISEASE_PREDICTION_FINAL/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Recreate all symptoms from dataset
# -----------------------------
df = pd.read_csv(dataset_path)
symptom_cols = df.columns[1:]
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())
all_symptoms = sorted(list(all_symptoms))  # sorted for easier search

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🩺 Disease Prediction Based on Symptoms")
st.write("Select your symptoms from the list below:")

# Split symptoms into columns for easier selection
cols = st.columns(3)  # show 3 columns of checkboxes
selected_symptoms = []

for i, symptom in enumerate(all_symptoms):
    if i % 3 == 0:
        col = cols[0]
    elif i % 3 == 1:
        col = cols[1]
    else:
        col = cols[2]
    if col.checkbox(symptom):
        selected_symptoms.append(symptom)

# Predict button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom!")
    else:
        input_vector = pd.DataFrame(0, index=[0], columns=all_symptoms)
        for symptom in selected_symptoms:
            input_vector.at[0, symptom] = 1
        
        input_scaled = scaler.transform(input_vector)
        pred_enc = model.predict(input_scaled)[0]
        pred_disease = le.inverse_transform([pred_enc])[0]
        st.success(f"Predicted Disease: **{pred_disease}**")

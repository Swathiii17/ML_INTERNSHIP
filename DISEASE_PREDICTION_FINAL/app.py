import pandas as pd
import streamlit as st
import pickle

# -----------------------------
# Load ML objects
# -----------------------------
with open("DISEASE_PREDICTION_FINAL/disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("DISEASE_PREDICTION_FINAL/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("DISEASE_PREDICTION_FINAL/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Load datasets for info
# -----------------------------
df_data = pd.read_csv("DISEASE_PREDICTION_FINAL/dataset.csv")
df_desc = pd.read_csv("DISEASE_PREDICTION_FINAL/symptom_Description.csv")
df_prec = pd.read_csv("DISEASE_PREDICTION_FINAL/symptom_precaution.csv")
df_sev = pd.read_csv("DISEASE_PREDICTION_FINAL/Symptom-severity.csv")

# -----------------------------
# Build symptom list
# -----------------------------
symptom_cols = df_data.columns[1:]
all_symptoms = sorted(list(symptom_cols))

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Disease Predictor", layout="wide")

st.title("🩺 Disease Prediction Based on Symptoms")
st.write("Select symptoms:")

# --- Searchable multiselect (better than checkboxes) ---
selected_symptoms = st.multiselect(
    "Type and select",
    options=all_symptoms
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Disease"):

    if not selected_symptoms:
        st.warning("Select at least one symptom")

    else:
       trained_symptoms = list(scaler.feature_names_in_)

       input_vector = pd.DataFrame(0, index=[0], columns=trained_symptoms)

       for s in selected_symptoms:
          if s in trained_symptoms:
             input_vector.at[0, s] = 1

       input_scaled = scaler.transform(input_vector)

       pred_enc = model.predict(input_scaled)[0]
       disease = le.inverse_transform([pred_enc])[0]

       st.success(f"Predicted Disease: {disease}")

       

        # -----------------------------
        # SHOW SEVERITY SCORE
        # -----------------------------
        total = 0
        for s in selected_symptoms:
            row = df_sev[df_sev["Symptom"] == s]
            if not row.empty:
                total += int(row["Severity"].values[0])

        st.subheader("📊 Severity Level")
        st.write(f"Score: {total}")

        if total < 10:
            st.write("Mild")
        elif total < 20:
            st.write("Moderate")
        else:
            st.write("Severe – consult doctor")

        # -----------------------------
        # DESCRIPTION
        # -----------------------------
        st.subheader("🧾 Disease Description")

        drow = df_desc[df_desc["Disease"] == disease]

        if not drow.empty:
            st.write(drow["Discription"].values[0])
        else:
            st.write("No description")

        # -----------------------------
        # PRECAUTIONS
        # -----------------------------
        st.subheader("💊 Precautions")

        prow = df_prec[df_prec["Disease"] == disease]

        if not prow.empty:
            for i in range(1, 5):
                col = f"Precaution_{i}"
                if col in prow.columns:
                    st.write(f"- {prow[col].values[0]}")
        else:
            st.write("No precautions")

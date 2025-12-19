import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# Title
# -----------------------------
st.title("📘 Student Score Prediction App")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    with open("Student_score_prediction/stu_score.pkl", "rb") as file:
        data = pickle.load(file)
    return data

df = load_data()   # df is ONLY DataFrame

# -----------------------------
# Show Dataset
# -----------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# -----------------------------
# Prepare Data
# -----------------------------
X = df[["Hours"]]
y = df["Scores"]

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()   # model is ML model
model.fit(X, y)

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
    prediction = model.predict([[hours]])
    st.success(f"Predicted Score: {prediction[0]:.2f}")

# -----------------------------
# Graph
# -----------------------------
fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, model.predict(X))
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Scores")

st.pyplot(fig)

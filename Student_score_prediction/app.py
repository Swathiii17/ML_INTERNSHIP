import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# App Title
# -----------------------------
st.title("📘 Student Score Prediction App")
st.write("Predict student scores based on study hours using Linear Regression")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_resource

def load_data():
    with open("stu_score.pkl", "rb") as file:
        data = pickle.load(file)
    return data

df = load_data()

# -----------------------------
# Show Dataset
# -----------------------------
if st.checkbox("Show Dataset"):
    st.write(df)

# -----------------------------
# Prepare Data
# -----------------------------
X = df[["Hours"]]
y = df["Scores"]

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# User Input
# -----------------------------
st.subheader("🔢 Enter Study Hours")
hours = st.number_input("Hours Studied:", min_value=0.0, max_value=24.0, step=0.5)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Score"):
    prediction = model.predict([[hours]])
    st.success(f"📊 Predicted Score: {prediction[0]:.2f}")

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📈 Regression Graph")

fig, ax = plt.subplots()
ax.scatter(X, y, color="blue", label="Actual Data")
ax.plot(X, model.predict(X), color="red", label="Regression Line")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Scores")
ax.legend()

st.pyplot(fig)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

# Load model safely
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except:
    st.error("❌ Model files not found. Please upload model.pkl and scaler.pkl")
    st.stop()

# Title
st.title("🍷 Wine Quality Prediction App")
st.markdown("### Predict whether your wine is Low, Medium, or High quality")

# Sidebar
st.sidebar.header("🧪 Enter Wine Features")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 15.0, 5.0)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.05)
free_sulfur = st.sidebar.slider("Free Sulfur Dioxide", 1, 70, 15)
total_sulfur = st.sidebar.slider("Total Sulfur Dioxide", 6, 300, 50)
density = st.sidebar.slider("Density", 0.990, 1.005, 0.995)
ph = st.sidebar.slider("pH", 2.5, 4.5, 3.2)
sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.7)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0)

# Input
input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                        residual_sugar, chlorides, free_sulfur,
                        total_sulfur, density, ph, sulphates, alcohol]])

# Prediction
if st.button("🔍 Predict Wine Quality"):
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)

    st.subheader("🎯 Prediction Result")

    if prediction[0] == 2:
        st.success("🍷 High Quality Wine")
    elif prediction[0] == 1:
        st.warning("🍷 Medium Quality Wine")
    else:
        st.error("🍷 Low Quality Wine")

# Feature Importance
st.write("---")
st.subheader("📊 Feature Importance")

try:
    importance = model.feature_importances_
    features = ["fixed acidity", "volatile acidity", "citric acid",
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH",
                "sulphates", "alcohol"]

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    fig = px.bar(df_imp, x="Importance", y="Feature",
                 orientation="h",
                 color="Importance")

    st.plotly_chart(fig, use_container_width=True)

except:
    st.info("Feature importance not available.")
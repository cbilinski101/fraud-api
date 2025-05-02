import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fraud Predictor", layout="centered")
st.title("🛡️ Fraud Detection Predictor")
st.caption("Enter transaction details to predict potential fraud.")

MODEL_PATH = "optimized_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Input fields
amt = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0)
city_pop = st.number_input("🏙️ City Population", min_value=0, value=5000)
age = st.slider("👤 Cardholder Age", 18, 100, 35)
hour_of_day = st.slider("🕒 Hour of Transaction", 0, 23, 12)
day_of_week = st.selectbox("📅 Day of Week", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

# Single category dropdown
category_options = {
    "Entertainment": "category_entertainment",
    "Food & Dining": "category_food_dining",
    "Gas & Transport": "category_gas_transport",
    "Grocery (Net)": "category_grocery_net",
    "Grocery (POS)": "category_grocery_pos",
    "Health & Fitness": "category_health_fitness",
    "Home": "category_home",
    "Kids & Pets": "category_kids_pets",
    "Misc (Net)": "category_misc_net",
    "Misc (POS)": "category_misc_pos",
    "Personal Care": "category_personal_care",
    "Shopping (Net)": "category_shopping_net",
    "Shopping (POS)": "category_shopping_pos",
    "Travel": "category_travel"
}

category_selected = st.selectbox("📦 Transaction Category", list(category_options.keys()))
category_vector = {v: (1 if v == category_options[category_selected] else 0) for v in category_options.values()}

# Prepare input
input_data = {
    "amt": amt,
    "city_pop": city_pop,
    "age": age,
    "day_of_week": day_of_week,
    "hour_of_day": hour_of_day
}
input_data.update(category_vector)

# Ensure feature order matches training
input_columns = [
    "amt", "city_pop", "age", "day_of_week", "hour_of_day",
    "category_entertainment", "category_food_dining", "category_gas_transport",
    "category_grocery_net", "category_grocery_pos", "category_health_fitness",
    "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos",
    "category_personal_care", "category_shopping_net", "category_shopping_pos", "category_travel"
]

# Predict
if st.button("🔍 Predict Fraud"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[input_columns]  # Reorder columns
    prediction = model.predict(input_df)[0]
    result = "🔴 Fraudulent Transaction" if prediction == 1 else "🟢 Legitimate Transaction"
    st.success(f"**Prediction Result:** {result}")

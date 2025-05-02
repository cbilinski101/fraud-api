import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Predictor", layout="wide")

MODEL_PATH = "optimized_model.joblib"
PRED_HISTORY_FILE = "prediction_history.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🛡️ Fraud Detection Dashboard")
st.markdown("Use this tool to predict if a credit card transaction is fraudulent based on user input.")

# Sidebar Explanation
with st.sidebar:
    st.header("🧠 How It Works")
    st.write("This app uses a trained Random Forest model to analyze transaction data and predict the likelihood of fraud.")
    st.markdown("**Inputs include:**")
    st.markdown("- Transaction amount, time, and cardholder age")
    st.markdown("- One transaction category (e.g., Travel, Grocery)")
    st.markdown("**Output:**")
    st.markdown("- A fraud or legit prediction")
    st.markdown("- Running statistics and visual breakdown")

# Input fields
with st.expander("📝 Enter Transaction Details"):
    col1, col2 = st.columns(2)
    amt = col1.number_input("💰 Transaction Amount", min_value=0.0, value=100.0)
    city_pop = col2.number_input("🏙️ City Population", min_value=0, value=5000)
    age = col1.slider("👤 Cardholder Age", 18, 100, 35)
    hour_of_day = col2.slider("🕒 Hour of Transaction", 0, 23, 12)
    day_of_week = col1.selectbox("📅 Day of Week", list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

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

# Build input data
input_data = {
    "amt": amt,
    "city_pop": city_pop,
    "age": age,
    "day_of_week": day_of_week,
    "hour_of_day": hour_of_day
}
input_data.update(category_vector)

# Column order for prediction
input_columns = [
    "amt", "city_pop", "age", "day_of_week", "hour_of_day",
    "category_entertainment", "category_food_dining", "category_gas_transport",
    "category_grocery_net", "category_grocery_pos", "category_health_fitness",
    "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos",
    "category_personal_care", "category_shopping_net", "category_shopping_pos", "category_travel"
]

# Predict button
if st.button("🔍 Predict Fraud"):
    input_df = pd.DataFrame([input_data])[input_columns]
    prediction = model.predict(input_df)[0]

    result = "🔴 Fraudulent Transaction" if prediction == 1 else "🟢 Legitimate Transaction"
    st.success(f"**Prediction Result:** {result}")

    # Save prediction
    new_row = pd.DataFrame([{**input_data, "prediction": int(prediction)}])
    if os.path.exists(PRED_HISTORY_FILE):
        history_df = pd.read_csv(PRED_HISTORY_FILE)
        history_df = pd.concat([history_df, new_row], ignore_index=True)
    else:
        history_df = new_row
    history_df.to_csv(PRED_HISTORY_FILE, index=False)

# Show stats if history exists
if os.path.exists(PRED_HISTORY_FILE):
    stats_df = pd.read_csv(PRED_HISTORY_FILE)
    fraud_count = stats_df["prediction"].sum()
    legit_count = len(stats_df) - fraud_count
    total = len(stats_df)
    fraud_rate = (fraud_count / total) * 100 if total else 0

    st.markdown("### 📊 Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔢 Total Predictions", total)
    col2.metric("🟢 Legitimate", legit_count)
    col3.metric("🔴 Fraudulent", fraud_count)

    # Charts
    st.markdown("#### 🔎 Fraud vs. Legitimate Chart")
    fig, ax = plt.subplots()
    ax.bar(["Legit", "Fraud"], [legit_count, fraud_count], color=["green", "red"])
    ax.set_ylabel("Number of Predictions")
    st.pyplot(fig)

    with st.expander("🧾 View Recent Predictions"):
        st.dataframe(stats_df.tail(10), use_container_width=True)
else:
    st.info("No predictions made yet. Submit a transaction to get started.")

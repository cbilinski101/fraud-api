
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import re
import time

st.set_page_config(page_title="Fraud Predictor Dashboard", layout="wide")

MODEL_PATH = "optimized_model.joblib"
PRED_HISTORY_FILE = "prediction_history.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🛡️ Fraud Detection Dashboard")
st.caption("Predict whether a credit card transaction may be fraudulent.")

with st.sidebar:
    st.header("🧠 How It Works")
    st.write("This app uses a trained LightGBM model to predict fraud risk.")
    st.markdown("### Features")
    st.markdown("- Transaction inputs")
    st.markdown("- Fraud prediction")
    st.markdown("- Live prediction tracking")
    st.markdown("- Model comparison visuals")

# ---------------- INPUT FORM ----------------

with st.expander("📝 Transaction Entry", expanded=True):
    col1, col2 = st.columns(2)

    amt = col1.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    city_pop = col2.number_input("City Population", min_value=0, value=5000)

    age = col1.slider("Cardholder Age", 18, 100, 35)
    hour = col2.slider("Hour of Day", 0, 23, 12)

    day = col1.selectbox(
        "Day of Week",
        list(range(7)),
        format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
    )

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

    selected_cat = st.selectbox("Transaction Category", list(category_options.keys()))
    cat_vector = {v: int(v == category_options[selected_cat]) for v in category_options.values()}

input_data = {
    "amt": amt,
    "city_pop": city_pop,
    "age": age,
    "day_of_week": day,
    "hour_of_day": hour
}

input_data.update(cat_vector)

input_cols = [
    "amt","city_pop","age","day_of_week","hour_of_day",
    "category_entertainment","category_food_dining","category_gas_transport",
    "category_grocery_net","category_grocery_pos","category_health_fitness",
    "category_home","category_kids_pets","category_misc_net","category_misc_pos",
    "category_personal_care","category_shopping_net","category_shopping_pos","category_travel"
]

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Fraud"):
    df = pd.DataFrame([input_data])[input_cols]

    pred = model.predict(df)[0]

    try:
        prob = model.predict_proba(df)[0][1]
    except:
        prob = None

    label = "🟢 Legitimate" if pred == 0 else "🔴 Fraudulent"

    st.subheader(f"Prediction: {label}")

    if prob is not None:
        st.progress(float(prob))
        st.write(f"Fraud Probability: **{prob:.2%}**")

    row = pd.DataFrame([{**input_data, "prediction": int(pred)}])

    if os.path.exists(PRED_HISTORY_FILE):
        history = pd.read_csv(PRED_HISTORY_FILE)
        history = pd.concat([history,row], ignore_index=True)
    else:
        history = row

    history.to_csv(PRED_HISTORY_FILE, index=False)

# ---------------- DASHBOARD METRICS ----------------

if os.path.exists(PRED_HISTORY_FILE):

    stats = pd.read_csv(PRED_HISTORY_FILE)

    fraud = stats["prediction"].sum()
    legit = len(stats) - fraud
    total = len(stats)

    st.markdown("## 📊 Prediction Summary")

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Predictions", total)
    col2.metric("Legitimate", legit)
    col3.metric("Fraudulent", fraud)

    st.bar_chart(pd.DataFrame({
        "Fraud":[fraud],
        "Legit":[legit]
    }))

    with st.expander("Recent Predictions"):
        st.dataframe(stats.tail(10), use_container_width=True)

# ---------------- SLIDESHOW ----------------

st.markdown("## 🎞️ Model Performance Slides")

slide_dir = "slides"

if os.path.exists(slide_dir):

    def natural_sort_key(filename):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', filename)]

    image_files = sorted(
        [f for f in os.listdir(slide_dir) if f.endswith(".png")],
        key=natural_sort_key
    )

    if len(image_files) > 0:

        if "slide_index" not in st.session_state:
            st.session_state.slide_index = 0

        img_path = os.path.join(slide_dir, image_files[st.session_state.slide_index])

        st.image(Image.open(img_path), use_container_width=True)

        col1,col2 = st.columns(2)

        if col1.button("⬅ Previous") and st.session_state.slide_index > 0:
            st.session_state.slide_index -= 1

        if col2.button("Next ➡") and st.session_state.slide_index < len(image_files)-1:
            st.session_state.slide_index += 1

else:
    st.info("Add images to the /slides folder to show model comparison charts.")

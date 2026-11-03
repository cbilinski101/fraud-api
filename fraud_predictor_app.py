import os
import streamlit as st

st.set_page_config(page_title="Fraud Risk Intelligence Dashboard", layout="wide")
st.title("🛡️ Fraud Risk Intelligence Dashboard")
st.caption("Interactive ML dashboard predicting fraudulent credit card transactions using LightGBM.")
st.caption("Portfolio demonstration — use fictional transaction details.")

# Render the introduction before importing data tools or loading the model.
import pandas as pd

MODEL_PATH = "optimized_model.joblib"
HISTORY_FILE = "prediction_history.csv"


@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)


with st.sidebar:
    st.header("Model Overview")
    st.write(
        "This dashboard uses a LightGBM machine learning model trained to detect potentially "
        "fraudulent credit card transactions."
    )
    st.markdown("### Features")
    st.markdown("- Real-time fraud prediction")
    st.markdown("- Fraud probability score")
    st.markdown("- Prediction analytics dashboard")
    st.markdown("- Feature importance visualization")

categories = {
    "Entertainment": "category_entertainment",
    "Food": "category_food_dining",
    "Gas": "category_gas_transport",
    "Grocery Net": "category_grocery_net",
    "Grocery POS": "category_grocery_pos",
    "Health": "category_health_fitness",
    "Home": "category_home",
    "Kids/Pets": "category_kids_pets",
    "Misc Net": "category_misc_net",
    "Misc POS": "category_misc_pos",
    "Personal Care": "category_personal_care",
    "Shopping Net": "category_shopping_net",
    "Shopping POS": "category_shopping_pos",
    "Travel": "category_travel",
}
cols = ["amt", "city_pop", "age", "day_of_week", "hour_of_day", *categories.values()]

st.subheader("Transaction Input")
with st.form("transaction_input"):
    col1, col2, col3 = st.columns(3)
    amt = col1.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    age = col2.slider("Cardholder Age", 18, 100, 35)
    city_pop = col3.number_input("City Population", min_value=0, value=5000)
    hour = col1.slider("Hour of Day", 0, 23, 12)
    day = col2.selectbox("Day of Week", range(7))
    selected = col3.selectbox("Category", list(categories.keys()))
    submitted = st.form_submit_button("Predict Fraud Risk")

if submitted:
    model = load_model()
    data = {
        "amt": amt,
        "city_pop": city_pop,
        "age": age,
        "day_of_week": day,
        "hour_of_day": hour,
        **{v: int(v == categories[selected]) for v in categories.values()},
    }
    df = pd.DataFrame([data])[cols]
    pred = model.predict(df)[0]
    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception:
        prob = None
    st.session_state["last_prediction"] = {"prediction": int(pred), "probability": prob}

    row = pd.DataFrame([{**data, "prediction": int(pred)}])
    if os.path.exists(HISTORY_FILE):
        hist = pd.concat([pd.read_csv(HISTORY_FILE), row], ignore_index=True)
    else:
        hist = row
    hist.to_csv(HISTORY_FILE, index=False)

# Keep the last result visible when the chart control triggers a rerun.
if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    label = "Potential Fraud" if result["prediction"] == 1 else "Legitimate Transaction"
    st.subheader(f"Prediction: {label}")
    if result["probability"] is not None:
        st.progress(result["probability"])
        st.write(f"Fraud Probability: **{result['probability']:.2%}**")

if os.path.exists(HISTORY_FILE):
    hist = pd.read_csv(HISTORY_FILE)
    fraud = hist["prediction"].sum()
    legit = len(hist) - fraud
    st.subheader("Prediction Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", len(hist))
    c2.metric("Legitimate", legit)
    c3.metric("Fraud", fraud)
    chart_df = pd.DataFrame({"type": ["Fraud", "Legitimate"], "count": [fraud, legit]})
    st.bar_chart(chart_df.set_index("type"))
    st.subheader("Recent Predictions")
    st.dataframe(hist.tail(10), use_container_width=True)

st.subheader("Model Feature Importance")
if st.checkbox("Show feature importance chart", value=False):
    try:
        model = load_model()
        df_imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
        df_imp = df_imp.sort_values("importance", ascending=False).head(10)
        st.bar_chart(df_imp, x="feature", y="importance", horizontal=True, sort="-importance")
    except Exception:
        st.info("Feature importance unavailable for this model.")
else:
    st.caption("Load the chart when you want to explore which inputs the model weighs most.")

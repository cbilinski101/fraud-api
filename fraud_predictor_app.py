
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Risk Intelligence Dashboard", layout="wide")

MODEL_PATH = "optimized_model.joblib"
HISTORY_FILE = "prediction_history.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🛡️ Fraud Risk Intelligence Dashboard")
st.caption("Interactive ML dashboard predicting fraudulent credit card transactions using LightGBM.")

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

st.subheader("Transaction Input")

col1, col2, col3 = st.columns(3)

amt = col1.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
age = col2.slider("Cardholder Age", 18, 100, 35)
city_pop = col3.number_input("City Population", min_value=0, value=5000)

hour = col1.slider("Hour of Day", 0, 23, 12)
day = col2.selectbox("Day of Week", range(7))

categories = {
"Entertainment":"category_entertainment",
"Food":"category_food_dining",
"Gas":"category_gas_transport",
"Grocery Net":"category_grocery_net",
"Grocery POS":"category_grocery_pos",
"Health":"category_health_fitness",
"Home":"category_home",
"Kids/Pets":"category_kids_pets",
"Misc Net":"category_misc_net",
"Misc POS":"category_misc_pos",
"Personal Care":"category_personal_care",
"Shopping Net":"category_shopping_net",
"Shopping POS":"category_shopping_pos",
"Travel":"category_travel"
}

selected = col3.selectbox("Category", list(categories.keys()))
cat_vector = {v:int(v == categories[selected]) for v in categories.values()}

data = {
"amt": amt,
"city_pop": city_pop,
"age": age,
"day_of_week": day,
"hour_of_day": hour
}

data.update(cat_vector)

cols=[
"amt","city_pop","age","day_of_week","hour_of_day",
"category_entertainment","category_food_dining","category_gas_transport",
"category_grocery_net","category_grocery_pos","category_health_fitness",
"category_home","category_kids_pets","category_misc_net","category_misc_pos",
"category_personal_care","category_shopping_net","category_shopping_pos","category_travel"
]

if st.button("Predict Fraud Risk"):

    df = pd.DataFrame([data])[cols]

    pred = model.predict(df)[0]

    try:
        prob = model.predict_proba(df)[0][1]
    except:
        prob = None

    label = "Legitimate Transaction"
    if pred == 1:
        label = "Potential Fraud"

    st.subheader(f"Prediction: {label}")

    if prob:
        st.progress(float(prob))
        st.write(f"Fraud Probability: **{prob:.2%}**")

    row = pd.DataFrame([{**data,"prediction":int(pred)}])

    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        hist = pd.concat([hist,row],ignore_index=True)
    else:
        hist = row

    hist.to_csv(HISTORY_FILE,index=False)

if os.path.exists(HISTORY_FILE):

    hist = pd.read_csv(HISTORY_FILE)

    fraud = hist["prediction"].sum()
    legit = len(hist) - fraud

    st.subheader("Prediction Dashboard")

    c1,c2,c3 = st.columns(3)

    c1.metric("Total Predictions", len(hist))
    c2.metric("Legitimate", legit)
    c3.metric("Fraud", fraud)

    chart_df = pd.DataFrame({
        "type":["Fraud","Legitimate"],
        "count":[fraud,legit]
    })

    st.bar_chart(chart_df.set_index("type"))

    st.subheader("Recent Predictions")
    st.dataframe(hist.tail(10), use_container_width=True)

st.subheader("Model Feature Importance")

try:
    importances = model.feature_importances_
    features = cols

    df_imp = pd.DataFrame({
        "feature":features,
        "importance":importances
    }).sort_values("importance",ascending=False).head(10)

    fig,ax = plt.subplots()
    ax.barh(df_imp["feature"], df_imp["importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

except:
    st.info("Feature importance unavailable for this model.")

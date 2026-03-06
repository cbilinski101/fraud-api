
# Fraud Predictor Dashboard

Streamlit dashboard for predicting fraudulent credit card transactions using a LightGBM model.

## Run Locally

pip install -r requirements.txt
streamlit run fraud_predictor_app.py

## Deploy on Render

Build Command:
pip install -r requirements.txt

Start Command:
streamlit run fraud_predictor_app.py --server.port $PORT --server.address 0.0.0.0

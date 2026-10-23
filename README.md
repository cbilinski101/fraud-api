
# 🛡️ Fraud Risk Intelligence Dashboard

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

An interactive **machine learning dashboard** that predicts fraudulent credit card transactions using a **LightGBM classification model**.

---

# 🚀 Live Deployment

**Main maintained application:** [Open Fraud Risk Intelligence Dashboard](https://fraud-risk-dashboard.onrender.com/)

Use this URL for testing, sharing, and future updates. This is the selected main deployment; other Render URLs are duplicate or earlier deployments.

This is a portfolio demonstration. Use fictional transaction data. Prediction history is currently stored on the service instance, shared across visitors, and may reset when Render restarts or redeploys it.

---

# 📊 Features

• Real-time fraud prediction  
• Fraud probability score  
• Prediction analytics dashboard  
• Feature importance visualization  

---

# 🧠 Model

Algorithm: **LightGBM Classifier**

Advantages:

• Handles imbalanced fraud datasets well  
• Fast predictions  
• High performance for classification problems  

---

# 🗂 Project Structure

fraud-risk-dashboard

fraud_predictor_app.py  
optimized_model.joblib  
requirements.txt  
render.yaml  
slides/  
README.md  

---

# ⚙️ Run Locally

pip install -r requirements.txt

streamlit run fraud_predictor_app.py

---

# ☁️ Deploy on Render

1. Push this repository to GitHub
2. Go to https://render.com
3. Click **New Web Service**
4. Connect your repository
5. Render will detect `render.yaml`
6. Click Deploy

---

# 👩‍💻 Author

Christine Bilinski  
GitHub: https://github.com/cbilinski101  
LinkedIn: https://linkedin.com/in/christine-b-19367b31b

---

# 📜 License

MIT License

## Dependency maintenance

Direct application dependencies are pinned to versions from the verified October 23, 2026 deployment. Certifi retains a security minimum of `2026.10.19` so builds can receive newer certificate bundles. Transitive dependencies are not fully locked. After dependency changes, verify deployment health and test predictions using fictional data.

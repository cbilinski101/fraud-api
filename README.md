# 🛡️ Fraud Predictor Dashboard

An interactive Streamlit dashboard for predicting fraudulent credit card transactions using a trained LightGBM model. Built with Python, Streamlit, and scikit-learn, and deployable via Render.

![Render Deploy](https://img.shields.io/badge/Render.com-Live-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)

---

## 🚀 Live Demo

🌐 [Launch the App on Render](https://fraud-predictor-dashboard.onrender.com)  
*(Replace this URL with your actual Render link after deployment)*

---

## 📊 Features

- 📝 User-friendly input form for transaction data
- 📦 One dropdown to select transaction category
- 🔍 Predicts whether a transaction is **fraudulent** or **legitimate**
- 📈 Real-time dashboard statistics:
  - Total number of predictions
  - Fraud vs. Legit count
  - Dynamic bar chart
  - View recent predictions in a table
- 🎞️ Slideshow of model performance comparison curves
- 🧠 Sidebar explanation of how the model works

---

## 🔧 How to Deploy (GitHub + Render)

1. **Fork or clone this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/fraud-predictor-dashboard.git
   ```
2. **Go to [Render.com](https://render.com)** → Click **New Web Service**
3. Connect your GitHub and select this repository
4. Render will auto-detect the `render.yaml` file and prefill:
   - **Build command**:  
     ```
     pip install -r requirements.txt
     ```
   - **Start command**:  
     ```
     streamlit run fraud_predictor_app.py --server.port 10000 --server.address 0.0.0.0
     ```
5. Choose the **Free plan**
6. Click **Create Web Service**
7. Done! 🎉 Your app is live!

---

## 📁 Project Structure

```
📦 fraud-predictor-dashboard
├── fraud_predictor_app.py       # Streamlit app with prediction logic and UI
├── optimized_model.joblib       # Trained LightGBM fraud detection model (Model 10)
├── requirements.txt             # Required Python libraries
├── render.yaml                  # Render deployment config
├── README.md                    # Project overview (this file)
└── /slides                      # Precision-recall and ROC comparison images
```

---

## 🧠 Model Details (Model 10 - LightGBM)

**Model Type:** LightGBM  
**Optimization:** Bayesian hyperparameter tuning  
**Training Notebook:** `Part2_OptimizedFraudDetection_Colab.ipynb`  
**Performance Summary (vs. Original Model):**

### 📈 ROC Curve Insights
- ✅ **AUC - Original Model:** 0.894  
- ✅ **AUC - Model 10:** 0.983  
- Model 10 (green) significantly outperforms the Original (red) in distinguishing fraud from non-fraud

### 📊 Precision-Recall Curve Insights
- ⚠️ Model 10 prioritizes **recall** over precision
- Original model has higher precision at some mid-recall values (~0.7–0.9)
- Model 10 catches more fraud at the cost of more false positives (a tradeoff often worth making)

---

### 🚀 Summary: Why Model 10 Is Better
✔ Higher AUC (0.983 vs. 0.894)  
✔ Higher Recall → better at catching fraud  
✔ Bayesian-tuned LightGBM model → faster, more precise

---

## 🙋‍♀️ Created By

**Christine Bilinski**  
📧 [cbilinski101@gmail.com](mailto:cbilinski101@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/christine-b-19367b31b)  
🔗 [GitHub](https://github.com/cbilinski101)

---

## 📝 License

This project is licensed under the MIT License.

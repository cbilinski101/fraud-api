# 🛡️ Fraud Predictor Dashboard

An interactive Streamlit dashboard for predicting fraudulent credit card transactions using a trained machine learning model. Built with Python, Streamlit, and scikit-learn, and deployable via Render.

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
├── optimized_model.joblib       # Pre-trained fraud detection model (Model 10)
├── requirements.txt             # Required Python libraries
├── render.yaml                  # Render deployment config
└── README.md                    # Project overview (this file)
```

---

## 🧠 Model Details (Model 10)

**Model:** `RandomForestClassifier`  
**Training Notebook:** `Part2_OptimizedFraudDetection_Colab.ipynb`  
**Final Features Used:**
```
['amt', 'city_pop', 'age', 'day_of_week', 'hour_of_day',
 'category_entertainment', 'category_food_dining', 'category_gas_transport',
 'category_grocery_net', 'category_grocery_pos', 'category_health_fitness',
 'category_home', 'category_kids_pets', 'category_misc_net', 'category_misc_pos',
 'category_personal_care', 'category_shopping_net', 'category_shopping_pos',
 'category_travel']
```

**Performance Summary (Model 10 vs. Original):**
- Accuracy: 0.994  
- Precision: 0.84  
- Recall: 0.92  
- F1 Score: 0.88  

✅ Model 10 showed superior fraud detection recall with minimal false positives compared to earlier models.

---

## 🙋‍♀️ Created By

**Christine Bilinski**  
📧 [cbilinski101@gmail.com](mailto:cbilinski101@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/christine-b-19367b31b)  
🔗 [GitHub](https://github.com/cbilinski101)

---

## 📝 License

This project is licensed under the MIT License.

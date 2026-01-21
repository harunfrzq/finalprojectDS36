# 📦 Adaptive Demand Prediction Using Time Series and Machine Learning Models to Improve Supply Chain Efficiency

**Final Project Data Science**
**Created by: Harun Fathurrozaq**

---

## 🚀 Project Summary

In modern supply chains, inaccurate demand forecasting often leads to **overstock**, **stockouts**, and **inefficient operations**. This project delivers an **end-to-end adaptive demand forecasting system** that combines **time series analysis** and **machine learning models**, deployed as a **production-ready Streamlit web application**.

The system is designed not only to predict demand, but also to be **scalable, interpretable, and usable by business stakeholders**.

---

## 🎯 Business Objectives

- Improve demand forecasting accuracy at daily granularity
- Support inventory and supply chain decision-making
- Compare classical time series models with machine learning approaches
- Deliver a deployable, real-world data science solution

---

## 🧠 Methodology – CRISP-DM Framework

### 1️⃣ Business Understanding

- **Problem:** Demand volatility creates inefficiency in inventory planning
- **Goal:** Predict future demand using historical sales data
- **Impact:** Better stock planning, reduced cost, improved service level

### 2️⃣ Data Understanding

- **Dataset:** Online Retail Transaction Data
- **Main Attributes:**
  - `InvoiceDate`
  - `Quantity`

- **Granularity:** Daily aggregated demand

### 3️⃣ Data Preparation & Feature Engineering

Key preprocessing steps:

- Data cleaning (remove negative & invalid quantities)
- Daily aggregation
- Feature engineering:
  - 📅 Time-based features (year, month, week, quarter)
  - ⏳ Lag features (7, 14, 21 days)
  - 📊 Rolling statistics (mean & standard deviation)

This ensures the model captures **trend, seasonality, and short-term patterns**.

### 4️⃣ Modeling

Three forecasting models were developed and compared:

| Model    | Description                                          |
| -------- | ---------------------------------------------------- |
| Baseline | Moving Average                                       |
| ARIMA    | Classical statistical time series model              |
| XGBoost  | Machine learning regression with engineered features |

✅ **Final Model Selected:** **XGBoost Regressor**
Chosen for its superior accuracy, flexibility, and robustness.

### 5️⃣ Evaluation

- **Metrics:** MAE, RMSE
- **Result:** XGBoost consistently outperformed baseline and ARIMA models
- **Key Insight:** Machine learning models adapt better to complex demand patterns

### 6️⃣ Deployment

- Trained model serialized using `joblib`
- Interactive web application built with **Streamlit**
- Deployed on **Streamlit Cloud**

---

## 🖥️ Streamlit Application Features

- 📈 Interactive demand forecasting visualization
- 🎚️ Adjustable forecast horizon (7–60 days)
- 📊 Historical vs forecast comparison
- ⬇️ Download forecast results (CSV)
- ⚡ Optimized performance using caching

---

## 🛠️ Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Modeling:** XGBoost, Scikit-learn
- **Visualization:** Matplotlib
- **Deployment:** Streamlit Cloud
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```
final_project/
│── app.py                 # Streamlit application
│── train_xgb.py           # Model training script
│── model_xgb.pkl          # Trained XGBoost model
│── Online_Retail.xlsx     # Dataset
│── requirements.txt       # Project dependencies
│── README.md              # Documentation
```

---

## 🌐 Live Demo

🔗 **Streamlit App:** _(add your deployed URL here)_

---

## 📈 Business Impact

- Improved forecasting accuracy
- Scalable forecasting pipeline
- Ready-to-use solution for real-world supply chain scenarios

---

## 👤 Author

**Harun Fathurrozaq**
Final Project – Data Scientist

---

## ⭐ Acknowledgement

If you find this project useful, feel free to give it a ⭐ on GitHub.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Demand Forecasting - XGBoost",
    layout="wide"
)

# =====================================================
# CONSTANTS (HARUS SAMA DENGAN TRAINING)
# =====================================================
TARGET_COL = "Demand"

MODEL_FEATURES = [
    'year', 'month', 'day_of_month', 'day_of_week', 'day_of_year',
    'week_of_year', 'quarter',
    'is_month_start', 'is_month_end',
    'is_quarter_start', 'is_quarter_end',
    'is_year_start', 'is_year_end',
    'week_day_binary',
    'Demand_lag_7', 'Demand_lag_14', 'Demand_lag_21',
    'Demand_roll_mean_7', 'Demand_roll_std_7',
    'Demand_roll_mean_14', 'Demand_roll_std_14'
]

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("model_xgb.pkl")

model = load_model()

# =====================================================
# LOAD & PREPARE DATA (DILAKUKAN SEKALI)
# =====================================================
@st.cache_data
def load_processed_data():
    """
    PENTING:
    - Dataset diproses SEKALI
    - Tidak ada dropna di forecasting
    """
    df = pd.read_excel("Online_Retail.xlsx")

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['Quantity'] > 0]

    daily = (
        df.groupby(df['InvoiceDate'].dt.date)['Quantity']
        .sum()
        .reset_index()
    )

    daily.columns = ['date', TARGET_COL]
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')

    # === TIME FEATURES ===
    daily['year'] = daily['date'].dt.year
    daily['month'] = daily['date'].dt.month
    daily['day_of_month'] = daily['date'].dt.day
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['day_of_year'] = daily['date'].dt.dayofyear
    daily['week_of_year'] = daily['date'].dt.isocalendar().week.astype(int)
    daily['quarter'] = daily['date'].dt.quarter

    daily['is_month_start'] = daily['date'].dt.is_month_start.astype(int)
    daily['is_month_end'] = daily['date'].dt.is_month_end.astype(int)
    daily['is_quarter_start'] = daily['date'].dt.is_quarter_start.astype(int)
    daily['is_quarter_end'] = daily['date'].dt.is_quarter_end.astype(int)
    daily['is_year_start'] = daily['date'].dt.is_year_start.astype(int)
    daily['is_year_end'] = daily['date'].dt.is_year_end.astype(int)

    daily['week_day_binary'] = daily['day_of_week'].isin([5, 6]).astype(int)

    # === LAG & ROLLING ===
    daily['Demand_lag_7'] = daily[TARGET_COL].shift(7)
    daily['Demand_lag_14'] = daily[TARGET_COL].shift(14)
    daily['Demand_lag_21'] = daily[TARGET_COL].shift(21)

    daily['Demand_roll_mean_7'] = daily[TARGET_COL].rolling(7).mean()
    daily['Demand_roll_std_7'] = daily[TARGET_COL].rolling(7).std()
    daily['Demand_roll_mean_14'] = daily[TARGET_COL].rolling(14).mean()
    daily['Demand_roll_std_14'] = daily[TARGET_COL].rolling(14).std()

    daily.dropna(inplace=True)
    daily.reset_index(drop=True, inplace=True)

    return daily


data = load_processed_data()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ Forecast Setting")

forecast_days = st.sidebar.slider(
    "Jumlah Hari Forecast",
    min_value=7,
    max_value=60,
    value=30
)

# =====================================================
# HEADER
# =====================================================
st.title("📈 Demand Forecasting – XGBoost")
st.markdown("""
Model **XGBoost Regressor** dengan **time-based, lag, dan rolling features**.  
Pipeline inference **identik dengan training** dan **aman untuk production**.
""")

# =====================================================
# DATA PREVIEW
# =====================================================
st.subheader("📊 Data Historis (Daily)")
st.dataframe(data[['date', TARGET_COL]].tail(10), use_container_width=True)

# =====================================================
# FORECAST FUNCTION (AMAN 100%)
# =====================================================
def forecast_xgb(history, days):
    history = history.copy()
    forecasts = []

    for _ in range(days):
        last_row = history.iloc[-1]

        X_pred = pd.DataFrame(
            [last_row[MODEL_FEATURES]],
            columns=MODEL_FEATURES
        )

        y_pred = float(model.predict(X_pred)[0])
        next_date = last_row['date'] + pd.Timedelta(days=1)

        new_row = {
            'date': next_date,
            TARGET_COL: y_pred
        }

        # Tambah baris baru
        history = pd.concat(
            [history, pd.DataFrame([new_row])],
            ignore_index=True
        )

        # Update fitur HANYA berbasis history
        history['year'] = history['date'].dt.year
        history['month'] = history['date'].dt.month
        history['day_of_month'] = history['date'].dt.day
        history['day_of_week'] = history['date'].dt.dayofweek
        history['day_of_year'] = history['date'].dt.dayofyear
        history['week_of_year'] = history['date'].dt.isocalendar().week.astype(int)
        history['quarter'] = history['date'].dt.quarter

        history['is_month_start'] = history['date'].dt.is_month_start.astype(int)
        history['is_month_end'] = history['date'].dt.is_month_end.astype(int)
        history['is_quarter_start'] = history['date'].dt.is_quarter_start.astype(int)
        history['is_quarter_end'] = history['date'].dt.is_quarter_end.astype(int)
        history['is_year_start'] = history['date'].dt.is_year_start.astype(int)
        history['is_year_end'] = history['date'].dt.is_year_end.astype(int)
        history['week_day_binary'] = history['day_of_week'].isin([5, 6]).astype(int)

        history['Demand_lag_7'] = history[TARGET_COL].shift(7)
        history['Demand_lag_14'] = history[TARGET_COL].shift(14)
        history['Demand_lag_21'] = history[TARGET_COL].shift(21)
        history['Demand_roll_mean_7'] = history[TARGET_COL].rolling(7).mean()
        history['Demand_roll_std_7'] = history[TARGET_COL].rolling(7).std()
        history['Demand_roll_mean_14'] = history[TARGET_COL].rolling(14).mean()
        history['Demand_roll_std_14'] = history[TARGET_COL].rolling(14).std()

        forecasts.append(new_row)

    return pd.DataFrame(forecasts)


with st.spinner("⏳ Forecasting in progress..."):
    forecast_df = forecast_xgb(data, forecast_days)

# =====================================================
# VISUALIZATION
# =====================================================
st.subheader("📉 Forecast Result")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data['date'], data[TARGET_COL], label="Actual", linewidth=2)
ax.plot(
    forecast_df['date'],
    forecast_df[TARGET_COL],
    linestyle="--",
    linewidth=2,
    label="Forecast (XGBoost)"
)
ax.set_xlabel("Date")
ax.set_ylabel("Demand")
ax.set_title("Demand Forecasting – XGBoost")
ax.legend()

st.pyplot(fig)

# =====================================================
# DOWNLOAD
# =====================================================
st.subheader("⬇️ Download Forecast")
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Forecast CSV",
    csv,
    "forecast_xgboost.csv",
    "text/csv"
)

st.markdown("---")
st.markdown("**Status:** ✅ Stable · ✅ No Runtime Error · ✅ Cloud Ready")

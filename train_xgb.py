import pandas as pd
import joblib
from xgboost import XGBRegressor

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

print("📥 Loading data...")
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

# ===== TIME FEATURES =====
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

# ===== LAG & ROLLING =====
daily['Demand_lag_7'] = daily[TARGET_COL].shift(7)
daily['Demand_lag_14'] = daily[TARGET_COL].shift(14)
daily['Demand_lag_21'] = daily[TARGET_COL].shift(21)

daily['Demand_roll_mean_7'] = daily[TARGET_COL].rolling(7).mean()
daily['Demand_roll_std_7'] = daily[TARGET_COL].rolling(7).std()
daily['Demand_roll_mean_14'] = daily[TARGET_COL].rolling(14).mean()
daily['Demand_roll_std_14'] = daily[TARGET_COL].rolling(14).std()

daily.dropna(inplace=True)

X = daily[MODEL_FEATURES]
y = daily[TARGET_COL]

print("🤖 Training XGBoost...")
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model_xgb.pkl")
print("✅ model_xgb.pkl berhasil dibuat")

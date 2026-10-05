import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==============================
# 1. Load & Preprocess Data
# ==============================
file_path = "dataset/Power_househol.csv"
print("Loading dataset...")
df = pd.read_csv(file_path, sep=";", low_memory=False)

# Clean missing values and format data types
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

numeric_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
for col in numeric_cols:
    df[col] = df[col].astype(float)

# Feature Engineering
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
)
df["Hour"] = df["Datetime"].dt.hour
df["Day"] = df["Datetime"].dt.day
df["Month"] = df["Datetime"].dt.month
df["DayOfWeek"] = df["Datetime"].dt.dayofweek

# Features & Target Selection
features = [
    "Hour",
    "Day",
    "Month",
    "DayOfWeek",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
X = df[features]
y = df["Global_active_power"]

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 2. Train Random Forest Model
# ==============================
print("\nTraining Random Forest Regressor (this may take 1-2 minutes)...")
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

# ==============================
# 3. Model Evaluation
# ==============================
print("\n========== MODEL EVALUATION ==========")
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE):    {mae:.4f} kW")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} kW")
print(f"R-squared (R² Score):          {r2:.4f}")

# ==============================
# 4. Save Model Artifact
# ==============================
os.makedirs("Models", exist_ok=True)
model_save_path = "Models/rf_power_model.pkl"
joblib.dump(rf_model, model_save_path)
print(f"\n✅ Model successfully saved to: {model_save_path}")
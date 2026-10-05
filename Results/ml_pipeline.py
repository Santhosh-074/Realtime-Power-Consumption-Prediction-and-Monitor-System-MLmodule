import joblib
import numpy as np
import pandas as pd


class PowerPredictionModule:

    def __init__(
        self, model_path="Models/rf_power_model.pkl", tariff_per_kwh=0.15
    ):
        """Loads the pre-trained model and sets electricity tariff rate ($/kWh or ₹/kWh)."""
        self.model = joblib.load(model_path)
        self.tariff = tariff_per_kwh

    def preprocess_dataframe(self, df):
        """Cleans incoming CSV data and extracts required datetime and numeric features."""
        data = df.copy()

        # Handle '?' missing values
        data.replace("?", np.nan, inplace=True)
        data.dropna(inplace=True)

        # Convert numeric columns
        numeric_cols = [
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]
        for col in numeric_cols:
            if col in data.columns:
                data[col] = data[col].astype(float)

        # Feature engineering from Date & Time strings
        if "Date" in data.columns and "Time" in data.columns:
            data["Datetime"] = pd.to_datetime(
                data["Date"] + " " + data["Time"], format="%d/%m/%Y %H:%M:%S"
            )
            data["Hour"] = data["Datetime"].dt.hour
            data["Day"] = data["Datetime"].dt.day
            data["Month"] = data["Datetime"].dt.month
            data["DayOfWeek"] = data["Datetime"].dt.dayofweek

        # Select exact feature sequence expected by Random Forest model
        feature_cols = [
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

        return data, data[feature_cols]

    def predict_batch_csv(
        self, df, high_threshold=4.0, moderate_threshold=2.5
    ):
        """Processes uploaded CSV, predicts power usage, calculates alerts and cost projections."""
        cleaned_df, X_features = self.preprocess_dataframe(df)

        # Generate active power predictions (kW)
        cleaned_df["Predicted_Active_Power_kW"] = self.model.predict(
            X_features
        )

        # Assign alert levels based on predicted power draw
        def assign_alert(power):
            if power > high_threshold:
                return "HIGH LOAD ALERT"
            elif power > moderate_threshold:
                return "MODERATE LOAD"
            return "NORMAL LOAD"

        cleaned_df["Alert_Status"] = cleaned_df[
            "Predicted_Active_Power_kW"
        ].apply(assign_alert)

        # Calculate estimated consumption (kWh) & cost estimates
        # Average active power (kW) over 24 hours gives daily kWh
        avg_kw = cleaned_df["Predicted_Active_Power_kW"].mean()

        est_daily_kwh = avg_kw * 24
        est_monthly_kwh = est_daily_kwh * 30

        cost_summary = {
            "avg_predicted_power_kw": round(float(avg_kw), 3),
            "est_daily_kwh": round(float(est_daily_kwh), 2),
            "est_monthly_kwh": round(float(est_monthly_kwh), 2),
            "est_daily_cost": round(float(est_daily_kwh * self.tariff), 2),
            "est_monthly_cost": round(float(est_monthly_kwh * self.tariff), 2),
            "total_alerts_triggered": int(
                (
                    cleaned_df["Alert_Status"] == "HIGH LOAD ALERT"
                ).sum()
            ),
        }

        return cleaned_df, cost_summary
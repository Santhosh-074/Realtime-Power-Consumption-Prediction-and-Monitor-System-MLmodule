import pandas as pd
from ml_pipeline import PowerPredictionModule

# 1. Initialize the module (uses Models/rf_power_model.pkl by default)
print("Loading ML Pipeline...")
predictor = PowerPredictionModule(
    model_path="Models/rf_power_model.pkl", tariff_per_kwh=0.15
)

# 2. Load sample data from your dataset to simulate an uploaded CSV
print("Loading sample data...")
sample_df = pd.read_csv("dataset/Power_househol.csv", sep=";", nrows=100)

# 3. Run predictions and cost calculations
print("Running batch prediction and cost analysis...")
results_df, cost_summary = predictor.predict_batch_csv(sample_df)

# 4. Display Results
print("\n========== PIPELINE TEST RESULTS ==========")
print(f"Average Predicted Power: {cost_summary['avg_predicted_power_kw']} kW")
print(f"Estimated Daily Cost:    ${cost_summary['est_daily_cost']}")
print(f"Estimated Monthly Cost:  ${cost_summary['est_monthly_cost']}")
print(
    f"High Load Alerts Triggered: {cost_summary['total_alerts_triggered']}"
)
print("\nFirst 5 Processed Output Rows:")
print(
    results_df[
        ["Date", "Time", "Predicted_Active_Power_kW", "Alert_Status"]
    ].head()
)
print("\n✅ Pipeline executed with ZERO errors!")
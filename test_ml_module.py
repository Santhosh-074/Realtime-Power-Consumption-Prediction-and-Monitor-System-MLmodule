import pandas as pd

from Results.ml_pipeline import PowerPredictionModule


# Load dataset
df = pd.read_csv(
    "dataset/Power_househol.csv",
    sep=";"
)

print("Dataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# Create ML module
ml = PowerPredictionModule(
    model_path="Models/rf_power_model.pkl"
)


# Run prediction
result_df, summary = ml.predict_batch_csv(df)


print("\n========== ML PREDICTION RESULT ==========")

print("\nFirst 5 predictions:")
print(
    result_df[
        [
            "Predicted_Active_Power_kW",
            "Alert_Status"
        ]
    ].head()
)


print("\n========== SUMMARY ==========")

for key, value in summary.items():
    print(f"{key}: {value}")
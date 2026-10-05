import joblib

# Load the saved binary pickle file back into Python
model = joblib.load("Models/rf_power_model.pkl")

print("========== MODEL INSPECTION ==========")
print(f"Model Type:            {type(model).__name__}")
print(f"Number of Decision Trees: {model.n_estimators}")
print(f"Max Tree Depth:           {model.max_depth}")
print(f"Features Used:         {model.n_features_in_}")
print("\nFeature Importances:")

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
for name, importance in zip(features, model.feature_importances_):
    print(f" - {name:22s}: {importance:.4f}")
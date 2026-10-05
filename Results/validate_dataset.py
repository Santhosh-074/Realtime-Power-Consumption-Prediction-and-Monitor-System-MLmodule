import pandas as pd

file_path = "dataset/Power_househol.csv"

df = pd.read_csv(
    file_path,
    sep=";",
    na_values=["?"]
)

required_columns = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

print("========== DATASET VALIDATION ==========")

if not missing_columns:
    print("VALID DATASET")
    print("All required columns are present.")
else:
    print("INVALID DATASET")
    print("Missing columns:")

    for col in missing_columns:
        print("-", col)

print("\nDataset shape:", df.shape)
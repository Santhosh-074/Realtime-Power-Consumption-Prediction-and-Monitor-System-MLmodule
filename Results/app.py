import datetime
import joblib
import pandas as pd
import streamlit as st
from ml_pipeline import PowerPredictionModule

# Page Configuration
st.set_page_config(
    page_title="Power Consumption Predictor", page_icon="⚡", layout="wide"
)

st.title("⚡ Household Electric Power Consumption & Cost Predictor")
st.write(
    "Predict active electrical power ($kW$), receive live high-load alerts, and estimate electricity bill costs in Indian Rupees (₹)."
)


# Load Trained Model and Pipeline
@st.cache_resource
def load_pipeline():
    # Set tariff_per_kwh=7.00 for ₹7 per unit/kWh
    return PowerPredictionModule(
        model_path="Models/rf_power_model.pkl", tariff_per_kwh=7.00
    )


try:
    pipeline = load_pipeline()
    st.sidebar.success("✅ ML Model & Pipeline Loaded!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading pipeline: {e}")
    st.stop()

# Select Mode
app_mode = st.sidebar.radio(
    "Choose Mode", ["Single Parameter Input", "Bulk CSV Upload"]
)

# ---------------------------------------------------------
# MODE 1: SINGLE PARAMETER INPUT
# ---------------------------------------------------------
if app_mode == "Single Parameter Input":
    st.sidebar.header("📥 Input Sensor Parameters")

    input_date = st.sidebar.date_input("Select Date", datetime.date.today())
    input_time = st.sidebar.time_input("Select Time", datetime.time(18, 0))

    st.sidebar.subheader("Electrical Measurements")
    voltage = st.sidebar.number_input(
        "Voltage (V)", min_value=100.0, max_value=300.0, value=230.0, step=0.1
    )
    global_intensity = st.sidebar.number_input(
        "Global Intensity (A)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1,
    )
    reactive_power = st.sidebar.number_input(
        "Global Reactive Power (kVAR)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.01,
    )

    st.sidebar.subheader("Sub-metering (Wh)")
    sub1 = st.sidebar.number_input(
        "Sub-metering 1 (Kitchen)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )
    sub2 = st.sidebar.number_input(
        "Sub-metering 2 (Laundry)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
    )
    sub3 = st.sidebar.number_input(
        "Sub-metering 3 (AC / Heater)",
        min_value=0.0,
        max_value=100.0,
        value=18.0,
        step=1.0,
    )

    # Convert inputs to DataFrame
    input_data = pd.DataFrame(
        [
            {
                "Hour": input_time.hour,
                "Day": input_date.day,
                "Month": input_date.month,
                "DayOfWeek": input_date.weekday(),
                "Global_reactive_power": reactive_power,
                "Voltage": voltage,
                "Global_intensity": global_intensity,
                "Sub_metering_1": sub1,
                "Sub_metering_2": sub2,
                "Sub_metering_3": sub3,
            }
        ]
    )

    st.subheader("📊 Single Prediction & Cost Estimation")

    if st.button("🚀 Predict Active Power", type="primary"):
        predicted_power = pipeline.model.predict(input_data)[0]

        # Cost Logic (Assuming sustained usage at this rate)
        tariff = 7.00  # ₹7.00 per kWh
        est_daily_cost = predicted_power * 24 * tariff
        est_monthly_cost = est_daily_cost * 30

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Power", f"{predicted_power:.3f} kW")

        with col2:
            st.metric("Est. Daily Cost", f"₹{est_daily_cost:.2f}")

        with col3:
            st.metric("Est. Monthly Cost", f"₹{est_monthly_cost:.2f}")

        st.markdown("---")

        # Alert Logic
        if predicted_power > 4.0:
            st.error("⚠️ HIGH LOAD ALERT: Power consumption exceeds 4.0 kW!")
        elif predicted_power > 2.5:
            st.warning("⚡ MODERATE LOAD: Fair power usage observed.")
        else:
            st.success("✅ NORMAL LOAD: Consumption within standard range.")

        st.subheader("📋 Submitted Input Parameters")
        st.dataframe(input_data)

# ---------------------------------------------------------
# MODE 2: BULK CSV UPLOAD
# ---------------------------------------------------------
else:
    st.subheader("📁 Upload Household Sensor CSV")
    st.write("Upload a semicolon (`;`) separated CSV file from the dataset.")

    uploaded_file = st.file_uploader("Choose a CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV
            raw_df = pd.read_csv(uploaded_file, sep=";")

            # Run ML Pipeline
            results_df, cost_summary = pipeline.predict_batch_csv(raw_df)

            st.subheader("💰 Electricity Cost & Load Summary (INR)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Predicted Power", f"{cost_summary['avg_predicted_power_kw']} kW")
            col2.metric("Est. Daily Cost", f"₹{cost_summary['est_daily_cost']}")
            col3.metric("Est. Monthly Cost", f"₹{cost_summary['est_monthly_cost']}")
            col4.metric("High Load Spikes", f"{cost_summary['total_alerts_triggered']}")

            st.markdown("---")
            st.subheader("📊 Detailed Prediction Results")
            st.dataframe(
                results_df[
                    [
                        "Date",
                        "Time",
                        "Predicted_Active_Power_kW",
                        "Alert_Status",
                    ]
                ]
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
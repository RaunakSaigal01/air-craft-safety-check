import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

# --------- Generate Simulated Dataset for Model Training ---------
np.random.seed(42)
n_rows = 2000

# Fuel and Weight Sensors
fuel_level_kg = np.random.normal(3000, 800, n_rows).clip(100, 5000)
fuel_flow_lpm = np.random.normal(12, 5, n_rows).clip(0, 30)
fuel_pressure_psi = np.random.normal(40, 7, n_rows).clip(20, 60)
fuel_leak = np.random.choice([0, 1], size=n_rows, p=[0.9, 0.1])
passenger_weight_kg = np.random.normal(6300, 500, n_rows).clip(4000, 7500)
cargo_weight_kg = np.random.normal(1500, 400, n_rows).clip(500, 3500)
fuel_weight_kg = fuel_level_kg * 0.8
total_weight_kg = passenger_weight_kg + cargo_weight_kg + fuel_weight_kg

# Engine Sensors
engine_temp_c = np.random.normal(700, 80, n_rows).clip(500, 950)
oil_pressure_psi = np.random.normal(60, 10, n_rows).clip(30, 100)
engine_rpm = np.random.normal(100, 10, n_rows).clip(85, 110)  # Simulating as % of max

# Safety Status Logic
status = np.where(
    (fuel_leak == 1) |
    (fuel_level_kg < 1000) |
    (total_weight_kg > 10000) |
    (engine_temp_c > 800) |
    (oil_pressure_psi < 40) |
    (oil_pressure_psi > 100) |
    (engine_rpm < 85) |
    (engine_rpm > 105),
    1, 0
)

# Create training DataFrame
df = pd.DataFrame({
    "fuel_level_kg": fuel_level_kg,
    "fuel_flow_lpm": fuel_flow_lpm,
    "fuel_pressure_psi": fuel_pressure_psi,
    "fuel_leak": fuel_leak,
    "passenger_weight_kg": passenger_weight_kg,
    "cargo_weight_kg": cargo_weight_kg,
    "fuel_weight_kg": fuel_weight_kg,
    "total_weight_kg": total_weight_kg,
    "engine_temp_c": engine_temp_c,
    "oil_pressure_psi": oil_pressure_psi,
    "engine_rpm": engine_rpm,
    "status": status
})

# Model Training
features = df.drop(columns=["status"])
labels = df["status"]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# --------- Streamlit Dashboard Starts ---------
st.title("✈️ ATR 72-600: Combined Aircraft Safety AI System")

st.sidebar.header("Enter Sensor Data (Fuel + Engine)")

# Fuel Inputs
fuel_level = st.sidebar.slider("Fuel Level (kg)", 100, 5000, 3000)
fuel_flow = st.sidebar.slider("Fuel Flow Rate (L/min)", 0, 30, 12)
fuel_pressure = st.sidebar.slider("Fuel Pressure (psi)", 20, 60, 40)
fuel_leak = st.sidebar.radio("Fuel Leak Detected?", ["No", "Yes"])
passenger_weight = st.sidebar.slider("Passenger Weight (kg)", 4000, 7500, 6300)
cargo_weight = st.sidebar.slider("Cargo Weight (kg)", 500, 3500, 1500)

# Engine Inputs
engine_temp = st.sidebar.slider("Engine Temp (°C)", 500, 950, 700)
oil_pressure = st.sidebar.slider("Oil Pressure (psi)", 30, 100, 60)
engine_rpm = st.sidebar.slider("Engine RPM (% max)", 85, 110, 100)

# Calculations
fuel_weight = fuel_level * 0.8
total_weight = passenger_weight + cargo_weight + fuel_weight

st.subheader("📋 Computed Values")
st.write(f"Fuel Weight: {fuel_weight:.2f} kg")
st.write(f"Total Aircraft Weight: {total_weight:.2f} kg")

# Prepare Input for Prediction
input_data = pd.DataFrame([{
    "fuel_level_kg": fuel_level,
    "fuel_flow_lpm": fuel_flow,
    "fuel_pressure_psi": fuel_pressure,
    "fuel_leak": 1 if fuel_leak == "Yes" else 0,
    "passenger_weight_kg": passenger_weight,
    "cargo_weight_kg": cargo_weight,
    "fuel_weight_kg": fuel_weight,
    "total_weight_kg": total_weight,
    "engine_temp_c": engine_temp,
    "oil_pressure_psi": oil_pressure,
    "engine_rpm": engine_rpm
}])

# Prediction
prediction = model.predict(input_data)[0]
pred_label = "UNSAFE" if prediction == 1 else "OK"

# Alert Display
if prediction == 1:
    st.error(f"🚨 ALERT: Aircraft Status - UNSAFE for Takeoff!")
else:
    st.success(f"✅ Aircraft Status: OK for Takeoff")

# Logging
log_entry = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "fuel_level_kg": fuel_level,
    "fuel_flow_lpm": fuel_flow,
    "fuel_pressure_psi": fuel_pressure,
    "fuel_leak": 1 if fuel_leak == "Yes" else 0,
    "passenger_weight_kg": passenger_weight,
    "cargo_weight_kg": cargo_weight,
    "fuel_weight_kg": fuel_weight,
    "total_weight_kg": total_weight,
    "engine_temp_c": engine_temp,
    "oil_pressure_psi": oil_pressure,
    "engine_rpm": engine_rpm,
    "AI_Prediction": pred_label
}

log_file = "combined_safety_log.csv"
try:
    log_df = pd.read_csv(log_file)
    log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
except FileNotFoundError:
    log_df = pd.DataFrame([log_entry])

log_df.to_csv(log_file, index=False)

# Show recent logs
st.subheader("📑 Recent Safety Check Logs")
st.dataframe(log_df.tail(10))

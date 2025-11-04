import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# --- App Title ---
st.title("🌦️ Smart Rainfall Forecasting & Crop Recommendation System")
st.markdown("Analyze rainfall trends, forecast future rainfall, and get crop recommendations based on climate patterns.")

# --- Load Data ---
df = pd.read_csv("rainfall_data.csv")
df["date"] = pd.to_datetime(df["date"])
st.success("✅ Data loaded successfully!")

# --- Display Charts ---
st.subheader("📊 Rainfall Trend (2000–2020)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["date"], df["rainfall"], color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Rainfall (mm)")
ax.set_title("Daily Rainfall Over Time")
st.pyplot(fig)

# --- Forecast Section ---
st.subheader("🔮 Predict Future Rainfall")
if st.button("Generate Forecast"):
    df_prophet = df[["date", "rainfall"]].rename(columns={"date": "ds", "rainfall": "y"})
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    model.plot(forecast)
    st.pyplot(plt)
    forecast.to_csv("forecast.csv", index=False)
    st.success("✅ Forecast saved as forecast.csv")

# --- Crop Recommendation Logic ---
st.subheader("🌾 Crop Recommendation")

avg_rain = df["rainfall"].mean()
avg_temp = df["temperature"].mean()

if avg_rain > 200 and avg_temp > 25:
    crop = "Rice 🌾"
elif avg_rain > 100:
    crop = "Maize 🌽"
else:
    crop = "Pulses 🌱"

st.info(f"**Suggested Crop:** {crop}")
st.caption(f"Average Rainfall: {avg_rain:.2f} mm | Average Temperature: {avg_temp:.2f}°C")

# --- Alert System ---
st.subheader("⚠️ Weather Alert System")
if avg_rain < 50:
    st.error("🚨 Drought Risk Detected! Consider drought-resistant crops.")
elif avg_rain > 300:
    st.warning("⚠️ High Rainfall — Flooding Risk! Plan drainage accordingly.")
else:
    st.success("✅ Rainfall is normal and suitable for most crops.")

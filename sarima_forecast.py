import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("rainfall_monthly.csv", parse_dates=["date"], index_col="date")
train = df.iloc[:-12]
test = df.iloc[-12:]

# Fit SARIMA model
model = sm.tsa.statespace.SARIMAX(train["precip"],
                                  order=(1,1,1),
                                  seasonal_order=(1,1,1,12),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
result = model.fit(disp=False)

# Forecast
forecast = result.get_forecast(steps=12)
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

print("✅ Forecasted rainfall for next 12 months:")
print(pred_mean)

# Plot
plt.figure(figsize=(10,5))
plt.plot(train.index, train["precip"], label="Train")
plt.plot(test.index, test["precip"], label="Test", color="orange")
plt.plot(pred_mean.index, pred_mean, label="Forecast", color="green")
plt.fill_between(conf_int.index,
                 conf_int.iloc[:,0],
                 conf_int.iloc[:,1], color="lightgreen", alpha=0.3)
plt.legend()
plt.show()
    
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("rainfall_monthly.csv", parse_dates=["date"], index_col="date")

# Plot rainfall
df["precip"].plot(figsize=(10,5), title="Monthly Rainfall (mm)")
plt.show()

# Seasonal decomposition
res = sm.tsa.seasonal_decompose(df["precip"], model="additive", period=12)
res.plot()
plt.show()
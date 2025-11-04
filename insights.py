import pandas as pd

df = pd.read_csv("rainfall_monthly.csv", parse_dates=["date"], index_col="date")
avg = df["precip"].mean()

# Example: using SARIMA forecast results
future_sum = 500  # replace with sum of forecasted values

if future_sum < 0.8*avg*12:
    print("⚠️ Low rainfall expected → grow drought-resistant crops")
elif future_sum > 1.2*avg*12:
    print("🌧 High rainfall expected → prepare for floods/waterlogging")
else:
    print("✅ Normal rainfall → standard cropping plan possible")
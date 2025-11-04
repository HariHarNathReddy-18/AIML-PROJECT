"""Smoke test: load rainfall_monthly.csv and run a direct SARIMAX fit (order=(2,1,3),(1,1,1,12)).
This avoids importing app.py UI code and simply validates that the SARIMA order fits on your data.
"""
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import numpy as np

ROOT = Path(r"e:\Projects\3rd year cbp\aiml project")
CSV = ROOT / 'rainfall_monthly.csv'
if not CSV.exists():
    print('Missing rainfall_monthly.csv')
    raise SystemExit(1)

# load CSV
monthly = pd.read_csv(CSV, parse_dates=['date'], index_col='date')
col = 'rainfall' if 'rainfall' in monthly.columns else monthly.columns[0]
train = monthly[col].iloc[:-12].clip(lower=0)

try:
    mod = sm.tsa.statespace.SARIMAX(train, order=(2,1,3), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False, maxiter=200)
    print('Fit successful. AIC=', res.aic)
    # quick forecast sanity check
    fc = res.get_forecast(steps=12)
    pred = fc.predicted_mean.clip(lower=0)
    print('Forecast head:', np.round(pred.values[:5],2))
except Exception as e:
    print('SARIMA fit failed:', type(e).__name__, e)

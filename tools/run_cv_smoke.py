import pandas as pd
import numpy as np
import math
import os
from prophet import Prophet
import statsmodels.api as sm

fn = 'rainfall_monthly.csv'
fnp = os.path.join(os.path.dirname(__file__), '..', fn)
if not os.path.exists(fnp):
    print('rainfall_monthly.csv not found at', fnp)
else:
    df = pd.read_csv(fnp, parse_dates=['date'], index_col='date')
    if 'rainfall' not in df.columns:
        print("File doesn't contain 'rainfall' column. Columns:", df.columns.tolist())
    else:
        s = df['rainfall'].clip(lower=0)
        def cv(series, horizon=3, folds=3):
            N = len(series)
            results = {'SARIMA':[], 'Prophet':[]}
            for i in range(folds):
                train_end = N - horizon*(folds - i)
                train = series.iloc[:train_end]
                test = series.iloc[train_end: train_end + horizon]
                if len(test) < horizon:
                    break
                # SARIMA
                try:
                    sar = sm.tsa.statespace.SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    fc = sar.get_forecast(steps=horizon).predicted_mean.clip(lower=0)
                    rmse = math.sqrt(np.mean((test.values - fc.values)**2))
                    mae = np.mean(np.abs(test.values - fc.values))
                except Exception as e:
                    rmse = None; mae = None
                results['SARIMA'].append({'rmse': rmse, 'mae': mae})
                # Prophet
                try:
                    dfp = pd.DataFrame({'ds': train.index, 'y': train.values})
                    m = Prophet()
                    m.fit(dfp)
                    future = m.make_future_dataframe(periods=horizon, freq='M')
                    fcst = m.predict(future)
                    pred = fcst.set_index('ds')['yhat'].reindex(test.index).clip(lower=0)
                    rmse = math.sqrt(np.mean((test.values - pred.values)**2))
                    mae = np.mean(np.abs(test.values - pred.values))
                except Exception as e:
                    rmse = None; mae = None
                results['Prophet'].append({'rmse': rmse, 'mae': mae})
            # aggregate
            summary = {}
            for k, vals in results.items():
                valid = [v for v in vals if v['rmse'] is not None]
                if not valid:
                    summary[k] = {'rmse': None, 'mae': None, 'details': vals}
                else:
                    summary[k] = {'rmse': float(np.mean([v['rmse'] for v in valid])), 'mae': float(np.mean([v['mae'] for v in valid])), 'details': vals}
            return summary
        res = cv(s, horizon=3, folds=3)
        print('Cross-validation summary (h=3, folds=3):')
        print(res)

"""Run a short rolling-origin CV (local implementation) and save results to tools/last_cv_result.json
This script avoids importing app.py to prevent Streamlit side-effects during import.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

def rolling_origin_cv(series: pd.Series, horizon: int = 1, n_splits: int = 3, models: list = ['SARIMA','Prophet','AutoARIMA']):
    results = {m: [] for m in models}
    N = len(series)
    total_required = horizon * n_splits
    if N <= total_required + 12:
        return {m: {'rmse': None, 'mae': None, 'details': []} for m in models}
    for i in range(n_splits):
        train_end = N - horizon * (n_splits - i)
        train = series.iloc[:train_end]
        test = series.iloc[train_end: train_end + horizon]
        if len(test) < horizon:
            break
        # SARIMA
        try:
            import statsmodels.api as sm
            res = sm.tsa.statespace.SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc = res.get_forecast(steps=horizon).predicted_mean.clip(lower=0)
            rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
            mae = float(np.mean(np.abs(test.values - fc.values)))
        except Exception:
            rmse = None; mae = None
        results['SARIMA'].append({'rmse': rmse, 'mae': mae})
        # Prophet
        try:
            from prophet import Prophet
            df_prop = pd.DataFrame({'ds': train.index, 'y': train.values})
            m = Prophet()
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=horizon, freq='M')
            fcst = m.predict(future)
            pred = fcst.set_index('ds')['yhat'].reindex(test.index).clip(lower=0)
            rmse = float(np.sqrt(np.mean((test.values - pred.values) ** 2)))
            mae = float(np.mean(np.abs(test.values - pred.values)))
        except Exception:
            rmse = None; mae = None
        results['Prophet'].append({'rmse': rmse, 'mae': mae})
        # AutoARIMA
        if PMDARIMA_AVAILABLE:
            try:
                am = pm.auto_arima(train, seasonal=True, m=12, error_action='ignore', suppress_warnings=True, stepwise=True)
                fc = am.predict(n_periods=horizon)
                fc = np.array(fc).clip(min=0)
                if len(fc) == len(test):
                    rmse = float(np.sqrt(np.mean((test.values - fc) ** 2)))
                    mae = float(np.mean(np.abs(test.values - fc)))
                else:
                    rmse = None; mae = None
            except Exception:
                rmse = None; mae = None
        else:
            rmse = None; mae = None
        results.setdefault('AutoARIMA', []).append({'rmse': rmse, 'mae': mae})
    # aggregate
    summary = {}
    for m, vals in results.items():
        valid = [v for v in vals if v.get('rmse') is not None]
        if not valid:
            summary[m] = {'rmse': None, 'mae': None, 'details': vals}
        else:
            summary[m] = {'rmse': float(np.mean([v['rmse'] for v in valid])), 'mae': float(np.mean([v['mae'] for v in valid])), 'details': vals}
    return summary


def main():
    p = Path(r"e:\Projects\3rd year cbp\aiml project\rainfall_monthly.csv")
    if not p.exists():
        print('monthly file missing:', p)
        raise SystemExit(1)
    df = pd.read_csv(p, parse_dates=['date'], index_col='date')
    col = 'rainfall' if 'rainfall' in df.columns else df.columns[0]
    series = df[col].clip(lower=0)
    print('Data length:', len(series))
    models = ['SARIMA','Prophet']
    if PMDARIMA_AVAILABLE:
        models.append('AutoARIMA')
    res = rolling_origin_cv(series, horizon=3, n_splits=2, models=models)
    out = Path(r"e:\Projects\3rd year cbp\aiml project\tools\last_cv_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    print('Wrote', out)


if __name__ == '__main__':
    main()

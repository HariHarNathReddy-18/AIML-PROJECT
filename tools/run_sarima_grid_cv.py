"""Run a wider SARIMA grid-search using rolling-origin CV and save results.
Saves output to tools/sarima_grid_cv_results.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import time

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

from datetime import datetime

OUT = Path(r"e:\Projects\3rd year cbp\aiml project\tools\sarima_grid_cv_results.json")
CSV = Path(r"e:\Projects\3rd year cbp\aiml project\rainfall_monthly.csv")

if not CSV.exists():
    print('rainfall_monthly.csv not found at', CSV)
    raise SystemExit(1)

print('Loading data from', CSV)
df = pd.read_csv(CSV, parse_dates=['date'], index_col='date')
col = 'rainfall' if 'rainfall' in df.columns else df.columns[0]
series = df[col].clip(lower=0).dropna()
print('Series length:', len(series))

# Grid ranges (wider)
p_range = list(range(0, 4))
q_range = list(range(0, 4))
P_range = list(range(0, 3))
Q_range = list(range(0, 2))
D = 1
d = 1
s = 12

cv_horizon = 3
cv_folds = 3

# helper to run rolling-origin CV for a given order
import statsmodels.api as sm

def eval_sarima_cfg(train_series, order, seasonal_order, horizon=cv_horizon, n_splits=cv_folds):
    vals = []
    N = len(train_series)
    total_required = horizon * n_splits
    if N <= total_required + 12:
        return None
    for i in range(n_splits):
        train_end = N - horizon * (n_splits - i)
        train = train_series.iloc[:train_end]
        test = train_series.iloc[train_end: train_end + horizon]
        if len(test) < horizon:
            break
        try:
            mod = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            fc = res.get_forecast(steps=horizon).predicted_mean.clip(lower=0)
            rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
            vals.append(rmse)
        except Exception as e:
            # any failure -> mark as invalid
            # print('cfg failure', order, seasonal_order, e)
            return None
    if not vals:
        return None
    return float(np.mean(vals))

print('Starting SARIMA grid-search CV: p', p_range, 'q', q_range, 'P', P_range, 'Q', Q_range)
start = time.time()
best_rmse = np.inf
best_cfg = None
results = []
count = 0
total = len(p_range) * len(q_range) * len(P_range) * len(Q_range)
for p in p_range:
    for q in q_range:
        for P in P_range:
            for Q in Q_range:
                count += 1
                order = (p, d, q)
                seasonal_order = (P, D, Q, s)
                print(f'Evaluating {count}/{total} order={order} seasonal={seasonal_order} ...', end=' ')
                try:
                    rmse = eval_sarima_cfg(series, order, seasonal_order)
                except Exception as e:
                    rmse = None
                print('rmse=', rmse)
                results.append({'order': order, 'seasonal_order': seasonal_order, 'rmse': rmse})
                if rmse is not None and rmse < best_rmse:
                    best_rmse = rmse
                    best_cfg = (order, seasonal_order)
end = time.time()
print('Grid-search completed in %.1f sec' % (end - start))

# sort top 10
valid = [r for r in results if r['rmse'] is not None]
valid_sorted = sorted(valid, key=lambda x: x['rmse'])

top_k = valid_sorted[:10]

# Evaluate models (SARIMA with best, Prophet, AutoARIMA) on rolling CV
from math import sqrt

def evaluate_models(series, best_order_seasonal, horizon=cv_horizon, folds=cv_folds):
    models = {}
    N = len(series)
    total_required = horizon * folds
    if N <= total_required + 12:
        return {}
    models['SARIMA'] = []
    models['Prophet'] = []
    models['AutoARIMA'] = []
    for i in range(folds):
        train_end = N - horizon * (folds - i)
        train = series.iloc[:train_end]
        test = series.iloc[train_end: train_end + horizon]
        # SARIMA
        try:
            mod = sm.tsa.statespace.SARIMAX(train, order=best_order_seasonal[0], seasonal_order=best_order_seasonal[1], enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            fc = res.get_forecast(steps=horizon).predicted_mean.clip(lower=0)
            rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
            mae = float(np.mean(np.abs(test.values - fc.values)))
        except Exception:
            rmse = None; mae = None
        models['SARIMA'].append({'rmse': rmse, 'mae': mae})
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
        models['Prophet'].append({'rmse': rmse, 'mae': mae})
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
        models['AutoARIMA'].append({'rmse': rmse, 'mae': mae})
    # aggregate
    summary = {}
    for m, vals in models.items():
        valid_v = [v for v in vals if v.get('rmse') is not None]
        if not valid_v:
            summary[m] = {'rmse': None, 'mae': None, 'details': vals}
        else:
            summary[m] = {'rmse': float(np.mean([v['rmse'] for v in valid_v])), 'mae': float(np.mean([v['mae'] for v in valid_v])), 'details': vals}
    return summary

print('Best cfg:', best_cfg, 'best_rmse', best_rmse)
eval_summary = None
if best_cfg is not None:
    eval_summary = evaluate_models(series, best_cfg)

out = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'grid_search': {
        'attempted': len(results),
        'valid': len(valid),
        'best_rmse': best_rmse if best_rmse < np.inf else None,
        'best_cfg': {'order': best_cfg[0] if best_cfg else None, 'seasonal_order': best_cfg[1] if best_cfg else None},
        'top_k': top_k
    },
    'evaluation': eval_summary,
    'pmdarima_available': PMDARIMA_AVAILABLE
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=2)

print('Wrote results to', OUT)

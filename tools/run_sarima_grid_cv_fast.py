"""Two-stage SARIMA grid-search: AIC filter then CV on top-K.
Saves results to tools/sarima_grid_cv_fast_results.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

CSV = Path(r"e:\Projects\3rd year cbp\aiml project\rainfall_monthly.csv")
OUT = Path(r"e:\Projects\3rd year cbp\aiml project\tools\sarima_grid_cv_fast_results.json")

if not CSV.exists():
    print('rainfall_monthly.csv not found at', CSV)
    raise SystemExit(1)

df = pd.read_csv(CSV, parse_dates=['date'], index_col='date')
col = 'rainfall' if 'rainfall' in df.columns else df.columns[0]
series = df[col].clip(lower=0).dropna()
print('Series length:', len(series))

# Wider ranges but two-stage filter
p_range = list(range(0, 5))
q_range = list(range(0, 5))
P_range = list(range(0, 3))
Q_range = list(range(0, 3))
d = 1
D = 1
s = 12

import statsmodels.api as sm

candidates = []
print('Stage 1: computing AIC for candidates')
count = 0
for p in p_range:
    for q in q_range:
        for P in P_range:
            for Q in Q_range:
                count += 1
                order = (p,d,q)
                seasonal_order = (P,D,Q,s)
                try:
                    mod = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                                     enforce_stationarity=False, enforce_invertibility=False)
                    # small maxiter to speed up; if fails, skip
                    res = mod.fit(disp=False, maxiter=50)
                    aic = float(res.aic)
                    candidates.append({'order': order, 'seasonal_order': seasonal_order, 'aic': aic})
                except Exception as e:
                    # skip failing configs
                    continue
print('Found', len(candidates), 'valid candidates')

# pick top-K by AIC
K = 8
candidates_sorted = sorted(candidates, key=lambda x: x['aic'])
topk = candidates_sorted[:K]
print('Top-K candidates by AIC:')
for c in topk:
    print(c)

# Stage 2: CV on top-K
cv_horizon = 3
cv_folds = 3

def eval_cfg_cv(series, order, seasonal_order, horizon=cv_horizon, n_splits=cv_folds):
    vals = []
    N = len(series)
    total_required = horizon * n_splits
    if N <= total_required + 12:
        return None
    for i in range(n_splits):
        train_end = N - horizon * (n_splits - i)
        train = series.iloc[:train_end]
        test = series.iloc[train_end: train_end + horizon]
        if len(test) < horizon:
            break
        try:
            mod = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                             enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=100)
            fc = res.get_forecast(steps=horizon).predicted_mean.clip(lower=0)
            rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
            vals.append(rmse)
        except Exception:
            return None
    if not vals:
        return None
    return float(np.mean(vals))

results = []
for c in topk:
    order = tuple(c['order'])
    seasonal_order = tuple(c['seasonal_order'])
    print('CV evaluating', order, seasonal_order)
    rmse = eval_cfg_cv(series, order, seasonal_order)
    results.append({'order': order, 'seasonal_order': seasonal_order, 'aic': c['aic'], 'cv_rmse': rmse})

# pick best by CV RMSE
valid = [r for r in results if r['cv_rmse'] is not None]
best = sorted(valid, key=lambda x: x['cv_rmse'])[0] if valid else None

out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'series_length': len(series),
    'stage1_candidates': len(candidates),
    'topk': results,
    'best_by_cv': best
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=2)
print('Wrote', OUT)

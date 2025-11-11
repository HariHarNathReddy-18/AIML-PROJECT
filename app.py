import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from prophet import Prophet
import numpy as np
from fetch_data import fetch_rainfall
from preprocess import preprocess
import datetime
import hashlib
import io
# optional dependency for auto_arima
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

CROP_DATABASE = {
    "Kharif": [
        {"name": "Rice", "min_rain": 150, "max_rain": 1000},
        {"name": "Maize", "min_rain": 50, "max_rain": 300},
        {"name": "Soybean", "min_rain": 60, "max_rain": 250},
        {"name": "Cotton", "min_rain": 50, "max_rain": 150},
        {"name": "Bajra", "min_rain": 30, "max_rain": 100},
        {"name": "Groundnut", "min_rain": 50, "max_rain": 200},
        {"name": "Pigeonpea", "min_rain": 60, "max_rain": 200},
        {"name": "Sorghum", "min_rain": 30, "max_rain": 100},
        {"name": "Urd", "min_rain": 40, "max_rain": 120},
        {"name": "Moong", "min_rain": 40, "max_rain": 120},
    ],
    "Rabi": [
        {"name": "Wheat", "min_rain": 20, "max_rain": 100},
        {"name": "Barley", "min_rain": 20, "max_rain": 80},
        {"name": "Gram", "min_rain": 20, "max_rain": 80},
        {"name": "Mustard", "min_rain": 20, "max_rain": 80},
        {"name": "Peas", "min_rain": 30, "max_rain": 100},
        {"name": "Lentil", "min_rain": 20, "max_rain": 80},
        {"name": "Oats", "min_rain": 20, "max_rain": 80},
    ],
    "Other": [
        {"name": "Sugarcane", "min_rain": 100, "max_rain": 2000},
        {"name": "Jute", "min_rain": 150, "max_rain": 2000},
        {"name": "Millet", "min_rain": 20, "max_rain": 80},
        {"name": "Sunflower", "min_rain": 30, "max_rain": 120},
        {"name": "Sesame", "min_rain": 30, "max_rain": 120},
    ]
}

# Try to load an expanded crop database from disk (overrides/extends built-in CROP_DATABASE)
try:
    import os
    crop_csv = os.path.join(os.path.dirname(__file__), 'crop_db.csv')
    if os.path.exists(crop_csv):
        df_crop = pd.read_csv(crop_csv)
        # normalize seasons and merge
        for _, row in df_crop.iterrows():
            season_key = row['season'] if pd.notnull(row['season']) else 'Other'
            entry = {'name': row['name'], 'min_rain': float(row['min_rain']), 'max_rain': float(row['max_rain'])}
            if season_key in CROP_DATABASE:
                CROP_DATABASE[season_key].append(entry)
            else:
                CROP_DATABASE[season_key] = [entry]
except Exception:
    pass

def recommend_crop(total_rainfall, season=None):
    if season is None:
        if total_rainfall < 600:
            return "Millet, Sorghum, Pulses", "Drought-resistant crops recommended due to low rainfall."
        elif total_rainfall > 1200:
            return "Rice, Sugarcane, Jute", "Water-loving crops recommended due to high rainfall."
        else:
            return "Maize, Cotton, Groundnut", "Moderate rainfall crops recommended."
    else:
        if season.lower() == "kharif":
            if total_rainfall < 600:
                return "Bajra, Moong, Urd", "Kharif: Drought-resistant crops recommended."
            elif total_rainfall > 1200:
                return "Rice, Maize, Soybean", "Kharif: Water-loving crops recommended."
            else:
                return "Maize, Cotton, Groundnut", "Kharif: Moderate rainfall crops recommended."
        elif season.lower() == "rabi":
            if total_rainfall < 600:
                return "Wheat, Barley, Gram", "Rabi: Drought-resistant crops recommended."
            elif total_rainfall > 1200:
                return "Mustard, Peas", "Rabi: Water-loving crops recommended."
            else:
                return "Wheat, Mustard, Lentil", "Rabi: Moderate rainfall crops recommended."
        else:
            return "Maize, Cotton, Groundnut", "Season not recognized. Moderate rainfall crops recommended."

def month_wise_crop_recommendation(pred_mean, season=None):
    month_crops = []
    crop_list = CROP_DATABASE.get(season, CROP_DATABASE["Other"])
    for date, rainfall in pred_mean.items():
        month = pd.to_datetime(date).strftime("%B")
        # Find all crops suitable for this month's rainfall
        suitable_crops = [crop["name"] for crop in crop_list if crop["min_rain"] <= rainfall <= crop["max_rain"]]
        if not suitable_crops:
            suitable_crops = ["No optimal crop"]
        month_crops.append({
            "Month": month,
            "Recommended Crops": ", ".join(suitable_crops),
            "Predicted Rainfall (mm)": round(rainfall, 2)
        })
    return pd.DataFrame(month_crops)

def fetch_rainfall_from_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=["date"])
    df.set_index("date", inplace=True)
    return df

st.set_page_config(page_title="Rainfall Prediction & Crop Recommendation", layout="wide")

# Dashboard header
st.title("Rainfall Prediction & Crop Recommendation 🌧️🌾")

# Sidebar inputs (location + date range + season)
st.sidebar.header("Data & Location")
lat = st.sidebar.number_input("Latitude", value=17.3850, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=78.4867, format="%.4f")
st.sidebar.caption("Tip: Find this on Google Maps (right-click on a location).")
city = st.sidebar.text_input("City (optional)", value="")
# Multi-location compare inputs
compare = st.sidebar.checkbox("Compare with another location")
if compare:
    lat2 = st.sidebar.number_input("Latitude (2)", value=28.7041, format="%.4f")
    lon2 = st.sidebar.number_input("Longitude (2)", value=77.1025, format="%.4f")
    city2 = st.sidebar.text_input("City 2 (optional)", value="")

start_date = st.sidebar.date_input("Start date", value=datetime.date(2000,1,1))
end_date = st.sidebar.date_input("End date", value=datetime.date.today())

# Cropping season
season = st.sidebar.selectbox("Cropping Season", ["Kharif", "Rabi", "None"])
st.sidebar.caption("This helps filter the crop database for relevant results.")
season = None if season == "None" else season


# Additional controls for forecasts & visuals
forecast_horizon = st.sidebar.slider("Forecast horizon (months)", min_value=3, max_value=60, value=12, step=1)
rolling_window = st.sidebar.slider("Rolling mean window (months)", min_value=3, max_value=24, value=12, step=1)
show_ci = st.sidebar.checkbox("Show forecast 95% CI", value=True)
enable_caching = st.sidebar.checkbox("Enable model caching (session)", value=True)
# Advanced UI toggle: hide complex controls by default for a simpler interface
advanced = st.sidebar.checkbox("Show advanced options", value=False)

def _series_hash(s: pd.Series) -> str:
    try:
        b = s.to_json().encode()
    except Exception:
        b = pd.Series(s).to_json().encode()
    return hashlib.md5(b).hexdigest()

def _fig_to_bytes(fig: go.Figure, fmt: str = 'png') -> tuple[bytes,str]:
    """Return (data, mime) for download. If PNG generation fails, fall back to HTML."""
    try:
        img = fig.to_image(format=fmt)
        return img, 'image/png'
    except Exception:
        html = fig.to_html(include_plotlyjs='cdn')
        return html.encode('utf-8'), 'text/html'

def _get_cached_model(key: str):
    cache = st.session_state.setdefault('model_cache', {})
    return cache.get(key)

def _set_cached_model(key: str, model_obj):
    cache = st.session_state.setdefault('model_cache', {})
    cache[key] = model_obj

def fit_sarima_cached(train_series: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    key = ('sarima', order, seasonal_order, _series_hash(train_series))
    if enable_caching:
        cached = _get_cached_model(str(key))
        if cached is not None:
            return cached
    model = sm.tsa.statespace.SARIMAX(train_series,
                                      order=order,
                                      seasonal_order=seasonal_order,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    res = model.fit(disp=False)
    if enable_caching:
        _set_cached_model(str(key), res)
    return res

def fit_prophet_cached(df_prop: pd.DataFrame):
    # df_prop must have 'ds' and 'y'
    key = ('prophet', hashlib.md5(df_prop.to_json().encode()).hexdigest())
    if enable_caching:
        cached = _get_cached_model(str(key))
        if cached is not None:
            return cached
    m = Prophet()
    m.fit(df_prop)
    if enable_caching:
        _set_cached_model(str(key), m)
    return m


def fit_auto_arima_cached(train_series: pd.Series, seasonal=True, m=12, max_p=3, max_q=3, max_P=1, max_Q=1):
    """Fit auto_arima (pmdarima) with caching. Falls back to None if pmdarima not available."""
    key = ('auto_arima', seasonal, m, max_p, max_q, max_P, max_Q, _series_hash(train_series))
    if enable_caching:
        cached = _get_cached_model(str(key))
        if cached is not None:
            return cached
    if not PMDARIMA_AVAILABLE:
        return None
    try:
        model = pm.auto_arima(train_series, seasonal=seasonal, m=m, max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        if enable_caching:
            _set_cached_model(str(key), model)
        return model
    except Exception:
        return None


def rolling_origin_cv(series: pd.Series, horizon: int = 1, n_splits: int = 3, models: list = ['SARIMA','Prophet','AutoARIMA']):
    """Simple rolling-origin CV. Returns average RMSE/MAE per model.
    For each split we expand the training window and forecast `horizon` steps ahead.
    """
    results = {m: [] for m in models}
    N = len(series)
    total_required = horizon * n_splits
    if N <= total_required + 12:
        # not enough data for requested CV
        return {m: {'rmse': None, 'mae': None, 'details': []} for m in models}

    for i in range(n_splits):
        # compute train end index for this fold
        train_end = N - horizon * (n_splits - i)
        train = series.iloc[:train_end]
        test = series.iloc[train_end: train_end + horizon]
        if len(test) < horizon:
            break

        # SARIMA: fit simple default model (1,1,1)(1,1,1,12)
        if 'SARIMA' in models:
            try:
                res = sm.tsa.statespace.SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                fc = res.get_forecast(steps=horizon).predicted_mean.clip(lower=0)
                rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
                mae = float(np.mean(np.abs(test.values - fc.values)))
            except Exception:
                rmse = None; mae = None
            results['SARIMA'].append({'rmse': rmse, 'mae': mae})

        # Prophet
        if 'Prophet' in models:
            try:
                df_prop = train.reset_index().rename(columns={'date':'ds', train.name if hasattr(train,'name') else train.index.name:'y'})
                # ensure columns
                df_prop = train.reset_index().rename(columns={train.index.name or 'index':'ds', train.name or 0:'y'})
                df_prop = train.reset_index().rename(columns={train.index.name or 'index':'ds'})
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

        # Auto ARIMA (pmdarima)
        if 'AutoARIMA' in models or 'AutoARIMA' in [m for m in models]:
            if PMDARIMA_AVAILABLE:
                try:
                    am = pm.auto_arima(train, seasonal=True, m=12, error_action='ignore', suppress_warnings=True, stepwise=True)
                    fc = am.predict(n_periods=horizon)
                    fc = np.array(fc).clip(min=0)
                    # align lengths
                    if len(fc) == len(test):
                        rmse = float(np.sqrt(np.mean((test.values - fc) ** 2)))
                        mae = float(np.mean(np.abs(test.values - fc)))
                    else:
                        rmse = None; mae = None
                except Exception:
                    rmse = None; mae = None
            else:
                # pmdarima not available
                rmse = None; mae = None
            # ensure key exists
            key = 'AutoARIMA' if 'AutoARIMA' in models else [m for m in models if m.lower().startswith('auto')]
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


def select_best_model_via_cv(series: pd.Series, models_to_eval: list, cv_horizon: int = 3, cv_folds: int = 3):
    """Run rolling_origin_cv and pick the model with lowest RMSE (if available).
    Returns (best_model_name, cv_summary)
    """
    cv_res = rolling_origin_cv(series, horizon=cv_horizon, n_splits=cv_folds, models=models_to_eval)
    # pick best by rmse where rmse is not None
    best = None
    best_rmse = np.inf
    for m, stats in cv_res.items():
        try:
            rmse = stats.get('rmse')
        except Exception:
            rmse = None
        if rmse is None:
            continue
        if rmse < best_rmse:
            best_rmse = rmse
            best = m
    # fallback: if none valid, prefer SARIMA, then Prophet, then AutoARIMA
    if best is None:
        for pref in ['SARIMA','Prophet','AutoARIMA']:
            if pref in models_to_eval:
                best = pref
                break
    return best, cv_res


def generate_pdf_report(text: str, filename: str = 'report.pdf', images: list[bytes] | None = None) -> bytes:
    """Generate a simple PDF containing the provided text and optional PNG images.
    Returns PDF bytes. Falls back to returning plain-text bytes if reportlab unavailable.
    """
    if not REPORTLAB_AVAILABLE:
        return text.encode('utf-8')
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    x_margin = 40
    y = height - 40
    # write text lines
    for line in text.split('\n'):
        c.setFont('Helvetica', 10)
        c.drawString(x_margin, y, line[:200])
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 40
    # add images (PNG bytes) after text
    if images:
        for img_bytes in images:
            try:
                if y < 200:
                    c.showPage()
                    y = height - 40
                img_buf = io.BytesIO(img_bytes)
                # place image with width up to page width minus margins
                img_w = width - 2 * x_margin
                # height will be scaled preserving aspect via reportlab ImageReader
                from reportlab.lib.utils import ImageReader
                img = ImageReader(img_buf)
                iw, ih = img.getSize()
                scale = min(img_w / iw, (y - 40) / ih) if ih > 0 else 1
                w = iw * scale
                h = ih * scale
                c.drawImage(img, x_margin, y - h, width=w, height=h)
                y -= (h + 20)
            except Exception:
                continue
    c.save()
    buf.seek(0)
    return buf.read()


def sarima_grid_search(train_series: pd.Series, p_range=(0,1,2), d=1, q_range=(0,1,2), P_range=(0,1), D=1, Q_range=(0,1), s=12):
    """Lightweight grid search over SARIMA orders. Returns best (order, seasonal_order) by AIC.
    This is intentionally small to be reasonably fast.
    """
    best_aic = np.inf
    best_cfg = None
    for p in p_range:
        for q in q_range:
            for P in P_range:
                for Q in Q_range:
                    try:
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, s)
                        mod = sm.tsa.statespace.SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                        res = mod.fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_cfg = (order, seasonal_order, res)
                    except Exception:
                        continue
    return best_cfg


def sarima_grid_search_cv(train_series: pd.Series, p_range=(0,1,2), d=1, q_range=(0,1,2), P_range=(0,1), D=1, Q_range=(0,1), s=12, cv_horizon=3, cv_folds=2):
    """Grid search SARIMA orders using rolling-origin CV (select by lowest avg RMSE).
    Returns (best_order, best_seasonal_order, best_rmse)
    """
    best_rmse = np.inf
    best_cfg = None
    # small helper to evaluate a single (order, seasonal_order) by CV
    def eval_cfg(order, seasonal_order):
        vals = []
        N = len(train_series)
        total_required = cv_horizon * cv_folds
        if N <= total_required + 12:
            return None
        for i in range(cv_folds):
            train_end = N - cv_horizon * (cv_folds - i)
            train = train_series.iloc[:train_end]
            test = train_series.iloc[train_end: train_end + cv_horizon]
            if len(test) < cv_horizon:
                break
            try:
                import statsmodels.api as sm
                mod = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                fc = res.get_forecast(steps=cv_horizon).predicted_mean.clip(lower=0)
                rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
                vals.append(rmse)
            except Exception:
                return None
        if not vals:
            return None
        return float(np.mean(vals))

    for p in p_range:
        for q in q_range:
            for P in P_range:
                for Q in Q_range:
                    try:
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, s)
                        avg_rmse = eval_cfg(order, seasonal_order)
                        if avg_rmse is None:
                            continue
                        if avg_rmse < best_rmse:
                            best_rmse = avg_rmse
                            best_cfg = (order, seasonal_order)
                    except Exception:
                        continue
    return (best_cfg[0], best_cfg[1], best_rmse) if best_cfg is not None else None


def evaluate_models_cv(series: pd.Series, horizon: int = 3, n_splits: int = 2, models: list = ['SARIMA','Prophet','AutoARIMA'], sarima_order=None, seasonal_order=(1,1,1,12)):
    """Evaluate specified models using rolling-origin CV. If sarima_order is provided, SARIMA will use it.
    Returns summary like rolling_origin_cv.
    """
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
        if 'SARIMA' in models:
            try:
                import statsmodels.api as sm
                if sarima_order is None:
                    order = (1,1,1)
                    seas = (1,1,1,12)
                else:
                    order = sarima_order
                    seas = seasonal_order
                res = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seas, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                fc = res.get_forecast(steps=horizon).predicted_mean.clip(lower=0)
                rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
                mae = float(np.mean(np.abs(test.values - fc.values)))
            except Exception:
                rmse = None; mae = None
            results['SARIMA'].append({'rmse': rmse, 'mae': mae})
        # Prophet
        if 'Prophet' in models:
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
        if 'AutoARIMA' in models:
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

# Main tabs
tab_home, tab_analysis, tab_forecast, tab_crop, tab_alerts = st.tabs(["Home", "Rainfall Analysis", "Forecast", "Crop Recommendation", "Alerts"])

# --- 2. REPLACE THIS ENTIRE SECTION (starts around line 559) ---

with tab_home:
    st.header("Welcome to the Rainfall & Crop Advisor! 🌧️🌾")
    st.markdown("Get an AI-powered rainfall forecast and crop recommendations in 3 simple steps.")
    
    st.subheader("Step 1: Set Your Location")
    st.markdown("Use the **sidebar on the left** to enter the **Latitude** and **Longitude** for your farm or area of interest.")
    
    st.subheader("Step 2: Run the Forecast")
    st.markdown("Go to the **'Forecast'** tab and click the **'Run Prediction'** button. This will train our AI models on historical data for your location.")
    
    st.subheader("Step 3: Get Your Recommendations")
    st.markdown("Once the forecast is complete, go to the **'Crop Recommendation'** tab to see which crops are best suited for the predicted rainfall, month by month.")
    
    st.info("You can also explore the **'Rainfall Analysis'** tab to see the historical rainfall patterns for your location.")
    
# --- END OF HOME TAB REPLACEMENT ---
with tab_analysis:
    st.header("Rainfall Analysis")
    st.write("Use this tab to visualize monthly and yearly rainfall summaries.")
    # Try to load monthly data if it exists
    try:
        monthly = pd.read_csv("rainfall_monthly.csv", parse_dates=["date"], index_col="date")
    except Exception:
        # If monthly not available yet, check if recent forecast run created 'monthly' variable
        if 'monthly' in globals():
            pass
        else:
            st.info("No monthly data found. Go to Forecast tab and run prediction to fetch data for your selected location and date range.")
            monthly = None

    if monthly is not None:
        # Ensure proper column name
        col = "rainfall" if "rainfall" in monthly.columns else monthly.columns[0]

    # Monthly time series (clean, non-negative, rolling mean)
        st.subheader("Monthly rainfall")
        st.markdown("""
        **How to read this:** The bar chart shows total rainfall for each month. The colored line is the rolling
        average (adjustable in the sidebar). Hover to see exact monthly values. Values are clipped to zero.
        """)
        # Plotly interactive monthly chart (consistent with other charts)
        series = monthly[col].clip(lower=0)
        rolling = series.rolling(window=rolling_window, min_periods=1).mean()
        y_max = max(series.max() if len(series)>0 else 0, rolling.max() if len(rolling)>0 else 0) * 1.15
        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(x=series.index, y=series.values, name='Monthly', marker_color='skyblue'))
        fig_month.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode='lines', name=f'{rolling_window}-mo rolling mean', line=dict(color='crimson', width=3)))
        fig_month.update_layout(title='Monthly rainfall over time', xaxis_title='Date', yaxis_title='Rainfall (mm)', yaxis=dict(range=[0, y_max if y_max>0 else 1], gridcolor='LightGray'))
        fig_month.update_xaxes(dtick='M12', tickformat='%Y')
        st.plotly_chart(fig_month, use_container_width=True)

        with st.expander('Yearly totals (click to expand)', expanded=False):
            st.subheader("Yearly total rainfall")
            st.markdown("""
            **How to read this:** Each bar is the total rainfall in a calendar year. Use this to compare
            wet and dry years at a glance. When comparing two locations, the bars are grouped side-by-side.
            Longer-term trends are useful for planning crops and water resources.
            """)
            # Yearly totals
            yearly = monthly.resample('YE').sum()
            # prepare non-negative yearly values
            yearly_vals = yearly[col].clip(lower=0)
            if compare and ('monthly2' in globals() and globals().get('monthly2') is not None):
                yearly2 = globals()['monthly2'].resample('YE').sum()
                yearly2_vals = yearly2[col].clip(lower=0)
                df_compare = pd.DataFrame({f'{city or "Loc1"}': yearly_vals.values, f'{city2 or "Loc2"}': yearly2_vals.values}, index=yearly.index.year)
                fig_y = go.Figure()
                fig_y.add_trace(go.Bar(x=df_compare.index.astype(str), y=df_compare[f'{city or "Loc1"}'], name=f'{city or "Loc1"}', marker_color='seagreen'))
                fig_y.add_trace(go.Bar(x=df_compare.index.astype(str), y=df_compare[f'{city2 or "Loc2"}'], name=f'{city2 or "Loc2"}', marker_color='lightsalmon'))
                fig_y.update_layout(barmode='group', xaxis_title='Year', yaxis_title='Total annual rainfall (mm)', yaxis=dict(range=[0, max(df_compare.max().max()*1.05, 1)], gridcolor='LightGray'))
                st.plotly_chart(fig_y, use_container_width=True)
            else:
                fig_y = go.Figure(go.Bar(x=yearly.index.year.astype(str), y=yearly_vals.values, marker_color='seagreen'))
                fig_y.update_layout(xaxis_title='Year', yaxis_title='Total annual rainfall (mm)', yaxis=dict(range=[0, max(yearly_vals.max()*1.05, 1)], gridcolor='LightGray'))
                st.plotly_chart(fig_y, use_container_width=True)

        with st.expander('Seasonal totals (click to expand)', expanded=False):
            # Seasonal averages (monsoon: Jun-Sep, summer: Mar-May, winter: Dec-Feb)
            df = monthly.copy()
            df['month'] = df.index.month
            s_monsoon = df[df['month'].isin([6,7,8,9])][col]
            monsoon = s_monsoon.groupby(s_monsoon.index.year).sum()
            s_summer = df[df['month'].isin([3,4,5])][col]
            summer = s_summer.groupby(s_summer.index.year).sum()
            s_winter = df[df['month'].isin([12,1,2])][col]
            winter = s_winter.groupby(s_winter.index.year).sum()
            seasonal = pd.DataFrame({'monsoon': monsoon, 'summer': summer, 'winter': winter}).fillna(0).clip(lower=0)
            st.subheader("Seasonal totals (per year)")
            st.markdown("""
            **How to read this:** The lines show total rainfall for each season (Monsoon, Summer, Winter) per year.
            Seasons are defined as: Monsoon = Jun–Sep, Summer = Mar–May, Winter = Dec–Feb.
            Use this to spot if certain seasons are becoming wetter or drier over time.
            """)
            # Interactive seasonal line chart
            fig_s = go.Figure()
            for col_s, color in zip(['monsoon','summer','winter'], ['royalblue','orange','darkcyan']):
                fig_s.add_trace(go.Scatter(x=seasonal.index, y=seasonal[col_s].values, mode='lines+markers', name=col_s.capitalize(), line=dict(color=color)))
            y_max_s = max(seasonal.max().max()*1.1, 1)
            fig_s.update_layout(xaxis_title='Year', yaxis_title='Seasonal total rainfall (mm)', yaxis=dict(range=[0, y_max_s], gridcolor='LightGray'))
            st.plotly_chart(fig_s, use_container_width=True)
            # Download seasonal chart
            try:
                s_bytes, s_mime = _fig_to_bytes(fig_s)
                st.download_button('Download seasonal chart', data=s_bytes, file_name='seasonal_totals.png', mime=s_mime)
            except Exception:
                pass

        with st.expander('Monthly heatmap (click to expand)', expanded=False):
            # Heatmap: year vs month
            heat = monthly.copy()
            heat['year'] = heat.index.year
            heat['month'] = heat.index.month
            pivot = heat.pivot_table(values=col, index='year', columns='month', aggfunc='sum').fillna(0).clip(lower=0)
            st.subheader("Heatmap: month vs year (total rainfall)")
            st.markdown("""
            **How to read this:** Rows are years and columns are months. Darker cells mean more rain in that month/year.
            This view makes it easy to spot seasonal patterns, anomalously wet/dry months, and multi-year shifts.
            Click and drag to zoom; hover for exact values.
            """)
            # Plotly heatmap: y=years, x=months
            months = [1,2,3,4,5,6,7,8,9,10,11,12]
            fig_h = go.Figure(data=go.Heatmap(z=pivot.values, x=months, y=pivot.index.astype(str), colorscale='Blues', zmin=0))
            fig_h.update_layout(xaxis_title='Month', yaxis_title='Year')
            st.plotly_chart(fig_h, use_container_width=True)
            # Download heatmap
            try:
                h_bytes, h_mime = _fig_to_bytes(fig_h)
                st.download_button('Download heatmap', data=h_bytes, file_name='heatmap.html' if h_mime=='text/html' else 'heatmap.png', mime=h_mime)
            except Exception:
                pass

        with st.expander('Average monthly rainfall (click to expand)', expanded=False):
            # Highlight top/bottom months overall
            monthly['month_name'] = monthly.index.strftime('%b')
            month_means = monthly.groupby('month_name')[col].mean().reindex(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
            st.subheader("Average rainfall by month (long-term)")
            st.markdown("""
            **How to read this:** Shows the long-term average rainfall for each month (averaged across all years).
            The highlighted bars show the historically wettest and driest months — helpful for planting schedules.
            """)
            # Average rainfall by month (Plotly)
            month_means = month_means.clip(lower=0)
            months_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(x=months_order, y=month_means.values, marker_color='cornflowerblue', name='Avg rainfall'))
            # highlight highest and lowest
            hi = month_means.idxmax()
            lo = month_means.idxmin()
            # add colored markers for hi/lo
            fig_b.add_trace(go.Bar(x=[hi], y=[month_means[hi]], marker_color='tomato', name='Highest'))
            fig_b.add_trace(go.Bar(x=[lo], y=[month_means[lo]], marker_color='gold', name='Lowest'))
            fig_b.update_layout(yaxis_title='Avg rainfall (mm)', barmode='overlay')
            fig_b.update_yaxes(range=[0, max(month_means.max()*1.15, 1)])
            st.plotly_chart(fig_b, use_container_width=True)
            # Download monthly average chart
            try:
                b_bytes, b_mime = _fig_to_bytes(fig_b)
                st.download_button('Download monthly-avg chart', data=b_bytes, file_name='monthly_avg.png', mime=b_mime)
            except Exception:
                pass

        # Interactive map (Plotly) — use scatter_geo for reliable base map without Mapbox token
        st.subheader("Location map")
        st.markdown("""
        **How to read this:** Each marker is a selected location. Marker color (and the colorbar) shows the
        average monthly rainfall over the last 12 months (mm). Marker size is scaled for visibility only.
        Categories (Low / Medium / High) are computed from the shown locations' values and provide a quick
        qualitative guide — see the numeric legend below for exact cutoffs.
        Hover a marker to see the location name and exact value.
        """)
        try:
            avg_recent = monthly[col].tail(12).mean()
        except Exception:
            avg_recent = monthly[col].mean()
        map_entries = [{'lat': lat, 'lon': lon, 'avg_rain': float(avg_recent) if pd.notnull(avg_recent) else 0.0, 'city': city or 'Selected location'}]
        if compare:
            # prepare second location average if available
            try:
                df2 = fetch_rainfall(lat=lat2, lon=lon2, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
                df2.to_csv("rainfall_data_loc2.csv")
                monthly2 = preprocess("rainfall_data_loc2.csv")
                avg2 = monthly2[col].tail(12).mean()
            except Exception:
                avg2 = 0.0
            map_entries.append({'lat': lat2, 'lon': lon2, 'avg_rain': float(avg2), 'city': city2 or 'Location 2'})
        map_df = pd.DataFrame(map_entries)
        # Ensure marker sizes are non-negative and have a sensible minimum
        map_df['avg_rain'] = map_df['avg_rain'].fillna(0)
        map_df['size_val'] = (map_df['avg_rain'].clip(lower=0) / max(map_df['avg_rain'].max(), 1) * 30).clip(lower=6)
        # create a simple categorical interpretation for quick reading
        q1 = map_df['avg_rain'].quantile(0.33)
        q2 = map_df['avg_rain'].quantile(0.66)
        def cat(v):
            if v <= q1:
                return 'Low'
            elif v <= q2:
                return 'Medium'
            else:
                return 'High'
        map_df['category'] = map_df['avg_rain'].apply(cat)

        # compute discrete Viridis swatches at midpoints so categories have consistent, expected colors
        try:
            min_val = map_df['avg_rain'].min()
            max_val = map_df['avg_rain'].max()
            def _norm(v):
                return 0.5 if max_val == min_val else (v - min_val) / (max_val - min_val)
            mid_low = (min_val + q1) / 2.0
            mid_med = (q1 + q2) / 2.0
            mid_high = (q2 + max_val) / 2.0
            cmap = plt.cm.get_cmap('viridis')
            col_low = mcolors.to_hex(cmap(_norm(mid_low)))
            col_med = mcolors.to_hex(cmap(_norm(mid_med)))
            col_high = mcolors.to_hex(cmap(_norm(mid_high)))
            color_map = {'Low': col_low, 'Medium': col_med, 'High': col_high}
        except Exception:
            # fallback to sensible defaults
            color_map = {'Low': '#2b83ba', 'Medium': '#abdda4', 'High': '#fdae61'}

        # Use discrete category colors so Low/Medium/High map to the exact swatches
        fig_map = px.scatter_geo(
            map_df,
            lat='lat',
            lon='lon',
            color='category',
            size='size_val',
            hover_name='city',
            hover_data={'avg_rain':':.2f','category':True,'lat':False,'lon':False,'size_val':False},
            projection='natural earth',
            color_discrete_map=color_map,
            title='Average monthly rainfall (last 12 months)'
        )
        fig_map.update_traces(marker=dict(opacity=0.9), selector=dict(mode='markers'))
        # Render map + interactive legend side-by-side
        try:
            left_col, right_col = st.columns([3,1])

            # Build interactive legend/controls in the right column
            with right_col:
                st.markdown("**Map controls & legend**")
                # compute cutoffs for display
                try:
                    cutoffs = [round(float(x),2) for x in [q1, q2]]
                except Exception:
                    cutoffs = [0.0, 0.0]

                # Explain limitation: clicks on Plotly legend can't be captured by Streamlit, provide controls
                st.caption("Tip: use the controls below to filter categories. Clicking the plotly legend does not trigger Streamlit callbacks, so use these buttons to filter.")

                # interactive selection - allow multiple categories
                all_options = ['Low', 'Medium', 'High']
                selected = st.multiselect('Show categories', options=['All'] + all_options, default=['All'])
                if 'All' in selected or len(selected) == 0:
                    selected_cats = all_options
                else:
                    selected_cats = [s for s in selected if s in all_options]

                if st.button('Reset selection'):
                    # reset by re-running with 'All'
                    st.experimental_rerun()

                # numeric legend display
                try:
                    # reuse the precomputed discrete color_map so legend matches the map
                    col_low = color_map.get('Low', '#2b83ba')
                    col_med = color_map.get('Medium', '#abdda4')
                    col_high = color_map.get('High', '#fdae61')
                    legend_df = pd.DataFrame({
                        'Category': ['Low', 'Medium', 'High'],
                        'Range (mm)': [f"<= {cutoffs[0]}", f"> {cutoffs[0]} and <= {cutoffs[1]}", f"> {cutoffs[1]}"],
                        'Color': [col_low, col_med, col_high]
                    })

                    # Recommendation verbosity/localization control
                    rec_level = st.selectbox('Recommendation level', options=['Simple (user-friendly)', 'Detailed (expert)'], index=0)

                    # Two levels of recommendation text: short (for non-experts) and detailed (for extension/advisors)
                    rec_simple = {
                        'Low': 'Use drought-resistant/short-duration crops; conserve water.',
                        'Medium': 'Suitable for most crops; follow normal practices.',
                        'High': 'Consider water-loving crops; ensure drainage.'
                    }
                    rec_detailed = {
                        'Low': ('Low rainfall (dry): prioritize drought-tolerant varieties (e.g., millets, sorghum),'
                                ' consider staggered sowing to avoid dry spells, implement rainwater harvesting, and use ' 
                                'mulching/zero-till to conserve soil moisture.'),
                        'Medium': ('Moderate rainfall: follow standard seed rates and fertilization. Consider intercrops ' 
                                   'to improve resilience. Monitor onset of monsoon and plan sowing dates accordingly.'),
                        'High': ('High rainfall: good for paddy and other water-loving crops. Ensure fields have proper ' 
                                 'drainage, adopt raised beds where needed, and implement erosion control measures.')
                    }

                    # choose which recommendation to display in the compact table
                    if rec_level.startswith('Simple'):
                        display_recs = [rec_simple[c] for c in legend_df['Category']]
                    else:
                        # show a short headline even in the detailed mode in the table; full details are in the expander
                        display_recs = [r.split(':')[0] for r in rec_detailed.values()]

                    # Render a clear legend with colored swatches using Streamlit columns (more reliable)
                    st.markdown('**Legend**')
                    for cat, rng, col_hex in zip(legend_df['Category'], legend_df['Range (mm)'], legend_df['Color']):
                        c1, c2 = st.columns([0.15, 3])
                        with c1:
                            st.markdown(f"<div style='width:18px;height:14px;background:{col_hex};border-radius:2px;'></div>", unsafe_allow_html=True)
                        with c2:
                            # show category, range and short recommendation inline
                            idx = legend_df[legend_df['Category'] == cat].index[0]
                            rec_text = display_recs[idx]
                            st.markdown(f"**{cat}** &nbsp; {rng}  \\ {rec_text}")

                    # Editable recommendations: allow users to customize and save guidance per band
                    # Load user overrides from session or disk
                    import json, os
                    rec_file = os.path.join(os.path.dirname(__file__), 'user_recs.json')
                    defaults = {
                        'Low': {'simple': rec_simple['Low'], 'detailed': rec_detailed['Low']},
                        'Medium': {'simple': rec_simple['Medium'], 'detailed': rec_detailed['Medium']},
                        'High': {'simple': rec_simple['High'], 'detailed': rec_detailed['High']}
                    }
                    # try load from disk
                    user_recs_disk = {}
                    try:
                        if os.path.exists(rec_file):
                            with open(rec_file, 'r', encoding='utf-8') as fh:
                                user_recs_disk = json.load(fh)
                    except Exception:
                        user_recs_disk = {}

                    # merge defaults with disk
                    merged = {k: {**defaults[k], **user_recs_disk.get(k, {})} for k in defaults}

                    # show editable fields inside expander
                    with st.expander('More info: recommended farming practices for each band (editable)'):
                        for cat in ['Low','Medium','High']:
                            st.markdown(f"**{cat}** — {legend_df.loc[legend_df['Category']==cat,'Range (mm)'].values[0]}")
                            simple_key = f"rec_{cat}_simple"
                            detailed_key = f"rec_{cat}_detailed"
                            # prefills: session_state -> disk -> defaults
                            pre_simple = st.session_state.get(simple_key, merged[cat]['simple'])
                            pre_detailed = st.session_state.get(detailed_key, merged[cat]['detailed'])
                            new_simple = st.text_area(f"{cat} — Short recommendation", value=pre_simple, key=simple_key, height=80)
                            new_detailed = st.text_area(f"{cat} — Detailed guidance", value=pre_detailed, key=detailed_key, height=140)
                            st.markdown('---')

                        col_save, col_reset = st.columns([1,1])
                        with col_save:
                            if st.button('Save recommendations'):
                                # gather and persist
                                out = {}
                                for cat in ['Low','Medium','High']:
                                    out[cat] = {'simple': st.session_state.get(f"rec_{cat}_simple"), 'detailed': st.session_state.get(f"rec_{cat}_detailed")} 
                                try:
                                    with open(rec_file, 'w', encoding='utf-8') as fh:
                                        json.dump(out, fh, ensure_ascii=False, indent=2)
                                    st.success('Recommendations saved to user_recs.json')
                                except Exception as e:
                                    st.error(f'Failed to save recommendations: {e}')
                        with col_reset:
                            if st.button('Reset to defaults'):
                                # clear session overrides and remove disk file
                                for cat in ['Low','Medium','High']:
                                    for suf in ['simple','detailed']:
                                        key = f"rec_{cat}_{suf}"
                                        if key in st.session_state:
                                            del st.session_state[key]
                                try:
                                    if os.path.exists(rec_file):
                                        os.remove(rec_file)
                                    st.success('Reset to defaults')
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f'Failed to reset defaults: {e}')
                except Exception:
                    try:
                        st.markdown(f"**Map legend:** Low <= {cutoffs[0]} mm  |  Medium <= {cutoffs[1]} mm  |  High > {cutoffs[1]} mm")
                    except Exception:
                        st.markdown("*Map legend not available.*")

            # Filter the map dataframe according to selection, then show it on the left
            try:
                filtered_df = map_df[map_df['category'].isin(selected_cats)].copy()
                if filtered_df.empty:
                    # show a hint and still display full map
                    with left_col:
                        st.info('No locations match the selected categories — showing all locations instead.')
                        st.plotly_chart(fig_map, use_container_width=True)
                else:
                    # rebuild the map for filtered set
                    # filtered map using discrete category colors for consistency with legend
                    fig_map_f = px.scatter_geo(
                        filtered_df,
                        lat='lat',
                        lon='lon',
                        color='category',
                        size='size_val',
                        hover_name='city',
                        hover_data={'avg_rain':':.2f','category':True,'lat':False,'lon':False,'size_val':False},
                        projection='natural earth',
                        color_discrete_map=color_map,
                        title='Average monthly rainfall (last 12 months)'
                    )
                    fig_map_f.update_traces(marker=dict(opacity=0.9), selector=dict(mode='markers'))
                    with left_col:
                        st.plotly_chart(fig_map_f, use_container_width=True)
            except Exception:
                # fallback: show original map and textual legend
                with left_col:
                    st.plotly_chart(fig_map, use_container_width=True)
                with right_col:
                    try:
                        st.markdown(f"**Map legend (categories):**\n\n- Low: <= {round(q1,2)} mm\n- Medium: > {round(q1,2)} and <= {round(q2,2)} mm\n- High: > {round(q2,2)} mm\n")
                    except Exception:
                        st.markdown("*Category cutoffs not available.*")

        except Exception:
            # single-column fallback
            st.plotly_chart(fig_map, use_container_width=True)
            try:
                cutoffs = [round(float(x),2) for x in [q1, q2]]
                st.markdown(f"**Map legend (categories):**\n\n- Low: <= {cutoffs[0]} mm\n- Medium: > {cutoffs[0]} and <= {cutoffs[1]} mm\n- High: > {cutoffs[1]} mm\n")
            except Exception:
                st.markdown("*Category cutoffs not available.*")

        # Visual separator to avoid overlapping/duplicated text below the map
        st.markdown('---')

with tab_forecast:
    st.header("Forecast")

    # --- MOVED ALL THESE CONTROLS INTO THE 'if advanced:' BLOCK ---
    if advanced:
        st.subheader("Advanced Model Selection")
        # Model selection
        model_choice = st.multiselect("Select model(s) to run", options=["SARIMA", "Prophet"], default=["SARIMA"])
        auto_select = st.checkbox('Auto-select best model via CV (uses rolling-origin CV)', value=False)
        
        # Auto ARIMA option
        use_auto_arima = False
        if PMDARIMA_AVAILABLE:
            use_auto_arima = st.checkbox('Enable Auto-ARIMA (pmdarima)', value=False)
            if use_auto_arima and 'AutoARIMA' not in model_choice:
                model_choice.append('AutoARIMA')
        else:
            st.caption('Auto-ARIMA not available (install pmdarima to enable).')

        # Optional SARIMA grid-search
        use_sarima_grid = st.checkbox('Use SARIMA grid-search (no pmdarima required)', value=False)
        if use_sarima_grid:
            p_max = st.slider('Max p', min_value=0, max_value=3, value=2)
            q_max = st.slider('Max q', min_value=0, max_value=3, value=2)
            P_max = st.slider('Max P (seasonal)', min_value=0, max_value=2, value=1)

        # Cross-validation controls
        st.markdown('**Model cross-validation (rolling-origin)**')
        cv_horizon = st.number_input('CV horizon (months per fold)', min_value=1, max_value=24, value=3)
        cv_folds = st.number_input('CV folds', min_value=1, max_value=6, value=3)
        run_cv = st.button('Run CV & Compare Models')
        st.markdown("---") # Add a separator
        
    else:
        # --- SET DEFAULTS SILENTLY FOR A FRESHER ---
        model_choice = ["SARIMA"]
        auto_select = False # Keep it simple, just run the default
        use_auto_arima = False
        use_sarima_grid = False
        cv_horizon = 3
        cv_folds = 3
        run_cv = False
    
    # --- THIS LINE WAS ALREADY HERE, BUT ADD THE "st.info" BENEATH IT ---
    run_clicked = st.button("Run Prediction")
    st.info("This may take a moment. We're fetching data and training an AI model for your specific location.") # <-- ADD THIS

    # --- LEAVE THE REST OF THE TAB (replay_clicked, etc.) AS IT IS ---
    # Optional SARIMA grid-search (no extra dependencies)
    use_sarima_grid = st.checkbox('Use SARIMA grid-search (no pmdarima required)', value=False)
    if use_sarima_grid:
        p_max = st.slider('Max p', min_value=0, max_value=3, value=2)
        q_max = st.slider('Max q', min_value=0, max_value=3, value=2)
        P_max = st.slider('Max P (seasonal)', min_value=0, max_value=2, value=1)

    # Cross-validation controls
    st.markdown('**Model cross-validation (rolling-origin)**')
    cv_horizon = st.number_input('CV horizon (months per fold)', min_value=1, max_value=24, value=3)
    cv_folds = st.number_input('CV folds', min_value=1, max_value=6, value=3)
    run_cv = st.button('Run CV & Compare Models')

    run_clicked = st.button("Run Prediction")
    replay_clicked = st.button("Replay last forecast")

    # Shortcut: load targeted SARIMA CV results (two-stage fast run) and allow user to apply best order
    try:
        import os, json
        targeted_path = os.path.join(os.path.dirname(__file__), 'tools', 'sarima_grid_cv_targeted_results.json')
        if os.path.exists(targeted_path):
            try:
                with open(targeted_path, 'r', encoding='utf-8') as fh:
                    _tres = json.load(fh)
                best = _tres.get('best_by_cv')
                topk = _tres.get('topk', [])
                if best is not None:
                    col_cv = st.container()
                    with col_cv:
                        st.markdown('**CV-selected SARIMA (from last targeted run)**')
                        try:
                            ord_tuple = tuple(best.get('order'))
                            s_ord = tuple(best.get('seasonal_order'))
                            st.write(f"Best order: `{ord_tuple}`  seasonal: `{s_ord}`")
                            st.write(f"AIC: `{best.get('aic')}`  CV_RMSE: `{best.get('cv_rmse')}`")
                        except Exception:
                            st.write(str(best))

                        # show top-K table with apply buttons
                        if topk:
                            st.markdown('Top candidates (by AIC then CV):')
                            df_top = pd.DataFrame([{
                                'order': tuple(c.get('order')),
                                'seasonal_order': tuple(c.get('seasonal_order')),
                                'aic': c.get('aic'),
                                'cv_rmse': c.get('cv_rmse')
                            } for c in topk])
                            # show table
                            st.dataframe(df_top)

                            # allow selecting one to apply
                            sel_idx = st.number_input('Select top-K index to apply (0 = best)', min_value=0, max_value=max(len(df_top)-1,0), value=0)
                            apply_col1, apply_col2 = st.columns([1,1])
                            with apply_col1:
                                if st.button('Apply selected for this run'):
                                    chosen = df_top.iloc[int(sel_idx)]
                                    st.session_state['best_sarima_order'] = (tuple(chosen['order']), tuple(chosen['seasonal_order']))
                                    st.session_state['best_cv_summary'] = chosen.to_dict()
                                    st.success('Selected candidate will be used for this run (session override set).')
                            with apply_col2:
                                if st.button('Make selected the default for all runs'):
                                        chosen = df_top.iloc[int(sel_idx)]
                                        # persist to session as default
                                        st.session_state['best_sarima_order_default'] = (tuple(chosen['order']), tuple(chosen['seasonal_order']))
                                        st.session_state['best_cv_summary_default'] = chosen.to_dict()
                                        # also persist to disk so it survives restarts
                                        try:
                                            out_path = os.path.join(os.path.dirname(__file__), 'tools', 'best_sarima_default.json')
                                            with open(out_path, 'w', encoding='utf-8') as _of:
                                                json.dump({'order': list(chosen['order']), 'seasonal_order': list(chosen['seasonal_order']), 'aic': chosen.get('aic'), 'cv_rmse': chosen.get('cv_rmse')}, _of, indent=2)
                                            st.success(f'Selected candidate saved as default (file: {out_path})')
                                        except Exception as _e:
                                            st.warning(f'Could not persist default to disk: {_e} — saved only to session.')

                        # show controls to use or clear saved best
                        c1, c2 = st.columns([1,1])
                        with c1:
                            if st.button('Use best SARIMA order for this run'):
                                st.session_state['best_sarima_order'] = (ord_tuple, s_ord)
                                st.session_state['best_cv_summary'] = best
                                st.info('Best SARIMA order saved to session and will be used when fitting below.')
                        with c2:
                            if st.button('Clear saved SARIMA override'):
                                for k in ['best_sarima_order','best_cv_summary','best_sarima_order_default','best_cv_summary_default']:
                                    if k in st.session_state:
                                        del st.session_state[k]
                                # remove persisted default file as well
                                try:
                                    fp = os.path.join(os.path.dirname(__file__), 'tools', 'best_sarima_default.json')
                                    if os.path.exists(fp):
                                        os.remove(fp)
                                except Exception:
                                    pass
                                st.info('Cleared saved SARIMA overrides from session and disk (if present).')
            except Exception:
                pass
    except Exception:
        pass
    if replay_clicked:
        # re-render from cache if present
        last = st.session_state.get('last_forecast')
        if last is None:
            st.warning('No cached forecast found. Run a new prediction first.')
        else:
            # inject cached objects into local names used by renderer
            monthly = last.get('monthly')
            pred_mean = last.get('pred_mean')
            conf_int = last.get('conf_int')
            fc_df = last.get('fc_df')
            seasonal_results = last.get('seasonal_results')
            prophet_pred = last.get('prophet_pred')
            # Render EDA chart
            series_all = monthly["rainfall"].clip(lower=0)
            rolling_all = series_all.rolling(window=rolling_window, min_periods=1).mean()
            y_max_all = max(series_all.max() if len(series_all)>0 else 0, rolling_all.max() if len(rolling_all)>0 else 0) * 1.15
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=series_all.index, y=series_all.values, name='Monthly', marker_color='skyblue', hovertemplate='%{x|%Y-%m}: %{y:.2f} mm'))
            fig1.add_trace(go.Scatter(x=rolling_all.index, y=rolling_all.values, mode='lines', name='Rolling mean', line=dict(color='crimson', width=3)))
            fig1.update_layout(title='Monthly Rainfall (mm) [Cached]', xaxis_title='Date', yaxis_title='Rainfall (mm)', yaxis=dict(range=[0, y_max_all if y_max_all>0 else 1], gridcolor='LightGray'))
            st.plotly_chart(fig1, use_container_width=True)
            # SARIMA chart
            fig2 = go.Figure()
            train_vals = last.get('train_vals')
            test_vals = last.get('test_vals')
            fig2.add_trace(go.Scatter(x=last.get('train_index'), y=train_vals, mode='lines', name='Train', line=dict(color='royalblue')))
            fig2.add_trace(go.Scatter(x=last.get('test_index'), y=test_vals, mode='lines+markers', name='Test', line=dict(color='orange')))
            fig2.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values, mode='lines+markers', name='SARIMA Forecast', line=dict(color='green')))
            if show_ci and conf_int is not None:
                ci_x = list(conf_int.index) + list(conf_int.index[::-1])
                ci_y = list(conf_int.iloc[:,1].values) + list(conf_int.iloc[:,0].values[::-1])
                ci_y = [max(0, v) for v in ci_y]
                fig2.add_trace(go.Scatter(x=ci_x, y=ci_y, fill='toself', fillcolor='rgba(144,238,144,0.3)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=True, name='95% CI'))
            if len(last.get('test_index', []))>0:
                fig2.add_vline(x=last.get('test_index')[0], line=dict(color='gray', dash='dash'), opacity=0.6)
            fig2.update_layout(title='SARIMA Forecast vs Actuals [Cached]', xaxis_title='Date', yaxis_title='Rainfall (mm)', yaxis=dict(range=[0, max(train_vals.max() if len(train_vals)>0 else 0, test_vals.max() if len(test_vals)>0 else 0, pred_mean.max() if len(pred_mean)>0 else 0) * 1.15], gridcolor='LightGray'))
            st.plotly_chart(fig2, use_container_width=True)
        # skip the normal run branch
    if run_cv:
        # run rolling-origin CV on the available monthly series
        with st.spinner('Running cross-validation (this may take a while)...'):
            series = monthly['rainfall'].clip(lower=0)
            models_to_eval = ['SARIMA', 'Prophet']
            if PMDARIMA_AVAILABLE and use_auto_arima:
                models_to_eval.append('AutoARIMA')
            cv_res = rolling_origin_cv(series, horizon=int(cv_horizon), n_splits=int(cv_folds), models=models_to_eval)
            # show results
            rows = []
            for m, stats in cv_res.items():
                rows.append({'model': m, 'rmse': stats['rmse'], 'mae': stats['mae']})
            cv_df = pd.DataFrame(rows).set_index('model')
            st.subheader('Cross-validation results')
            st.table(cv_df)
            # save to session so user can compare after runs
            st.session_state['last_cv'] = cv_res
            # also prepare fold-level details for download and inspection
            # cv_res[m]['details'] is a list of dicts per fold with rmse/mae
            try:
                rows_d = []
                for m, stats in cv_res.items():
                    dets = stats.get('details', [])
                    for i, d in enumerate(dets):
                        rows_d.append({'model': m, 'fold': i, 'rmse': d.get('rmse'), 'mae': d.get('mae')})
                if rows_d:
                    df_details = pd.DataFrame(rows_d)
                    st.markdown('**Per-fold CV details**')
                    st.dataframe(df_details)
                    try:
                        csv_d = df_details.to_csv(index=False).encode('utf-8')
                        st.download_button('Download CV fold-level details (CSV)', data=csv_d, file_name='cv_fold_details.csv', mime='text/csv')
                    except Exception:
                        pass
                    st.session_state['last_cv_details'] = df_details.to_dict(orient='records')
            except Exception:
                pass
    if run_clicked:
        with st.spinner("Fetching and processing data..."):
            today = end_date.strftime("%Y-%m-%d")
            df = fetch_rainfall(lat=lat, lon=lon, start=start_date.strftime("%Y-%m-%d"), end=today)
            df.to_csv("rainfall_data.csv")
            monthly = preprocess("rainfall_data.csv")
            # If comparing, fetch second location data and create monthly2
            if compare:
                try:
                    df2 = fetch_rainfall(lat=lat2, lon=lon2, start=start_date.strftime("%Y-%m-%d"), end=today)
                    df2.to_csv("rainfall_data_loc2.csv")
                    monthly2 = preprocess("rainfall_data_loc2.csv")
                    globals()['monthly2'] = monthly2
                except Exception:
                    globals()['monthly2'] = None

    # EDA plot (interactive with Plotly)
    series_all = monthly["rainfall"].clip(lower=0)
    rolling_all = series_all.rolling(window=rolling_window, min_periods=1).mean()
    y_max_all = max(series_all.max() if len(series_all)>0 else 0, rolling_all.max() if len(rolling_all)>0 else 0) * 1.15
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=series_all.index, y=series_all.values, name='Monthly', marker_color='skyblue', hovertemplate='%{x|%Y-%m}: %{y:.2f} mm'))
    fig1.add_trace(go.Scatter(x=rolling_all.index, y=rolling_all.values, mode='lines', name='12-mo rolling mean', line=dict(color='crimson', width=3)))
    fig1.update_layout(title='Monthly Rainfall (mm)', xaxis_title='Date', yaxis_title='Rainfall (mm)', yaxis=dict(range=[0, y_max_all if y_max_all>0 else 1], gridcolor='LightGray'))
    fig1.update_xaxes(dtick="M12", tickformat="%Y")
    st.plotly_chart(fig1, use_container_width=True)

    # download EDA figure
    try:
        eda_bytes, eda_mime = _fig_to_bytes(fig1)
        st.download_button("Download Monthly chart", data=eda_bytes, file_name="monthly_rainfall.png", mime=eda_mime)
    except Exception:
        pass

    # SARIMA Forecast
    train = monthly.iloc[:-12]
    test = monthly.iloc[-12:]
    # If user requested auto-selection, run CV on the training window to pick the best model
    if auto_select:
        with st.spinner('Running CV to auto-select best model...'):
            eval_models = list(model_choice)
            # include AutoARIMA if available and enabled
            if PMDARIMA_AVAILABLE and use_auto_arima and 'AutoARIMA' not in eval_models:
                eval_models.append('AutoARIMA')
            best_model, cv_res = select_best_model_via_cv(train['rainfall'].clip(lower=0), eval_models, cv_horizon, cv_folds)
            st.info(f'Auto-selected model: **{best_model}** (based on CV RMSE)')
            # show cv table
            rows = []
            for m, stats in cv_res.items():
                rows.append({'model': m, 'rmse': stats.get('rmse'), 'mae': stats.get('mae')})
            st.table(pd.DataFrame(rows).set_index('model'))
            # enforce selected model for the remainder of the run
            model_choice = [best_model]
    # fit SARIMA (with optional caching or grid-search). Honor any user/session override from targeted CV.
    # If a best order was saved into session_state['best_sarima_order'], prefer that.
    # prefer an explicit per-run override, otherwise use a saved default override if present
    sarima_override = None
    if isinstance(st.session_state.get('best_sarima_order'), (list, tuple)):
        sarima_override = st.session_state.get('best_sarima_order')
    elif isinstance(st.session_state.get('best_sarima_order_default'), (list, tuple)):
        sarima_override = st.session_state.get('best_sarima_order_default')
    # fit SARIMA (with optional caching or grid-search)
    # ensure `result` is always defined to avoid NameError in downstream logic
    result = None
    if sarima_override is not None:
        # user requested a specific order from CV results
        try:
            order_use, seasonal_use = sarima_override
            result = fit_sarima_cached(train['rainfall'], order=tuple(order_use), seasonal_order=tuple(seasonal_use))
        except Exception:
            # fallback to default grid or default model
            sarima_override = None
            result = None
    elif 'use_sarima_grid' in globals() and use_sarima_grid:
        # run lightweight grid search
        p_range = list(range(0, p_max+1))
        q_range = list(range(0, q_max+1))
        P_range = list(range(0, P_max+1))
        best = sarima_grid_search(train['rainfall'], p_range=p_range, q_range=q_range, P_range=P_range, Q_range=(0,1), s=12)
        if best is not None:
            order, seasonal_order, res_model = best
            result = res_model
        else:
            result = fit_sarima_cached(train["rainfall"], order=(1,1,1), seasonal_order=(1,1,1,12))
    else:
        # if no override and not using grid, use default SARIMA
        if result is None:
            result = fit_sarima_cached(train["rainfall"], order=(1,1,1), seasonal_order=(1,1,1,12))
    forecast = result.get_forecast(steps=forecast_horizon)
    pred_mean = forecast.predicted_mean.clip(lower=0)
    conf_int = forecast.conf_int()
    # ensure confidence interval lower bounds are non-negative for plotting
    conf_int.iloc[:,0] = conf_int.iloc[:,0].clip(lower=0)
    conf_int.iloc[:,1] = conf_int.iloc[:,1].clip(lower=0)

    # SARIMA Forecast (interactive Plotly)
    train_vals = train["rainfall"].clip(lower=0)
    test_vals = test["rainfall"].clip(lower=0)
    y_top = max(train_vals.max() if len(train_vals)>0 else 0, test_vals.max() if len(test_vals)>0 else 0,
                pred_mean.max() if len(pred_mean)>0 else 0, conf_int.iloc[:,1].max() if conf_int.shape[1]>1 else 0) * 1.15
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train.index, y=train_vals, mode='lines', name='Train', line=dict(color='royalblue')))
    fig2.add_trace(go.Scatter(x=test.index, y=test_vals, mode='lines+markers', name='Test', line=dict(color='orange')))
    fig2.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values, mode='lines+markers', name='SARIMA Forecast', line=dict(color='green')))
    # Add CI as filled area (optional)
    if show_ci:
        ci_x = list(conf_int.index) + list(conf_int.index[::-1])
        ci_y = list(conf_int.iloc[:,1].values) + list(conf_int.iloc[:,0].values[::-1])
        # clip ci values to >=0
        ci_y = [max(0, v) for v in ci_y]
        fig2.add_trace(go.Scatter(x=ci_x, y=ci_y, fill='toself', fillcolor='rgba(144,238,144,0.3)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=True, name='95% CI'))
    # vertical test separator
    if len(test.index)>0:
        fig2.add_vline(x=test.index[0], line=dict(color='gray', dash='dash'), opacity=0.6)
    fig2.update_layout(title='SARIMA Forecast vs Actuals', xaxis_title='Date', yaxis_title='Rainfall (mm)', yaxis=dict(range=[0, y_top if y_top>0 else 1], gridcolor='LightGray'))
    fig2.update_xaxes(tickformat='%Y-%m')
    st.plotly_chart(fig2, use_container_width=True)
    # Download SARIMA figure and forecast CSV
    try:
        sar_bytes, sar_mime = _fig_to_bytes(fig2)
        st.download_button("Download SARIMA chart", data=sar_bytes, file_name="sarima_forecast.png", mime=sar_mime)
    except Exception:
        pass
    try:
        fc_df = pd.DataFrame({'date': pred_mean.index, 'forecast': pred_mean.values})
        csv = fc_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download SARIMA forecast CSV', data=csv, file_name='sarima_forecast.csv', mime='text/csv')
    except Exception:
        pass

    # store forecast outputs in session for replay
    try:
        st.session_state['last_forecast'] = {
            'monthly': monthly,
            'train_index': train.index,
            'test_index': test.index,
            'train_vals': train["rainfall"].clip(lower=0),
            'test_vals': test["rainfall"].clip(lower=0),
            'pred_mean': pred_mean,
            'conf_int': conf_int,
            'fc_df': fc_df,
            'seasonal_results': seasonal_results if 'seasonal_results' in locals() else None,
            'prophet_pred': prophet_pred if 'prophet_pred' in locals() else None
        }
    except Exception:
        pass

    results = []
    # Evaluate SARIMA
    if "SARIMA" in model_choice:
        sarima_pred = pred_mean
        rmse_s = float(np.sqrt(np.mean((test["rainfall"] - sarima_pred) ** 2)))
        mae_s = float(np.mean(np.abs(test["rainfall"] - sarima_pred)))
        results.append({"model": "SARIMA", "rmse": rmse_s, "mae": mae_s})

    # Prophet forecast
    if "Prophet" in model_choice:
        # Prepare data for Prophet
        df_prop = train["rainfall"].reset_index().rename(columns={"date": "ds", "rainfall": "y"})
        m = fit_prophet_cached(df_prop)
        future = m.make_future_dataframe(periods=forecast_horizon, freq='M')
        fcst = m.predict(future)
        # align prophet predictions to test index (if overlap) and clip
        prophet_pred = fcst.set_index('ds')['yhat'].reindex(test.index).clip(lower=0)
        y_top_p = max(train["rainfall"].clip(lower=0).max() if len(train)>0 else 0, test["rainfall"].clip(lower=0).max() if len(test)>0 else 0,
                      prophet_pred.max() if len(prophet_pred)>0 else 0) * 1.15
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=train.index, y=train["rainfall"].clip(lower=0), mode='lines', name='Train', line=dict(color='royalblue')))
        fig_p.add_trace(go.Scatter(x=test.index, y=test["rainfall"].clip(lower=0), mode='lines+markers', name='Test', line=dict(color='orange')))
        fig_p.add_trace(go.Scatter(x=prophet_pred.index, y=prophet_pred.values, mode='lines+markers', name='Prophet Forecast', line=dict(color='purple')))
        if len(test.index)>0:
            fig_p.add_vline(x=test.index[0], line=dict(color='gray', dash='dash'), opacity=0.6)
        fig_p.update_layout(title='Prophet Forecast vs Actuals', xaxis_title='Date', yaxis_title='Rainfall (mm)', yaxis=dict(range=[0, y_top_p if y_top_p>0 else 1], gridcolor='LightGray'))
        fig_p.update_xaxes(tickformat='%Y-%m')
        st.plotly_chart(fig_p, use_container_width=True)
        rmse_p = float(np.sqrt(np.mean((test["rainfall"] - prophet_pred) ** 2)))
        mae_p = float(np.mean(np.abs(test["rainfall"] - prophet_pred)))
        results.append({"model": "Prophet", "rmse": rmse_p, "mae": mae_p})

    if results:
        res_df = pd.DataFrame(results).set_index('model')
        st.subheader("Model comparison (lower is better)")
        st.table(res_df)

        # --- Seasonal forecasting (monsoon/summer/winter) ---
        st.subheader("Seasonal forecasts")
        seasons = {
            'Monsoon (Jun-Sep)': [6,7,8,9],
            'Summer (Mar-May)': [3,4,5],
            'Winter (Dec-Feb)': [12,1,2]
        }
        seasonal_results = []
        for sname, months in seasons.items():
            # aggregate seasonal totals per year
            s_df = monthly.copy()
            s_df['month'] = s_df.index.month
            filtered = s_df[s_df['month'].isin(months)]
            s_series = filtered[col].groupby(filtered.index.year).sum()
            if len(s_series) < 3:
                # not enough history
                seasonal_results.append({'season': sname, 'model': 'NA', 'forecast': None, 'rmse': None, 'mae': None})
                continue

            # prepare yearly index as datetime for prophet
            s_index = pd.to_datetime(s_series.index.astype(str) + '-12-31')
            s_series.index = s_index

            # train/test split (last year as test)
            train_s = s_series.iloc[:-1]
            test_s = s_series.iloc[-1:]

            # SARIMA on yearly series
            try:
                sar_model = sm.tsa.statespace.SARIMAX(train_s,
                                                      order=(1,0,0),
                                                      seasonal_order=(0,0,0,0),
                                                      enforce_stationarity=False,
                                                      enforce_invertibility=False)
                sar_res = sar_model.fit(disp=False)
                sar_fc = sar_res.get_forecast(steps=1).predicted_mean
                sar_pred_val = float(sar_fc.iloc[0])
                # compute metrics if test exists
                rmse_s = float(np.sqrt(np.mean((test_s.values - sar_fc.values) ** 2))) if len(test_s)>0 else None
                mae_s = float(np.mean(np.abs(test_s.values - sar_fc.values))) if len(test_s)>0 else None
            except Exception:
                sar_pred_val = None
                rmse_s = None
                mae_s = None

            seasonal_results.append({'season': sname, 'model': 'SARIMA', 'forecast': sar_pred_val, 'rmse': rmse_s, 'mae': mae_s})

            # Prophet on yearly series
            try:
                df_prop_s = train_s.reset_index().rename(columns={'index':'ds', 0:'y'})
                df_prop_s.columns = ['ds','y']
                m_s = Prophet()
                m_s.fit(df_prop_s)
                future_s = m_s.make_future_dataframe(periods=1, freq='Y')
                fcst_s = m_s.predict(future_s)
                prop_pred = fcst_s.set_index('ds')['yhat'].loc[future_s['ds'].iloc[-1]]
                prop_pred_val = float(prop_pred)
                # metrics
                # align prophet prediction index with test_s index for metric
                if len(test_s)>0:
                    rmse_p = float(np.sqrt(np.mean((test_s.values - prop_pred_val) ** 2)))
                    mae_p = float(np.mean(np.abs(test_s.values - prop_pred_val)))
                else:
                    rmse_p = None
                    mae_p = None
            except Exception:
                prop_pred_val = None
                rmse_p = None
                mae_p = None

            seasonal_results.append({'season': sname, 'model': 'Prophet', 'forecast': prop_pred_val, 'rmse': rmse_p, 'mae': mae_p})

        # show seasonal results table
        sres_df = pd.DataFrame(seasonal_results)
        st.table(sres_df)
        # Download seasonal forecasts CSV
        try:
            s_csv = sres_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download seasonal forecasts CSV', data=s_csv, file_name='seasonal_forecasts.csv', mime='text/csv')
        except Exception:
            pass

        # plot seasonal forecasts per model
        try:
            plot_df = sres_df.pivot(index='season', columns='model', values='forecast')
            st.subheader('Seasonal forecast comparison (next year)')
            st.bar_chart(plot_df.fillna(0))
        except Exception:
            pass

        avg = monthly["rainfall"].mean()
        future_sum = pred_mean.sum()
        crop, crop_msg = recommend_crop(future_sum, season)
        st.success(f"Crop Recommendation: {crop}")
        st.info(crop_msg)
        st.write(f"Average monthly rainfall: **{avg:.2f} mm**")
        st.write(f"Forecasted total rainfall (next 12 months): **{future_sum:.2f} mm**")

        month_crops_df = month_wise_crop_recommendation(pred_mean, season)
        st.subheader("Month-wise Crop Recommendation")
        st.dataframe(month_crops_df)

        # Download report
        report = f"Rainfall Prediction & Crop Recommendation\n"
        report += f"Location: Latitude {lat}, Longitude {lon}\n"
        report += f"Average monthly rainfall: {avg:.2f} mm\n"
        report += f"Forecasted total rainfall (next 12 months): {future_sum:.2f} mm\n"
        report += f"Crop Recommendation: {crop}\n{crop_msg}\n\n"
        report += "Month-wise Crop Recommendation:\n"
        for _, row in month_crops_df.iterrows():
            report += f"{row['Month']}: {row['Recommended Crops']} (Predicted Rainfall: {row['Predicted Rainfall (mm)']:.2f} mm)\n"
        st.download_button("Download Report", report, file_name="rainfall_report.txt")
        # PDF report (images + summary) using reportlab when available
        try:
            images = []
            # include SARIMA chart image
            try:
                sar_bytes, sar_mime = _fig_to_bytes(fig2)
                if sar_mime == 'image/png':
                    images.append(sar_bytes)
            except Exception:
                pass
            # include EDA chart image
            try:
                eda_bytes, eda_mime = _fig_to_bytes(fig1)
                if eda_mime == 'image/png':
                    images.append(eda_bytes)
            except Exception:
                pass
            pdf_bytes = generate_pdf_report(report, filename='rainfall_report.pdf', images=images)
            # If reportlab not installed, generate_pdf_report returns plain bytes of text
            mime = 'application/pdf' if REPORTLAB_AVAILABLE else 'text/plain'
            st.download_button('Download PDF report', data=pdf_bytes, file_name='rainfall_report.pdf' if REPORTLAB_AVAILABLE else 'rainfall_report.txt', mime=mime)
        except Exception:
            pass
    else:
        st.info("Set location and season in the sidebar, then click 'Run Prediction'.")

st.warning("Note: Rainfall data is from NASA POWER. For official planning, use IMD or local sources.")

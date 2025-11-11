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

# --- NEW LIBRARY TO IMPORT ---
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# --- ADDED Pmdarima (Auto-ARIMA) ---
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

# --- ADDED Reportlab (PDF) ---
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# --- All your helper functions (from your old code) ---
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

try:
    import os
    crop_csv = os.path.join(os.path.dirname(__file__), 'crop_db.csv')
    if os.path.exists(crop_csv):
        df_crop = pd.read_csv(crop_csv)
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

# --- NEW HELPER FUNCTIONS FOR PDF REPORT ---
def _fig_to_bytes(fig: go.Figure, fmt: str = 'png') -> tuple[bytes,str]:
    """Return (data, mime) for download. If PNG generation fails, fall back to HTML."""
    try:
        img = fig.to_image(format=fmt)
        return img, 'image/png'
    except Exception:
        html = fig.to_html(include_plotlyjs='cdn')
        return html.encode('utf-8'), 'text/html'

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
                if y < 300: # Give more space for images
                    c.showPage()
                    y = height - 40
                img_buf = io.BytesIO(img_bytes)
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
# --- END OF PDF FUNCTIONS ---

def _series_hash(s: pd.Series) -> str:
    try:
        b = s.to_json().encode()
    except Exception:
        b = pd.Series(s).to_json().encode()
    return hashlib.md5(b).hexdigest()

def _get_cached_model(key: str):
    cache = st.session_state.setdefault('model_cache', {})
    return cache.get(key)

def _set_cached_model(key: str, model_obj):
    cache = st.session_state.setdefault('model_cache', {})
    cache[key] = model_obj

def fit_sarima_cached(train_series: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    key = ('sarima', order, seasonal_order, _series_hash(train_series))
    cached = _get_cached_model(str(key))
    if cached is not None:
        return cached
    model = sm.tsa.statespace.SARIMAX(train_series,
                                      order=order,
                                      seasonal_order=seasonal_order,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    res = model.fit(disp=False)
    _set_cached_model(str(key), res)
    return res

# --- NEW AUTO-ARIMA FUNCTION ---
def fit_auto_arima_cached(train_series: pd.Series, seasonal=True, m=12, max_p=3, max_q=3, max_P=1, max_Q=1):
    """Fit auto_arima (pmdarima) with caching. Falls back to None if pmdarima not available."""
    key = ('auto_arima', seasonal, m, max_p, max_q, max_P, max_Q, _series_hash(train_series))
    cached = _get_cached_model(str(key))
    if cached is not None:
        return cached
    if not PMDARIMA_AVAILABLE:
        return None
    try:
        model = pm.auto_arima(train_series, seasonal=seasonal, m=m, max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        _set_cached_model(str(key), model)
        return model
    except Exception:
        return None
# --- END OF NEW FUNCTION ---

# --- NEW FUNCTION TO FIND COORDINATES FROM A CITY NAME ---
@st.cache_data(ttl=3600)
def get_lat_lon(city_name):
    """Converts a city name to latitude and longitude."""
    if not city_name:
        return None
    try:
        geolocator = Nominatim(user_agent="rainfall_app")
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except (GeocoderTimedOut, GeocoderServiceError):
        st.error("Error: The location service (Nominatim) is busy or unavailable. Please try again later or enter coordinates manually.")
        return None
# --- END OF NEW FUNCTION ---


# --- START OF NEW, SIMPLIFIED APP LOGIC ---

st.set_page_config(page_title="Rainfall & Crop Advisor", layout="wide")

# Dashboard header
st.title("Rainfall Prediction & Crop Recommendation 🌧️🌾")

# Check if a forecast has been run
if 'forecast_run_complete' not in st.session_state:
    
    # --- STEP 1: THE HOME SCREEN (INPUTS) ---
    st.header("Get Your Local Rainfall & Crop Forecast")
    st.markdown("Enter your location to get an AI-powered forecast and crop recommendations.")

    # Input Method: City or Manual Lat/Lon
    input_method = st.radio("How do you want to enter your location?",
                            ("Search by City/Town", "Enter Manually (Advanced)"))

    lat = None
    lon = None
    city_name_for_report = "N/A" # For PDF report
    
    if input_method == "Search by City/Town":
        city = st.text_input("Enter your City, Town, or District name (e.g., 'Hyderabad')")
        if city:
            coords = get_lat_lon(city)
            if coords:
                lat, lon = coords
                city_name_for_report = city # Save city name
                st.success(f"Found location: Latitude={lat:.4f}, Longitude={lon:.4f}")
            else:
                st.error("Could not find that location. Please try a different name or use the manual method.")
    else:
        st.subheader("Manual Coordinate Entry")
        lat = st.number_input("Latitude", value=17.3850, format="%.4f")
        lon = st.number_input("Longitude", value=78.4867, format="%.4f")
        city_name_for_report = f"{lat:.4f}, {lon:.4f}" # Use coords as name
        st.caption("Tip: Find this on Google Maps (right-click on a location).")

    # Shared Inputs
    season = st.selectbox("Select Your Cropping Season:", ["Kharif", "Rabi", "None"])
    season_key = None if season == "None" else season.lower()
    
    start_date = datetime.date(2000,1,1) # Use a fixed historical start date
    end_date = datetime.date.today()
    forecast_horizon = 12 # Use a fixed 12-month forecast

    # --- THE "RUN" BUTTON ---
    if st.button("Get My Forecast", type="primary", use_container_width=True, disabled=(lat is None)):
        
        with st.spinner("This may take a moment... We're fetching years of data and training an AI model for your location."):
            
            # --- 1. Fetch and Process Data ---
            df = fetch_rainfall(lat=lat, lon=lon, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
            df.to_csv("rainfall_data.csv")
            monthly = preprocess("rainfall_data.csv")

            # --- 2. Train the Model (Auto-Pilot AI) ---
            train = monthly.iloc[:-12]
            
            # --- UPDATED: Find the best model automatically ---
            st.write("Finding the best forecast model for your area...") # Show progress
            
            best_model = fit_auto_arima_cached(train["rainfall"])
            
            if best_model is not None:
                st.write("Best model found! Fitting...")
                order = best_model.order
                seasonal_order = best_model.seasonal_order
                result = fit_sarima_cached(train["rainfall"], order=order, seasonal_order=seasonal_order)
            else:
                st.write("Auto-ARIMA not available, using standard model...")
                result = fit_sarima_cached(train["rainfall"], order=(1,1,1), seasonal_order=(1,1,1,12))
            # --- END OF UPDATE ---
            
            # --- 3. Get Forecast ---
            forecast = result.get_forecast(steps=forecast_horizon)
            pred_mean = forecast.predicted_mean.clip(lower=0)
            conf_int = forecast.conf_int()
            conf_int.iloc[:,0] = conf_int.iloc[:,0].clip(lower=0)
            conf_int.iloc[:,1] = conf_int.iloc[:,1].clip(lower=0)
            
            # --- 4. Save Everything to Session State ---
            st.session_state['pred_mean'] = pred_mean
            st.session_state['conf_int'] = conf_int
            st.session_state['season'] = season_key
            st.session_state['monthly_data'] = monthly
            st.session_state['location_name'] = city_name_for_report # Save location name
            st.session_state['forecast_run_complete'] = True
            st.rerun() # Re-run the script to show the results page
            
else:
    # --- STEP 2: THE RESULTS DASHBOARD ---
    
    # Load all the data we saved
    pred_mean = st.session_state.get('pred_mean')
    conf_int = st.session_state.get('conf_int')
    season = st.session_state.get('season')
    monthly = st.session_state.get('monthly_data')
    location_name = st.session_state.get('location_name', 'N/A')
    
    st.header(f"Your Forecast & Recommendations")
    
    # --- 1. Show ALERTS First (Most Important) ---
    with st.container(border=True):
        st.subheader("⚠️ Forecast Alerts")
        forecast_period = pred_mean.head(3)
        ci_period = conf_int.head(3)
        alerts_found = 0
        
        lower_bound_low = ci_period.iloc[:, 0] < 15 
        if lower_bound_low.all(): 
            st.warning("**Potential Drought Alert:** The forecast suggests a high probability of significantly below-average rainfall for the next 3 months.")
            alerts_found += 1
        
        upper_bound_high = ci_period.iloc[:, 1] > 400 
        if upper_bound_high.any(): 
            st.error("**Potential High-Rainfall Alert:** The forecast includes the possibility of extreme rainfall (>400mm) in at least one of the next 3 months.")
            alerts_found += 1
        
        if alerts_found == 0:
            st.success("✅ **All Clear:** No immediate extreme weather alerts based on the forecast model.")

    # --- 2. Show CROP RECOMMENDATIONS Second ---
    with st.container(border=True):
        st.subheader("🌾 Crop Recommendations")
        future_sum = pred_mean.sum()
        crop, crop_msg = recommend_crop(future_sum, season) 
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Outlook", crop, help=crop_msg)
        with col2:
            st.metric(f"Forecasted Total (Next {len(pred_mean)} months)", f"{future_sum:.0f} mm")
        
        st.markdown("---")
        st.subheader("Month-by-Month Crop Suitability")
        month_crops_df = month_wise_crop_recommendation(pred_mean, season) 
        st.dataframe(month_crops_df)

    # --- 3. Show the FORECAST Chart Third ---
    with st.container(border=True):
        st.subheader("📈 12-Month Rainfall Forecast")
        st.info("The green line is the most likely forecast. The shaded area is the 'Likely Range' (95% Confidence Interval).")

        fig_fc = go.Figure()
        # Add historical data
        fig_fc.add_trace(go.Scatter(x=monthly.index, y=monthly['rainfall'].clip(lower=0), mode='lines', name='Historical Rainfall', line=dict(color='gray')))
        # Add forecast
        fig_fc.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values, mode='lines+markers', name='Forecasted Rainfall', line=dict(color='green')))
        # Add confidence interval
        ci_x = list(conf_int.index) + list(conf_int.index[::-1])
        ci_y = list(conf_int.iloc[:,1].values) + list(conf_int.iloc[:,0].values[::-1])
        ci_y = [max(0, v) for v in ci_y]
        fig_fc.add_trace(go.Scatter(x=ci_x, y=ci_y, fill='toself', fillcolor='rgba(0,128,0,0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False, name='Likely Range'))
        
        fig_fc.update_layout(title='Rainfall Forecast vs. History', xaxis_title='Date', yaxis_title='Rainfall (mm)')
        st.plotly_chart(fig_fc, use_container_width=True)

    # --- 4. Hide all other charts in an expander ---
    with st.expander("Show Detailed Historical Analysis"):
        st.subheader("Historical Rainfall Patterns")
        
        # Monthly time series
        series = monthly['rainfall'].clip(lower=0)
        rolling = series.rolling(window=12, min_periods=1).mean()
        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(x=series.index, y=series.values, name='Monthly Rainfall', marker_color='skyblue'))
        fig_month.add_trace(go.Scatter(x=rolling.index, y=rolling.values, mode='lines', name='12-Month Trend', line=dict(color='crimson', width=3)))
        fig_month.update_layout(title='Historical Monthly Rainfall and Long-Term Trend', xaxis_title='Date', yaxis_title='Rainfall (mm)')
        st.plotly_chart(fig_month, use_container_width=True)

        # Yearly totals
        st.subheader("Total Rainfall by Year")
        col = "rainfall"
        yearly = monthly.resample('YE').sum()
        yearly_vals = yearly[col].clip(lower=0)
        fig_y = go.Figure(go.Bar(x=yearly.index.year.astype(str), y=yearly_vals.values, marker_color='seagreen'))
        fig_y.update_layout(xaxis_title='Year', yaxis_title='Total Annual Rainfall (mm)')
        st.plotly_chart(fig_y, use_container_width=True)
        
    # --- Add a "Start Over" button ---
    if st.button("Start New Forecast (Clear Data)"):
        st.session_state.clear()
        st.rerun()

    # --- NEW DOWNLOAD BLOCK ---
    st.markdown("---")
    st.subheader("Download Your Forecast")
    
    # We need to re-create the report text
    report = f"Rainfall Prediction & Crop Recommendation\n"
    report += f"Location: {location_name}\n\n"
    report += "== FORECAST ALERTS ==\n"
    # (Re-run alert logic for text)
    forecast_period = pred_mean.head(3)
    ci_period = conf_int.head(3)
    alerts_found = 0
    lower_bound_low = ci_period.iloc[:, 0] < 15
    if lower_bound_low.all(): 
        report += "Potential Drought Alert: High probability of <15mm/month rainfall for the next 3 months.\n"
        alerts_found += 1
    upper_bound_high = ci_period.iloc[:, 1] > 400 
    if upper_bound_high.any(): 
        report += "Potential High-Rainfall Alert: Possibility of >400mm rainfall in at least one of the next 3 months.\n"
        alerts_found += 1
    if alerts_found == 0:
        report += "All Clear: No immediate extreme weather alerts detected.\n"
    
    report += "\n== OVERALL RECOMMENDATION ==\n"
    future_sum = pred_mean.sum() # Need to recalculate this
    crop, crop_msg = recommend_crop(future_sum, season)
    report += f"Crops: {crop}\n"
    report += f"Reason: {crop_msg}\n"
    report += f"Forecasted Total (Next 12 months): {future_sum:.2f} mm\n"
    
    report += "\n== MONTH-BY-MONTH SUITABILITY ==\n"
    month_crops_df = month_wise_crop_recommendation(pred_mean, season)
    report += month_crops_df.to_string()
    
    # Re-create the plot for the PDF
    fig_fc_report = go.Figure()
    fig_fc_report.add_trace(go.Scatter(x=monthly.index, y=monthly['rainfall'].clip(lower=0), mode='lines', name='Historical'))
    fig_fc_report.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean.values, mode='lines+markers', name='Forecast'))
    
    try:
        # Try to get the image bytes
        img_bytes, img_mime = _fig_to_bytes(fig_fc_report)
        images = [img_bytes] if img_mime == 'image/png' else []
    except Exception:
        images = []

    # Generate the report bytes
    pdf_bytes = generate_pdf_report(report, images=images)
    
    st.download_button(
        label="Download Full PDF Report" if REPORTLAB_AVAILABLE else "Download Full Report (.txt)",
        data=pdf_bytes,
        file_name="Rainfall_Forecast_Report.pdf" if REPORTLAB_AVAILABLE else "Rainfall_Forecast_Report.txt",
        mime="application/pdf" if REPORTLAB_AVAILABLE else "text/plain"
    )
    # --- END OF NEW BLOCK ---

st.warning("Note: Rainfall data is from NASA POWER. For official planning, use IMD or local sources.")
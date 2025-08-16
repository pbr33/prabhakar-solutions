import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple
import warnings

# Suppress statsmodels warnings
warnings.filterwarnings("ignore")

@st.cache_data
def detect_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """Detects anomalies in price and volume data using Isolation Forest."""
    if data.empty or len(data) < 20:
        return pd.DataFrame()
    
    features = data[['Close', 'Volume']].copy()
    features['Price_Change'] = features['Close'].pct_change().abs()
    features['Volume_Change'] = features['Volume'].pct_change().abs()
    features.dropna(inplace=True)

    if features.empty:
        return pd.DataFrame()

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(features)
    
    predictions = model.predict(features)
    features['anomaly'] = predictions
    
    return features[features['anomaly'] == -1]

@st.cache_data
def generate_forecast(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a 30-day price forecast using an ARIMA model."""
    if data.empty or len(data) < 50:
        return pd.DataFrame(), pd.DataFrame()
        
    close_prices = data['Close'].asfreq('D').fillna(method='ffill')
    
    try:
        model = ARIMA(close_prices, order=(5, 1, 0))
        fitted_model = model.fit()
        
        forecast_result = fitted_model.get_forecast(steps=30)
        forecast_df = forecast_result.summary_frame(alpha=0.05)
        
        history_df = close_prices.to_frame(name='Historical Price')
        plot_df = pd.concat([history_df, forecast_df['mean'].to_frame(name='Forecasted Price')])

        return plot_df, forecast_df
    except Exception as e:
        print(f"ARIMA forecast failed: {e}")
        return pd.DataFrame(), pd.DataFrame()


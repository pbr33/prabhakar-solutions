import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple
import warnings
import numpy as np

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

    try:
        model = IsolationForest(contamination=0.05, random_state=42)  # Increased contamination for more detections
        model.fit(features)
        
        predictions = model.predict(features)
        features['anomaly'] = predictions
        
        anomalies = features[features['anomaly'] == -1]
        return anomalies
    except Exception as e:
        st.warning(f"Anomaly detection failed: {e}")
        return pd.DataFrame()

@st.cache_data
def generate_forecast(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a 30-day price forecast using ARIMA model with enhanced error handling."""
    if data.empty or len(data) < 50:
        st.warning("Insufficient data for forecasting. Need at least 50 data points.")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        # Prepare the data
        close_prices = data['Close'].copy()
        
        # Handle missing values
        close_prices = close_prices.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure we have a proper datetime index
        if not isinstance(close_prices.index, pd.DatetimeIndex):
            close_prices.index = pd.to_datetime(close_prices.index)
        
        # Create daily frequency if not already
        close_prices = close_prices.asfreq('D', method='ffill')
        
        # Remove any remaining NaN values
        close_prices = close_prices.dropna()
        
        if len(close_prices) < 30:
            st.warning("Insufficient clean data for forecasting after preprocessing.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Try different ARIMA orders for better fit
        arima_orders = [(5,1,0), (3,1,1), (2,1,2), (1,1,1), (1,0,1)]
        best_model = None
        best_aic = float('inf')
        
        for order in arima_orders:
            try:
                model = ARIMA(close_prices, order=order)
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
                    
            except Exception:
                continue
        
        if best_model is None:
            st.error("Could not fit ARIMA model with any configuration.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Generate forecast
        forecast_steps = 30
        forecast_result = best_model.get_forecast(steps=forecast_steps)
        forecast_df = forecast_result.summary_frame(alpha=0.05)
        
        # Rename columns for clarity
        forecast_df.columns = ['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper']
        
        # Create forecast dates
        last_date = close_prices.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_df.index = forecast_dates
        
        # Combine historical and forecast data for plotting
        history_df = close_prices.to_frame(name='Historical Price')
        
        # Create a combined dataframe for plotting
        plot_df = pd.DataFrame(index=pd.date_range(start=close_prices.index[0], end=forecast_dates[-1], freq='D'))
        plot_df['Historical Price'] = history_df['Historical Price']
        plot_df['Forecasted Price'] = forecast_df['mean']
        
        # Add model performance metrics to forecast_df
        forecast_df['model_aic'] = best_aic
        forecast_df['model_order'] = str(best_model.model.order)
        
        return plot_df, forecast_df
        
    except Exception as e:
        st.error(f"ARIMA forecast failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

def generate_simple_forecast(data: pd.DataFrame, days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a simple trend-based forecast as fallback."""
    try:
        close_prices = data['Close'].tail(60)  # Use last 60 days
        
        # Calculate simple trend
        x = np.arange(len(close_prices))
        coeffs = np.polyfit(x, close_prices.values, 1)
        trend_slope = coeffs[0]
        
        # Generate forecast dates
        last_date = close_prices.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        # Create simple linear forecast
        last_price = close_prices.iloc[-1]
        forecast_values = []
        
        for i in range(days):
            forecast_price = last_price + (trend_slope * (i + 1))
            forecast_values.append(forecast_price)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'mean': forecast_values,
            'mean_se': [abs(last_price * 0.02)] * days,  # 2% standard error
            'mean_ci_lower': [val * 0.95 for val in forecast_values],
            'mean_ci_upper': [val * 1.05 for val in forecast_values]
        }, index=forecast_dates)
        
        # Create plot DataFrame
        history_df = close_prices.to_frame(name='Historical Price')
        plot_df = pd.DataFrame(index=pd.date_range(start=close_prices.index[0], end=forecast_dates[-1], freq='D'))
        plot_df['Historical Price'] = history_df['Historical Price']
        plot_df['Forecasted Price'] = forecast_df['mean']
        
        return plot_df, forecast_df
        
    except Exception as e:
        st.error(f"Simple forecast failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def calculate_forecast_metrics(historical_data: pd.DataFrame, forecast_data: pd.DataFrame) -> dict:
    """Calculate forecast accuracy metrics and summary statistics."""
    if historical_data.empty or forecast_data.empty:
        return {}
    
    try:
        # Calculate basic forecast metrics
        start_price = historical_data['Close'].iloc[-1]
        end_price = forecast_data['mean'].iloc[-1]
        forecast_return = ((end_price - start_price) / start_price) * 100
        
        # Calculate volatility metrics
        forecast_volatility = forecast_data['mean'].std()
        confidence_width = (forecast_data['mean_ci_upper'] - forecast_data['mean_ci_lower']).mean()
        
        # Trend analysis
        trend_direction = "Bullish" if end_price > start_price else "Bearish" if end_price < start_price else "Sideways"
        trend_strength = abs(forecast_return)
        
        return {
            'start_price': start_price,
            'end_price': end_price,
            'forecast_return_pct': forecast_return,
            'forecast_volatility': forecast_volatility,
            'confidence_width': confidence_width,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'forecast_days': len(forecast_data)
        }
        
    except Exception as e:
        st.warning(f"Could not calculate forecast metrics: {e}")
        return {}

def detect_price_patterns(data: pd.DataFrame) -> dict:
    """Detect basic price patterns for anomaly context."""
    if data.empty or len(data) < 20:
        return {}
    
    try:
        recent_data = data.tail(20)
        
        # Calculate pattern indicators
        price_trend = "Uptrend" if recent_data['Close'].iloc[-1] > recent_data['Close'].iloc[0] else "Downtrend"
        volatility = recent_data['Close'].std()
        avg_volume = recent_data['Volume'].mean()
        recent_volume = recent_data['Volume'].iloc[-1]
        volume_spike = recent_volume > (avg_volume * 1.5)
        
        # Price levels
        support_level = recent_data['Low'].min()
        resistance_level = recent_data['High'].max()
        current_price = recent_data['Close'].iloc[-1]
        
        # Position within range
        price_range = resistance_level - support_level
        if price_range > 0:
            position_in_range = (current_price - support_level) / price_range
        else:
            position_in_range = 0.5
        
        return {
            'trend': price_trend,
            'volatility': volatility,
            'volume_spike': volume_spike,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'position_in_range': position_in_range,
            'avg_volume': avg_volume,
            'recent_volume': recent_volume
        }
        
    except Exception as e:
        st.warning(f"Pattern detection failed: {e}")
        return {}

def generate_anomaly_report(anomalies: pd.DataFrame, patterns: dict) -> str:
    """Generate a text report about detected anomalies."""
    if anomalies.empty:
        return "No significant anomalies detected in recent trading activity."
    
    try:
        report = f"**Anomaly Detection Summary:**\n\n"
        report += f"- **Total Anomalies Detected:** {len(anomalies)}\n"
        report += f"- **Time Period:** {anomalies.index[0].strftime('%Y-%m-%d')} to {anomalies.index[-1].strftime('%Y-%m-%d')}\n\n"
        
        # Most recent anomaly
        latest_anomaly = anomalies.iloc[-1]
        report += f"**Most Recent Anomaly ({anomalies.index[-1].strftime('%Y-%m-%d')}):**\n"
        report += f"- Price: ${latest_anomaly['Close']:.2f}\n"
        report += f"- Volume: {latest_anomaly['Volume']:,.0f}\n"
        report += f"- Price Change: {latest_anomaly['Price_Change']:.1%}\n"
        report += f"- Volume Change: {latest_anomaly['Volume_Change']:.1%}\n\n"
        
        # Pattern context
        if patterns:
            report += f"**Market Context:**\n"
            report += f"- Current Trend: {patterns.get('trend', 'Unknown')}\n"
            report += f"- Volume Activity: {'High' if patterns.get('volume_spike', False) else 'Normal'}\n"
            report += f"- Price Position: {patterns.get('position_in_range', 0) * 100:.1f}% of recent range\n"
        
        return report
        
    except Exception as e:
        return f"Error generating anomaly report: {e}"
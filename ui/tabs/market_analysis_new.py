import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
import time

# Professional styling CSS
def apply_professional_styling():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .pattern-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .pattern-card:hover {
        transform: translateY(-5px);
    }
    
    .bullish-signal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    
    .bearish-signal {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
    
    .neutral-signal {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
    }
    
    .analysis-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #e9ecef;
    }
    
    .indicator-gauge {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .pattern-annotation {
        font-size: 12px;
        font-weight: bold;
        background: rgba(255,255,255,0.9);
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #ccc;
    }
    
    .real-time-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 3rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Real-time data fetcher using EODHD API
class EODHDDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
    
    def get_real_time_data(self, symbol):
        """Get real-time price data"""
        url = f"{self.base_url}/real-time/{symbol}?api_token={self.api_key}&fmt=json"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch real-time data: {e}")
            return {}
    
    def get_historical_data(self, symbol, period='1y'):
        """Get historical OHLCV data"""
        end_date = datetime.now()
        if period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '6m':
            start_date = end_date - timedelta(days=180)
        elif period == '3m':
            start_date = end_date - timedelta(days=90)
        elif period == '1m':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=365)
        
        url = f"{self.base_url}/eod/{symbol}"
        params = {
            'api_token': self.api_key,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'period': 'd',
            'fmt': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns to match expected format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted_close', 'Volume']
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            st.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol):
        """Get fundamental data"""
        url = f"{self.base_url}/fundamentals/{symbol}?api_token={self.api_key}&fmt=json"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {}

# Advanced technical analysis functions
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    if df.empty:
        return df
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def detect_candlestick_patterns(df):
    """Enhanced pattern detection with proper visualization coordinates"""
    patterns = []
    
    if len(df) < 5:
        return patterns
    
    # Get recent data for pattern detection
    recent_data = df.tail(100)
    
    for i in range(4, len(recent_data)):
        current = recent_data.iloc[i]
        prev1 = recent_data.iloc[i-1]
        prev2 = recent_data.iloc[i-2]
        prev3 = recent_data.iloc[i-3]
        prev4 = recent_data.iloc[i-4]
        
        date = current.name
        
        # Doji pattern
        body_size = abs(current['Close'] - current['Open'])
        candle_range = current['High'] - current['Low']
        if candle_range > 0 and body_size / candle_range < 0.1:
            patterns.append({
                'name': 'Doji',
                'type': 'Neutral',
                'date': date,
                'price': current['High'] * 1.01,  # Slightly above high for visibility
                'confidence': 75,
                'description': 'Indecision in the market, potential reversal signal'
            })
        
        # Hammer pattern
        lower_shadow = min(current['Open'], current['Close']) - current['Low']
        upper_shadow = current['High'] - max(current['Open'], current['Close'])
        if (candle_range > 0 and lower_shadow > 2 * body_size and 
            upper_shadow < 0.1 * candle_range and body_size > 0):
            patterns.append({
                'name': 'Hammer',
                'type': 'Bullish',
                'date': date,
                'price': current['High'] * 1.02,
                'confidence': 80,
                'description': 'Potential bullish reversal after downtrend'
            })
        
        # Shooting Star
        if (candle_range > 0 and upper_shadow > 2 * body_size and 
            lower_shadow < 0.1 * candle_range and body_size > 0):
            patterns.append({
                'name': 'Shooting Star',
                'type': 'Bearish',
                'date': date,
                'price': current['High'] * 1.02,
                'confidence': 80,
                'description': 'Potential bearish reversal after uptrend'
            })
        
        # Bullish Engulfing
        if (prev1['Close'] < prev1['Open'] and  # Previous red candle
            current['Close'] > current['Open'] and  # Current green candle
            current['Open'] < prev1['Close'] and  # Current opens below prev close
            current['Close'] > prev1['Open']):  # Current closes above prev open
            patterns.append({
                'name': 'Bullish Engulfing',
                'type': 'Bullish',
                'date': date,
                'price': current['High'] * 1.02,
                'confidence': 85,
                'description': 'Strong bullish reversal signal'
            })
        
        # Bearish Engulfing
        if (prev1['Close'] > prev1['Open'] and  # Previous green candle
            current['Close'] < current['Open'] and  # Current red candle
            current['Open'] > prev1['Close'] and  # Current opens above prev close
            current['Close'] < prev1['Open']):  # Current closes below prev open
            patterns.append({
                'name': 'Bearish Engulfing',
                'type': 'Bearish',
                'date': date,
                'price': current['High'] * 1.02,
                'confidence': 85,
                'description': 'Strong bearish reversal signal'
            })
    
    return sorted(patterns, key=lambda x: x['date'], reverse=True)[:10]

def create_professional_chart(df, symbol, patterns=None):
    """Create a professional-looking chart with clean pattern annotations"""
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f'{symbol} - Professional Technical Analysis',
            'RSI (14)',
            'MACD',
            'Volume'
        ),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing_line_color='#26C6DA',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#26C6DA',
            decreasing_fillcolor='#EF5350'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['SMA_20'],
                name='SMA 20',
                line=dict(color='#FF9800', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['SMA_50'],
                name='SMA 50',
                line=dict(color='#9C27B0', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                line=dict(color='rgba(128,128,128,0.3)', width=1),
                showlegend=False,
                name='BB Upper'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                line=dict(color='rgba(128,128,128,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False,
                name='BB Lower'
            ),
            row=1, col=1
        )
    
    # Add pattern annotations
    if patterns:
        for pattern in patterns[:8]:  # Limit to prevent overcrowding
            color_map = {
                'Bullish': '#4CAF50',
                'Bearish': '#F44336',
                'Neutral': '#FF9800'
            }
            
            symbol_map = {
                'Bullish': 'triangle-up',
                'Bearish': 'triangle-down',
                'Neutral': 'circle'
            }
            
            fig.add_trace(
                go.Scatter(
                    x=[pattern['date']],
                    y=[pattern['price']],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=color_map.get(pattern['type'], '#FF9800'),
                        symbol=symbol_map.get(pattern['type'], 'circle'),
                        line=dict(width=2, color='white')
                    ),
                    text=pattern['name'][:3],  # Abbreviated name
                    textposition='top center',
                    textfont=dict(size=9, color=color_map.get(pattern['type'], '#FF9800')),
                    name=f"{pattern['name']} ({pattern['confidence']}%)",
                    hovertemplate=(
                        f"<b>{pattern['name']}</b><br>"
                        f"Type: {pattern['type']}<br>"
                        f"Confidence: {pattern['confidence']}%<br>"
                        f"Date: {pattern['date'].strftime('%Y-%m-%d')}<br>"
                        f"{pattern['description']}<extra></extra>"
                    ),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # RSI subplot
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='#9C27B0', width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)
    
    # MACD subplot
    if all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='#2196F3', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color='#FF5722', width=2)
            ),
            row=3, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ),
            row=3, col=1
        )
    
    # Volume subplot
    volume_colors = ['#26C6DA' if close >= open else '#EF5350' 
                    for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=volume_colors,
            opacity=0.7
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{symbol} - Real-Time Technical Analysis",
            x=0.5,
            font=dict(size=24, color='#2C3E50')
        ),
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis4_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="RSI",
        yaxis3_title="MACD",
        yaxis4_title="Volume",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
        xaxis_rangeslider_visible=False
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_pattern_display_chart(patterns, df):
    """Create a separate, clean chart specifically for pattern visualization"""
    if not patterns:
        return None
    
    # Create a simple, clean chart focused on patterns
    fig = go.Figure()
    
    # Simple candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index[-50:],  # Last 50 days for clarity
            open=df['Open'].tail(50),
            high=df['High'].tail(50),
            low=df['Low'].tail(50),
            close=df['Close'].tail(50),
            name='Price',
            increasing_line_color='#4CAF50',
            decreasing_line_color='#F44336'
        )
    )
    
    # Add only pattern annotations
    for i, pattern in enumerate(patterns[:5]):  # Show only top 5 patterns
        color_map = {
            'Bullish': '#4CAF50',
            'Bearish': '#F44336', 
            'Neutral': '#FF9800'
        }
        
        symbol_map = {
            'Bullish': 'triangle-up',
            'Bearish': 'triangle-down',
            'Neutral': 'diamond'
        }
        
        fig.add_trace(
            go.Scatter(
                x=[pattern['date']],
                y=[pattern['price']],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=color_map.get(pattern['type'], '#FF9800'),
                    symbol=symbol_map.get(pattern['type'], 'circle'),
                    line=dict(width=3, color='white')
                ),
                text=f"{pattern['name']}<br>{pattern['confidence']}%",
                textposition='top center',
                textfont=dict(size=10, color=color_map.get(pattern['type'], '#FF9800')),
                name=pattern['name'],
                hovertemplate=(
                    f"<b>{pattern['name']}</b><br>"
                    f"Type: {pattern['type']}<br>"
                    f"Confidence: {pattern['confidence']}%<br>"
                    f"Date: {pattern['date'].strftime('%Y-%m-%d')}<br>"
                    f"{pattern['description']}<extra></extra>"
                )
            )
        )
    
    fig.update_layout(
        title="Detected Candlestick Patterns (Recent 50 Days)",
        height=500,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        xaxis_rangeslider_visible=False
    )
    
    return fig

def render():
    """Main render function for the enhanced market analysis tab"""
    
    # Apply professional styling
    apply_professional_styling()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ Professional Market Analysis Suite</h1>
        <p>Real-time data • Advanced patterns • Professional insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get configuration
    symbol = st.session_state.get('selected_symbol', 'AAPL.US')
    api_key = st.session_state.get('eodhd_api_key', '')
    
    if not api_key:
        st.error("⚠️ EODHD API key required for real-time data. Please configure in settings.")
        return
    
    # Real-time status indicator
    st.markdown('<div class="real-time-badge">🔴 LIVE DATA</div>', unsafe_allow_html=True)
    
    # Initialize data fetcher
    fetcher = EODHDDataFetcher(api_key)
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        period = st.selectbox("📅 Time Period", ['1m', '3m', '6m', '1y'], index=3)
    
    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=False)
    
    with col3:
        show_patterns = st.checkbox("🎯 Show Patterns", value=True)
    
    with col4:
        advanced_indicators = st.checkbox("📊 Advanced Indicators", value=True)
    
    # Load data
    with st.spinner("📡 Fetching real-time market data..."):
        # Get historical data
        historical_data = fetcher.get_historical_data(symbol, period)
        
        if historical_data.empty:
            st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol.")
            return
        
        # Get real-time data for current price
        real_time_data = fetcher.get_real_time_data(symbol)
        current_price = real_time_data.get('close', historical_data['Close'].iloc[-1])
        price_change = real_time_data.get('change', 0)
        price_change_pct = real_time_data.get('change_p', 0)
    
    # Current price display
    price_color = "green" if price_change >= 0 else "red"
    price_arrow = "⬆️" if price_change >= 0 else "⬇️"
    
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin:0; color: {price_color};">
            {symbol} ${current_price:.2f} 
            {price_arrow} {price_change:+.2f} ({price_change_pct:+.2f}%)
        </h2>
        <p style="margin:0; color: #666;">Last updated: {datetime.now().strftime('%H:%M:%S EST')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate technical indicators
    with st.spinner("🔬 Calculating technical indicators..."):
        enhanced_data = calculate_technical_indicators(historical_data)
    
    # Pattern detection
    patterns = []
    if show_patterns:
        with st.spinner("🎯 Detecting candlestick patterns..."):
            patterns = detect_candlestick_patterns(enhanced_data)
    
    # Main chart
    st.markdown("### 📈 Interactive Price Chart")
    main_chart = create_professional_chart(enhanced_data, symbol, patterns if show_patterns else None)
    st.plotly_chart(main_chart, use_container_width=True)
    
    # Pattern analysis section
    if patterns:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 🎯 Pattern Analysis Dashboard")
        
        # Pattern summary
        bullish_patterns = [p for p in patterns if p['type'] == 'Bullish']
        bearish_patterns = [p for p in patterns if p['type'] == 'Bearish']
        neutral_patterns = [p for p in patterns if p['type'] == 'Neutral']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card bullish-signal">
                <h3 style="color: #28a745; margin: 0;">🟢 Bullish Patterns</h3>
                <h2 style="margin: 0;">{len(bullish_patterns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card bearish-signal">
                <h3 style="color: #dc3545; margin: 0;">🔴 Bearish Patterns</h3>
                <h2 style="margin: 0;">{len(bearish_patterns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card neutral-signal">
                <h3 style="color: #ffc107; margin: 0;">🟡 Neutral Patterns</h3>
                <h2 style="margin: 0;">{len(neutral_patterns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed pattern cards
        st.markdown("#### 📋 Recent Pattern Detections")
        
        for pattern in patterns[:6]:  # Show top 6 patterns
            pattern_class = f"{pattern['type'].lower()}-signal"
            confidence_color = "#28a745" if pattern['confidence'] >= 80 else "#ffc107" if pattern['confidence'] >= 60 else "#dc3545"
            
            st.markdown(f"""
            <div class="pattern-card {pattern_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: #2C3E50;">{pattern['name']} ({pattern['type']})</h4>
                        <p style="margin: 5px 0; color: #666;">📅 {pattern['date'].strftime('%Y-%m-%d %H:%M')}</p>
                        <p style="margin: 5px 0; color: #666;">{pattern['description']}</p>
                    </div>
                    <div style="text-align: center;">
                        <div style="background: {confidence_color}; color: white; padding: 8px 12px; border-radius: 20px; font-weight: bold;">
                            {pattern['confidence']}%
                        </div>
                        <small style="color: #666;">Confidence</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clean pattern visualization chart
        if st.button("🔍 Show Clean Pattern Chart", key="show_pattern_chart"):
            pattern_chart = create_pattern_display_chart(patterns, enhanced_data)
            if pattern_chart:
                st.plotly_chart(pattern_chart, use_container_width=True)
    
    # Technical indicators dashboard
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📊 Technical Indicators Dashboard")
    
    if not enhanced_data.empty:
        latest = enhanced_data.iloc[-1]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = latest.get('RSI', 0)
            rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            rsi_color = "#28a745" if rsi < 30 else "#dc3545" if rsi > 70 else "#ffc107"
            
            st.markdown(f"""
            <div class="indicator-gauge">
                <h4 style="margin: 0; color: #2C3E50;">RSI (14)</h4>
                <h2 style="margin: 5px 0; color: {rsi_color};">{rsi:.1f}</h2>
                <p style="margin: 0; color: {rsi_color}; font-weight: bold;">{rsi_signal}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            macd_color = "#28a745" if macd > macd_signal else "#dc3545"
            
            st.markdown(f"""
            <div class="indicator-gauge">
                <h4 style="margin: 0; color: #2C3E50;">MACD</h4>
                <h2 style="margin: 5px 0; color: {macd_color};">{macd:.3f}</h2>
                <p style="margin: 0; color: {macd_color}; font-weight: bold;">{macd_trend}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            bb_position = "Above Upper" if latest['Close'] > latest.get('BB_Upper', 0) else \
                         "Below Lower" if latest['Close'] < latest.get('BB_Lower', 0) else "Within Bands"
            bb_color = "#dc3545" if bb_position == "Above Upper" else \
                      "#28a745" if bb_position == "Below Lower" else "#ffc107"
            
            st.markdown(f"""
            <div class="indicator-gauge">
                <h4 style="margin: 0; color: #2C3E50;">Bollinger Bands</h4>
                <h3 style="margin: 5px 0; color: {bb_color}; font-size: 16px;">{bb_position}</h3>
                <p style="margin: 0; color: #666;">Position</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            atr = latest.get('ATR', 0)
            volatility_level = "High" if atr > enhanced_data['ATR'].quantile(0.8) else \
                              "Low" if atr < enhanced_data['ATR'].quantile(0.2) else "Medium"
            vol_color = "#dc3545" if volatility_level == "High" else \
                       "#28a745" if volatility_level == "Low" else "#ffc107"
            
            st.markdown(f"""
            <div class="indicator-gauge">
                <h4 style="margin: 0; color: #2C3E50;">Volatility (ATR)</h4>
                <h2 style="margin: 5px 0; color: {vol_color};">${atr:.2f}</h2>
                <p style="margin: 0; color: {vol_color}; font-weight: bold;">{volatility_level}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Market Summary Section
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📈 Market Summary & Insights")
    
    # Generate market insights
    total_patterns = len(patterns)
    bullish_count = len([p for p in patterns if p['type'] == 'Bullish'])
    bearish_count = len([p for p in patterns if p['type'] == 'Bearish'])
    
    # Determine overall sentiment
    if bullish_count > bearish_count:
        overall_sentiment = "Bullish"
        sentiment_color = "#28a745"
        sentiment_icon = "📈"
    elif bearish_count > bullish_count:
        overall_sentiment = "Bearish"
        sentiment_color = "#dc3545"
        sentiment_icon = "📉"
    else:
        overall_sentiment = "Neutral"
        sentiment_color = "#ffc107"
        sentiment_icon = "➡️"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="analysis-section">
            <h3 style="color: #2C3E50; margin-top: 0;">🎯 Technical Analysis Summary</h3>
            <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {sentiment_color};">
                <h4 style="color: {sentiment_color}; margin: 0;">
                    {sentiment_icon} Overall Sentiment: {overall_sentiment}
                </h4>
                <ul style="margin: 1rem 0; color: #2C3E50;">
                    <li><strong>Pattern Analysis:</strong> {total_patterns} patterns detected ({bullish_count} bullish, {bearish_count} bearish)</li>
                    <li><strong>RSI Signal:</strong> {rsi_signal} at {rsi:.1f}</li>
                    <li><strong>MACD Trend:</strong> {macd_trend}</li>
                    <li><strong>Volatility:</strong> {volatility_level} (ATR: ${atr:.2f})</li>
                    <li><strong>Price vs Bollinger Bands:</strong> {bb_position}</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Quick actions
        st.markdown("""
        <div class="analysis-section">
            <h4 style="color: #2C3E50; margin-top: 0;">⚡ Quick Actions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Export Data", key="export_data"):
            # Prepare export data
            export_data = enhanced_data.tail(100).round(4)
            csv_data = export_data.to_csv()
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv'
            )
        
        if st.button("🔔 Set Alert", key="set_alert"):
            st.info("Alert functionality coming soon!")
        
        if st.button("📱 Share Analysis", key="share_analysis"):
            st.info("Share functionality coming soon!")
    
    # Fundamental data section (if available)
    fundamental_data = fetcher.get_fundamental_data(symbol)
    if fundamental_data:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### 📊 Fundamental Overview")
        
        try:
            highlights = fundamental_data.get('Highlights', {})
            valuation = fundamental_data.get('Valuation', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                market_cap = highlights.get('MarketCapitalization', 'N/A')
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #2C3E50;">Market Cap</h4>
                    <h3 style="margin: 5px 0; color: #667eea;">{market_cap}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                pe_ratio = highlights.get('PERatio', 'N/A')
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #2C3E50;">P/E Ratio</h4>
                    <h3 style="margin: 5px 0; color: #667eea;">{pe_ratio}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                dividend_yield = highlights.get('DividendYield', 'N/A')
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #2C3E50;">Dividend Yield</h4>
                    <h3 style="margin: 5px 0; color: #667eea;">{dividend_yield}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                week_52_high = highlights.get('52WeekHigh', 'N/A')
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #2C3E50;">52W High</h4>
                    <h3 style="margin: 5px 0; color: #667eea;">{week_52_high}</h3>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning("Fundamental data structure differs from expected format.")
    
    # Auto refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666; border-top: 1px solid #eee; margin-top: 3rem;">
        <p>Powered by EODHD API • Real-time market data • Professional analysis</p>
        <small>Data provided for informational purposes only. Not investment advice.</small>
    </div>
    """, unsafe_allow_html=True)
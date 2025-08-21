import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

from config import get_config

def create_signal_visual(signal_data):
    """Create a visual representation of the AI signal instead of JSON."""
    if not signal_data:
        return
    
    # Parse JSON if it's a string
    if isinstance(signal_data, str):
        try:
            # Extract JSON from the response
            json_start = signal_data.find('{')
            json_end = signal_data.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = signal_data[json_start:json_end]
                parsed_data = json.loads(json_str)
            else:
                # Fallback if no JSON found
                parsed_data = {
                    "signal": "Neutral",
                    "confidence": 50,
                    "time_horizon": "Medium",
                    "key_catalyst": "Technical Analysis",
                    "risk_level": "Medium"
                }
        except:
            parsed_data = {
                "signal": "Neutral",
                "confidence": 50,
                "time_horizon": "Medium", 
                "key_catalyst": "Analysis Complete",
                "risk_level": "Medium"
            }
    else:
        parsed_data = signal_data
    
    # Create visual signal dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal = parsed_data.get('signal', 'Neutral')
        confidence = parsed_data.get('confidence', 50)
        
        # Signal indicator
        if signal.lower() == 'bullish':
            signal_color = "🟢"
            signal_bg = "#d4edda"
        elif signal.lower() == 'bearish':
            signal_color = "🔴"
            signal_bg = "#f8d7da"
        else:
            signal_color = "🟡"
            signal_bg = "#fff3cd"
        
        st.markdown(f"""
        <div style="background-color: {signal_bg}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2>{signal_color} {signal.upper()}</h2>
            <h3>Confidence: {confidence}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        time_horizon = parsed_data.get('time_horizon', 'Medium')
        risk_level = parsed_data.get('risk_level', 'Medium')
        
        st.markdown(f"""
        <div style="background-color: #e9ecef; padding: 20px; border-radius: 10px;">
            <h4>📅 Time Horizon</h4>
            <p><strong>{time_horizon}</strong></p>
            <h4>⚠️ Risk Level</h4>
            <p><strong>{risk_level}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        key_catalyst = parsed_data.get('key_catalyst', 'Technical Analysis')
        
        st.markdown(f"""
        <div style="background-color: #e9ecef; padding: 20px; border-radius: 10px;">
            <h4>🎯 Key Catalyst</h4>
            <p><strong>{key_catalyst}</strong></p>
        </div>
        """, unsafe_allow_html=True)

def add_pattern_annotations_to_chart(fig, patterns, data):
    """Add pattern annotations directly to the candlestick chart."""
    if not patterns:
        return fig
    
    for pattern in patterns[:5]:  # Show top 5 patterns
        pattern_date = pattern['date']
        
        # Find the price at pattern date
        try:
            if pattern_date in data.index:
                pattern_price = data.loc[pattern_date, 'High']
                
                # Choose marker based on pattern type
                if pattern['type'] == 'Bullish':
                    marker_color = 'green'
                    marker_symbol = 'triangle-up'
                elif pattern['type'] == 'Bearish':
                    marker_color = 'red'
                    marker_symbol = 'triangle-down'
                else:
                    marker_color = 'orange'
                    marker_symbol = 'circle'
                
                # Add pattern marker
                fig.add_trace(
                    go.Scatter(
                        x=[pattern_date],
                        y=[pattern_price + (pattern_price * 0.02)],  # Slightly above the high
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=marker_color,
                            symbol=marker_symbol,
                            line=dict(width=2, color='white')
                        ),
                        text=pattern['name'],
                        textposition='top center',
                        textfont=dict(size=10, color=marker_color),
                        name=f"{pattern['name']} ({pattern['type']})",
                        hovertemplate=f"<b>{pattern['name']}</b><br>" +
                                    f"Type: {pattern['type']}<br>" +
                                    f"Date: {pattern_date.strftime('%Y-%m-%d')}<br>" +
                                    f"Confidence: {pattern.get('confidence', 'N/A')}%<br>" +
                                    f"Description: {pattern['description']}<extra></extra>"
                    ),
                    row=1, col=1
                )
        except Exception as e:
            continue
    
    return fig

def render():
    """Renders the enhanced Market Analysis tab with all missing features."""
    st.markdown("## 📊 Advanced Market Analysis & Pattern Recognition")
    cfg = get_config()
    symbol = cfg['selected_symbol']
    llm = cfg['llm']
    api_key = cfg['eodhd_api_key']

    # Import functions locally to avoid circular imports
    try:
        from services.data_fetcher import get_market_data_yfinance
        from analysis.technical import (
            detect_real_candlestick_patterns, 
            calculate_advanced_indicators,
            calculate_fibonacci_levels,
            get_fundamental_data,
            get_sentiment_data,
            generate_comprehensive_analysis_report
        )
        from analysis.predictive import detect_anomalies, generate_forecast
        from analysis.reporting import generate_forecast_analysis_report
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.error("Please ensure all required modules are properly installed and configured.")
        return

    # Load market data
    with st.spinner("Loading market data..."):
        market_data = get_market_data_yfinance(symbol)

    if market_data.empty:
        st.error(f"Could not load market data for {symbol}. Please check the symbol or try again.")
        return

    # Calculate advanced indicators
    with st.spinner("Calculating advanced technical indicators..."):
        try:
            enhanced_data = calculate_advanced_indicators(market_data)
            fibonacci_levels = calculate_fibonacci_levels(enhanced_data)
        except Exception as e:
            st.warning(f"Error calculating advanced indicators: {e}")
            enhanced_data = market_data
            fibonacci_levels = {}

    # --- Main Candlestick Chart with Enhanced Features ---
    st.markdown("### 📈 Interactive Candlestick Chart with Pattern Detection")
    
    # Create subplot for main chart and volume
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price Action with Patterns', 'Technical Indicators', 'Volume'),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=enhanced_data.index,
            open=enhanced_data['Open'],
            high=enhanced_data['High'],
            low=enhanced_data['Low'],
            close=enhanced_data['Close'],
            name=symbol,
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # Add moving averages if available
    if 'EMA_20' in enhanced_data.columns:
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['EMA_20'], 
                      name='EMA 20', line=dict(color='orange', width=2)),
            row=1, col=1
        )
    
    if 'SMA_50' in enhanced_data.columns:
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['SMA_50'], 
                      name='SMA 50', line=dict(color='blue', width=2)),
            row=1, col=1
        )

    # Add Bollinger Bands if available
    if all(col in enhanced_data.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['BBU_20_2.0'], 
                      name='BB Upper', line=dict(color='gray', dash='dash', width=1),
                      fill=None),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['BBL_20_2.0'], 
                      name='BB Lower', line=dict(color='gray', dash='dash', width=1),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )

    # Add Fibonacci levels if available
    if fibonacci_levels:
        current_price = enhanced_data['Close'].iloc[-1]
        for level_name, level_price in fibonacci_levels.items():
            if abs(level_price - current_price) / current_price < 0.3:  # Show relevant levels
                fig.add_hline(
                    y=level_price, 
                    line_dash="dot", 
                    line_color="purple",
                    annotation_text=f"Fib {level_name}: ${level_price:.2f}",
                    row=1, col=1
                )

    # Add RSI to second subplot
    if 'RSI_14' in enhanced_data.columns:
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['RSI_14'], 
                      name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", row=2, col=1)

    # Volume chart with color coding
    colors = ['green' if close >= open else 'red' 
              for close, open in zip(enhanced_data['Close'], enhanced_data['Open'])]
    
    fig.add_trace(
        go.Bar(x=enhanced_data.index, y=enhanced_data['Volume'], 
               name='Volume', marker_color=colors, opacity=0.6),
        row=3, col=1
    )

    # Auto-detect and display patterns on chart
    if st.button("🔍 Detect & Show Patterns on Chart", key="auto_detect_patterns"):
        with st.spinner("Detecting candlestick patterns..."):
            try:
                patterns = detect_real_candlestick_patterns(enhanced_data)
                st.session_state.detected_patterns = patterns
                
                if patterns:
                    # Add pattern annotations to chart
                    fig = add_pattern_annotations_to_chart(fig, patterns, enhanced_data)
                    st.success(f"✅ Detected {len(patterns)} patterns and added to chart!")
                else:
                    st.info("No significant patterns detected in recent price action.")
            except Exception as e:
                st.error(f"Error detecting patterns: {e}")

    # Update layout
    fig.update_layout(
        title=f"{symbol} - Advanced Technical Analysis with Real-Time Indicators",
        height=900,
        xaxis3_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="RSI",
        yaxis3_title="Volume",
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Pattern Detection Section ---
    st.markdown("---")
    st.markdown("### 🎯 Real-Time Pattern Detection & Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🔍 Detected Candlestick Patterns")
        if st.button("🚀 Analyze Patterns", help="Detect and analyze candlestick patterns"):
            with st.spinner("Analyzing candlestick patterns..."):
                try:
                    patterns = detect_real_candlestick_patterns(enhanced_data)
                    st.session_state.detected_patterns = patterns
                    
                    if patterns:
                        # Display patterns in cards
                        for i, pattern in enumerate(patterns[:4]):  # Show top 4 patterns
                            pattern_color = "🟢" if pattern['type'] == 'Bullish' else "🔴" if pattern['type'] == 'Bearish' else "⚪"
                            
                            with st.container():
                                st.markdown(f"""
                                <div style="border: 2px solid {'green' if pattern['type'] == 'Bullish' else 'red' if pattern['type'] == 'Bearish' else 'orange'}; 
                                            border-radius: 10px; padding: 15px; margin: 10px 0; 
                                            background-color: {'#d4edda' if pattern['type'] == 'Bullish' else '#f8d7da' if pattern['type'] == 'Bearish' else '#fff3cd'};">
                                    <h4>{pattern_color} {pattern['name']} ({pattern['type']})</h4>
                                    <p><strong>📅 Date:</strong> {pattern['date'].strftime('%Y-%m-%d')}</p>
                                    <p><strong>🎯 Confidence:</strong> {pattern.get('confidence', 'N/A')}%</p>
                                    <p><strong>💡 Description:</strong> {pattern['description']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No significant patterns detected in recent price action.")
                except Exception as e:
                    st.error(f"Error detecting patterns: {e}")

    with col2:
        st.markdown("#### 📊 Technical Indicator Dashboard")
        if not enhanced_data.empty:
            latest = enhanced_data.iloc[-1]
            
            # RSI Gauge
            rsi = latest.get('RSI_14', 0)
            if rsi > 0:
                fig_rsi = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = rsi,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "RSI (14)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70}
                    }
                ))
                fig_rsi.update_layout(height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Key metrics
            col2a, col2b = st.columns(2)
            with col2a:
                macd = latest.get('MACD_12_26_9', 0)
                macd_signal = latest.get('MACDs_12_26_9', 0)
                if macd != 0 and macd_signal != 0:
                    macd_trend = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
                    st.metric("MACD Signal", macd_trend, f"{macd:.3f}")
                
            with col2b:
                if 'ATR_14' in enhanced_data.columns:
                    atr = latest.get('ATR_14', 0)
                    current_price = latest.get('Close', 1)
                    if atr > 0 and current_price > 0:
                        vol_pct = (atr / current_price) * 100
                        st.metric("Volatility (ATR)", f"{vol_pct:.1f}%", f"${atr:.2f}")

    # --- AI Analysis Section ---
    st.markdown("---")
    st.markdown("### 🧠 AI-Powered Comprehensive Analysis")
    
    if st.button("🎯 Generate AI Analysis", disabled=(llm is None), 
                help="Generate comprehensive AI-powered market analysis"):
        
        with st.spinner("🤖 AI is analyzing market data... This may take a moment."):
            try:
                # Get additional data
                fundamental_data = get_fundamental_data(symbol, api_key) if api_key else {}
                sentiment_data = get_sentiment_data(symbol, api_key) if api_key else {}
                patterns = st.session_state.get('detected_patterns', [])
                
                # Generate comprehensive report
                comprehensive_report = generate_comprehensive_analysis_report(
                    llm, symbol, patterns, enhanced_data, fundamental_data, 
                    sentiment_data, fibonacci_levels
                )
                
                st.session_state.comprehensive_analysis = comprehensive_report
                
                # Create visual signal representation
                st.markdown("#### 🎯 AI Signal Dashboard")
                create_signal_visual(comprehensive_report)
                
            except Exception as e:
                st.error(f"Error generating comprehensive analysis: {e}")
    
    # Display full analysis
    if 'comprehensive_analysis' in st.session_state:
        with st.expander("📋 Full AI Analysis Report", expanded=False):
            st.markdown(st.session_state.comprehensive_analysis)

    # --- Predictive Analytics Section ---
    st.markdown("---")
    st.markdown("### 🔮 Predictive Analytics & Forecasting")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 📈 30-Day Price Forecast")
        if st.button("🔮 Generate ARIMA Forecast", key="generate_forecast"):
            with st.spinner("Building ARIMA model and generating forecast..."):
                try:
                    plot_df, forecast_df = generate_forecast(enhanced_data)
                    st.session_state.forecast_plot_df = plot_df
                    st.session_state.forecast_df = forecast_df
                    
                    if not forecast_df.empty:
                        st.success("✅ Forecast generated successfully!")
                    else:
                        st.warning("⚠️ Unable to generate forecast with current data.")
                        
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")

        # Display forecast chart and table
        if 'forecast_plot_df' in st.session_state and not st.session_state.forecast_plot_df.empty:
            # Create forecast chart
            fig_forecast = go.Figure()
            
            # Historical data
            historical_data = st.session_state.forecast_plot_df.dropna(subset=['Historical Price'])
            if not historical_data.empty:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=historical_data.index, 
                        y=historical_data['Historical Price'], 
                        name='Historical Price',
                        line=dict(color='blue', width=2)
                    )
                )
            
            # Forecast data
            forecast_data = st.session_state.forecast_plot_df.dropna(subset=['Forecasted Price'])
            if not forecast_data.empty:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_data.index, 
                        y=forecast_data['Forecasted Price'], 
                        name='30-Day Forecast',
                        line=dict(color='red', dash='dot', width=3)
                    )
                )
            
            # Add confidence intervals if available
            if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
                forecast_df = st.session_state.forecast_df
                if 'mean_ci_upper' in forecast_df.columns and 'mean_ci_lower' in forecast_df.columns:
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['mean_ci_upper'],
                            name='Upper Confidence (95%)',
                            line=dict(color='lightgray', dash='dash'),
                            showlegend=False
                        )
                    )
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['mean_ci_lower'],
                            name='Lower Confidence (95%)',
                            line=dict(color='lightgray', dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(128,128,128,0.2)',
                            showlegend=False
                        )
                    )

            fig_forecast.update_layout(
                title=f"{symbol} - 30 Day ARIMA Forecast",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Show forecast table with download option
            if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
                st.markdown("#### 📊 Forecast Data Table")
                
                # Format the forecast table for better display
                display_forecast = st.session_state.forecast_df.copy()
                display_forecast.index = display_forecast.index.strftime('%Y-%m-%d')
                display_forecast = display_forecast.round(2)
                
                st.dataframe(display_forecast, use_container_width=True)
                
                # Download button for forecast data
                csv_forecast = display_forecast.to_csv()
                st.download_button(
                    label="📥 Download Forecast Data (CSV)",
                    data=csv_forecast,
                    file_name=f"{symbol}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv',
                    key="download_forecast"
                )
            
            # Generate forecast analysis
            if llm and st.button("📊 Analyze Forecast with AI", key="analyze_forecast"):
                with st.spinner("Generating AI forecast analysis..."):
                    try:
                        forecast_analysis = generate_forecast_analysis_report(
                            llm, symbol, st.session_state.forecast_df
                        )
                        st.markdown("#### 🤖 AI Forecast Analysis")
                        st.markdown(forecast_analysis)
                    except Exception as e:
                        st.error(f"Error generating forecast analysis: {e}")

    with col4:
        st.markdown("#### ⚠️ Anomaly Detection")
        if st.button("🔍 Detect Price/Volume Anomalies", key="detect_anomalies"):
            with st.spinner("Analyzing for market anomalies..."):
                try:
                    anomalies = detect_anomalies(enhanced_data)
                    st.session_state.anomalies = anomalies
                    
                    if not anomalies.empty:
                        st.success(f"✅ Detected {len(anomalies)} anomalies")
                    else:
                        st.info("No significant anomalies detected.")
                        
                except Exception as e:
                    st.error(f"Error detecting anomalies: {e}")

        if 'anomalies' in st.session_state and not st.session_state.anomalies.empty:
            st.markdown(f"**🚨 {len(st.session_state.anomalies)} anomalies detected:**")
            
            # Show recent anomalies in cards
            recent_anomalies = st.session_state.anomalies.tail(3)
            for idx, row in recent_anomalies.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="border: 2px solid orange; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #fff3cd;">
                        <h5>📅 {idx.strftime('%Y-%m-%d')}</h5>
                        <p><strong>💰 Price:</strong> ${row['Close']:.2f}</p>
                        <p><strong>📊 Volume:</strong> {row['Volume']:,.0f}</p>
                        <p><strong>⚡ Price Change:</strong> {row['Price_Change']:.1%}</p>
                        <p><strong>📈 Volume Change:</strong> {row['Volume_Change']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # --- Export Section ---
    st.markdown("---")
    st.markdown("### 📤 Export Analysis & Data")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        if st.button("📊 Export Technical Data", key="export_tech_data"):
            # Prepare comprehensive export data
            export_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            export_data = enhanced_data.tail(100)[export_columns].copy()
                
            # Add technical indicators if available
            indicator_cols = ['RSI_14', 'MACD_12_26_9', 'EMA_20', 'SMA_50', 'BBU_20_2.0', 'BBL_20_2.0', 'ATR_14']
            for col in indicator_cols:
                if col in enhanced_data.columns:
                    export_data[col] = enhanced_data[col].tail(100)
            
            csv_data = export_data.to_csv()
            st.download_button(
                label="💾 Download Technical Data (CSV)",
                data=csv_data,
                file_name=f"{symbol}_technical_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                key="download_tech_data"
            )
    
    with col6:
        if 'comprehensive_analysis' in st.session_state:
            if st.button("📄 Export AI Analysis Report", key="export_ai_report"):
                report_text = st.session_state.comprehensive_analysis
                st.download_button(
                    label="📋 Download AI Report (TXT)",
                    data=report_text,
                    file_name=f"{symbol}_ai_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime='text/plain',
                    key="download_ai_report"
                )
    
    with col7:
        if 'detected_patterns' in st.session_state:
            if st.button("🎯 Export Pattern Data", key="export_pattern_data"):
                patterns_df = pd.DataFrame(st.session_state.detected_patterns)
                if not patterns_df.empty:
                    patterns_csv = patterns_df.to_csv(index=False)
                    st.download_button(
                        label="📈 Download Patterns (CSV)",
                        data=patterns_csv,
                        file_name=f"{symbol}_patterns_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv',
                        key="download_patterns"
                    )
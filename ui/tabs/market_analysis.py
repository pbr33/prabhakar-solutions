import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from config import get_config

def render():
    """Renders the enhanced Market Analysis tab with real pattern detection."""
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
    st.markdown("### 📈 Interactive Candlestick Chart with Technical Overlays")
    
    # Create subplot for main chart and volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Price Action', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=enhanced_data.index,
            open=enhanced_data['Open'],
            high=enhanced_data['High'],
            low=enhanced_data['Low'],
            close=enhanced_data['Close'],
            name=symbol
        ),
        row=1, col=1
    )

    # Add moving averages if available
    if 'EMA_20' in enhanced_data.columns:
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['EMA_20'], 
                      name='EMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'SMA_50' in enhanced_data.columns:
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['SMA_50'], 
                      name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )

    # Add Bollinger Bands if available
    if all(col in enhanced_data.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['BBU_20_2.0'], 
                      name='BB Upper', line=dict(color='gray', dash='dash', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=enhanced_data.index, y=enhanced_data['BBL_20_2.0'], 
                      name='BB Lower', line=dict(color='gray', dash='dash', width=1)),
            row=1, col=1
        )

    # Add Fibonacci levels if available
    if fibonacci_levels:
        current_price = enhanced_data['Close'].iloc[-1]
        for level_name, level_price in fibonacci_levels.items():
            if abs(level_price - current_price) / current_price < 0.2:  # Only show nearby levels
                fig.add_hline(
                    y=level_price, 
                    line_dash="dot", 
                    line_color="purple",
                    annotation_text=f"Fib {level_name}: ${level_price:.2f}",
                    row=1, col=1
                )

    # Volume chart
    colors = ['red' if close < open else 'green' 
              for close, open in zip(enhanced_data['Close'], enhanced_data['Open'])]
    
    fig.add_trace(
        go.Bar(x=enhanced_data.index, y=enhanced_data['Volume'], 
               name='Volume', marker_color=colors),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{symbol} - Advanced Technical Analysis",
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Pattern Detection and Analysis Section ---
    st.markdown("---")
    st.markdown("### 🔍 Real-Time Pattern Detection & Multi-Factor Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🎯 Detected Candlestick Patterns")
        if st.button("🚀 Detect Real Patterns", help="Uses TA-Lib for accurate pattern detection"):
            with st.spinner("Analyzing candlestick patterns..."):
                try:
                    patterns = detect_real_candlestick_patterns(enhanced_data)
                    st.session_state.detected_patterns = patterns
                    
                    if patterns:
                        # Display patterns in a nice format
                        for pattern in patterns[:5]:  # Show top 5 patterns
                            pattern_color = "🟢" if pattern['type'] == 'Bullish' else "🔴" if pattern['type'] == 'Bearish' else "⚪"
                            st.markdown(f"""
                            **{pattern_color} {pattern['name']}** ({pattern['type']})  
                            📅 Date: {pattern['date'].strftime('%Y-%m-%d')}  
                            💡 {pattern['description']}  
                            🎯 Confidence: {pattern.get('confidence', 'N/A')}%
                            """)
                            st.markdown("---")
                    else:
                        st.info("No significant patterns detected in recent price action.")
                except Exception as e:
                    st.error(f"Error detecting patterns: {e}")
                    st.info("Pattern detection failed. Please ensure TA-Lib is installed correctly.")

    with col2:
        st.markdown("#### 📊 Technical Indicator Summary")
        if not enhanced_data.empty:
            latest = enhanced_data.iloc[-1]
            
            # Create indicator dashboard
            col2a, col2b = st.columns(2)
            
            with col2a:
                # RSI with color coding
                rsi = latest.get('RSI_14', 0)
                if rsi > 0:
                    rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                    st.metric("RSI (14)", f"{rsi:.1f}", help="Relative Strength Index")
                    st.markdown(f"{rsi_color} {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}")
                else:
                    st.metric("RSI (14)", "N/A")
                
                # MACD
                macd = latest.get('MACD_12_26_9', 0)
                macd_signal = latest.get('MACDs_12_26_9', 0)
                if macd != 0 and macd_signal != 0:
                    macd_trend = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
                    st.metric("MACD", f"{macd:.3f}")
                    st.markdown(macd_trend)
                else:
                    st.metric("MACD", "N/A")
            
            with col2b:
                # Bollinger Band position
                if all(col in enhanced_data.columns for col in ['BBU_20_2.0', 'BBL_20_2.0']):
                    bb_upper = latest.get('BBU_20_2.0', 0)
                    bb_lower = latest.get('BBL_20_2.0', 0)
                    current_close = latest.get('Close', 0)
                    
                    if bb_upper > bb_lower > 0 and current_close > 0:
                        bb_pos = ((current_close - bb_lower) / (bb_upper - bb_lower) * 100)
                        bb_color = "🔴" if bb_pos > 80 else "🟢" if bb_pos < 20 else "🟡"
                        st.metric("BB Position", f"{bb_pos:.1f}%")
                        st.markdown(f"{bb_color} {'Upper band' if bb_pos > 80 else 'Lower band' if bb_pos < 20 else 'Middle range'}")
                    else:
                        st.metric("BB Position", "N/A")
                else:
                    st.metric("BB Position", "N/A")
                
                # Volume trend
                if 'OBV' in enhanced_data.columns and len(enhanced_data) >= 5:
                    obv_current = latest.get('OBV', 0)
                    obv_past = enhanced_data['OBV'].iloc[-5] if len(enhanced_data) >= 5 else obv_current
                    
                    if obv_past != 0:
                        obv_change = ((obv_current - obv_past) / abs(obv_past) * 100)
                        obv_trend = "🟢 Accumulation" if obv_change > 0 else "🔴 Distribution"
                        st.metric("OBV Trend", f"{obv_change:.1f}%")
                        st.markdown(obv_trend)
                    else:
                        st.metric("OBV Trend", "N/A")
                else:
                    st.metric("OBV Trend", "N/A")

    # --- Comprehensive AI Analysis ---
    st.markdown("---")
    st.markdown("### 🧠 AI-Powered Comprehensive Analysis")
    
    if st.button("🎯 Generate Complete Analysis", disabled=(llm is None), 
                help="Combines technical, fundamental, and sentiment analysis"):
        
        with st.spinner("Gathering comprehensive market intelligence..."):
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
            except Exception as e:
                st.error(f"Error generating comprehensive analysis: {e}")
    
    # Display comprehensive analysis
    if 'comprehensive_analysis' in st.session_state:
        st.markdown("#### 📋 Complete Market Intelligence Report")
        st.markdown(st.session_state.comprehensive_analysis)

    # --- Predictive Analytics Section ---
    st.markdown("---")
    st.markdown("### 🔮 Predictive Analytics & Anomaly Detection")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🎯 30-Day Price Forecast")
        if st.button("Generate ARIMA Forecast"):
            with st.spinner("Building ARIMA model and generating forecast..."):
                try:
                    plot_df, forecast_df = generate_forecast(enhanced_data)
                    st.session_state.forecast_plot_df = plot_df
                    st.session_state.forecast_df = forecast_df
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")

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
                        line=dict(color='blue')
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
                        line=dict(color='red', dash='dot')
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
                            name='Upper Confidence',
                            line=dict(color='lightgray', dash='dash'),
                            showlegend=False
                        )
                    )
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['mean_ci_lower'],
                            name='Lower Confidence',
                            line=dict(color='lightgray', dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(128,128,128,0.2)',
                            showlegend=False
                        )
                    )

            fig_forecast.update_layout(
                title=f"{symbol} - 30 Day Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Generate forecast analysis
            if llm and 'forecast_df' in st.session_state:
                if st.button("📊 Analyze Forecast"):
                    with st.spinner("Generating forecast analysis..."):
                        try:
                            forecast_analysis = generate_forecast_analysis_report(
                                llm, symbol, st.session_state.forecast_df
                            )
                            st.markdown("**Forecast Analysis:**")
                            st.markdown(forecast_analysis)
                        except Exception as e:
                            st.error(f"Error generating forecast analysis: {e}")

    with col4:
        st.markdown("#### ⚠️ Anomaly Detection")
        if st.button("Detect Price/Volume Anomalies"):
            with st.spinner("Analyzing for market anomalies..."):
                try:
                    anomalies = detect_anomalies(enhanced_data)
                    st.session_state.anomalies = anomalies
                except Exception as e:
                    st.error(f"Error detecting anomalies: {e}")

        if 'anomalies' in st.session_state and not st.session_state.anomalies.empty:
            st.markdown(f"**🚨 {len(st.session_state.anomalies)} anomalies detected:**")
            
            # Show recent anomalies
            recent_anomalies = st.session_state.anomalies.tail(5)
            for idx, row in recent_anomalies.iterrows():
                st.markdown(f"""
                📅 **{idx.strftime('%Y-%m-%d')}**  
                💰 Price: ${row['Close']:.2f}  
                📊 Volume: {row['Volume']:,.0f}  
                ⚡ Price Change: {row['Price_Change']:.1%}  
                📈 Volume Change: {row['Volume_Change']:.1%}
                """)
                st.markdown("---")
        else:
            if 'anomalies' in st.session_state:
                st.info("No significant anomalies detected in recent trading.")

    # --- Market Intelligence Dashboard ---
    st.markdown("---")
    st.markdown("### 📈 Market Intelligence Dashboard")
    
    # Key metrics summary
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if len(enhanced_data) >= 2:
            current_price = enhanced_data['Close'].iloc[-1]
            price_change = current_price - enhanced_data['Close'].iloc[-2]
            price_change_pct = (price_change / enhanced_data['Close'].iloc[-2]) * 100
            st.metric(
                "Current Price", 
                f"${current_price:.2f}", 
                f"{price_change:+.2f} ({price_change_pct:+.1f}%)"
            )
        else:
            st.metric("Current Price", "N/A")
    
    with col6:
        if len(enhanced_data) >= 252:
            high_52w = enhanced_data['High'].tail(252).max()
            low_52w = enhanced_data['Low'].tail(252).min()
            st.metric("52W Range", f"${low_52w:.2f} - ${high_52w:.2f}")
        else:
            high_period = enhanced_data['High'].max()
            low_period = enhanced_data['Low'].min()
            st.metric("Period Range", f"${low_period:.2f} - ${high_period:.2f}")
    
    with col7:
        if len(enhanced_data) >= 20:
            avg_volume = enhanced_data['Volume'].tail(20).mean()
            current_volume = enhanced_data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            st.metric(
                "Volume vs Avg", 
                f"{current_volume:,.0f}", 
                f"{volume_ratio:.1f}x average"
            )
        else:
            current_volume = enhanced_data['Volume'].iloc[-1]
            st.metric("Current Volume", f"{current_volume:,.0f}")
    
    with col8:
        if 'ATR_14' in enhanced_data.columns:
            volatility = enhanced_data['ATR_14'].iloc[-1]
            current_price = enhanced_data['Close'].iloc[-1]
            if current_price > 0:
                vol_pct = (volatility / current_price) * 100
                st.metric("Volatility (ATR)", f"{vol_pct:.1f}%")
            else:
                st.metric("Volatility (ATR)", "N/A")
        else:
            st.metric("Volatility (ATR)", "N/A")

    # Additional insights
    st.markdown("#### 💡 Key Insights")
    insights = []
    
    # Generate dynamic insights based on data
    if not enhanced_data.empty:
        latest = enhanced_data.iloc[-1]
        
        if 'RSI_14' in enhanced_data.columns:
            rsi = latest['RSI_14']
            if rsi > 70:
                insights.append("🔴 RSI indicates overbought conditions - potential reversal signal")
            elif rsi < 30:
                insights.append("🟢 RSI indicates oversold conditions - potential buying opportunity")
        
        if all(col in enhanced_data.columns for col in ['EMA_20', 'SMA_50']):
            if latest['EMA_20'] > latest['SMA_50']:
                insights.append("🟢 Short-term momentum is bullish (EMA 20 > SMA 50)")
            else:
                insights.append("🔴 Short-term momentum is bearish (EMA 20 < SMA 50)")
        
        if 'Volume' in enhanced_data.columns and len(enhanced_data) >= 20:
            avg_vol = enhanced_data['Volume'].tail(20).mean()
            if latest['Volume'] > avg_vol * 1.5:
                insights.append("📊 Unusually high volume - potential breakout signal")
    
    # Display insights
    if insights:
        for insight in insights:
            st.markdown(f"• {insight}")
    else:
        st.info("Analysis complete - review technical indicators and patterns above for trading signals.")

    # --- Export Options ---
    st.markdown("---")
    st.markdown("### 📤 Export Analysis")
    
    col9, col10 = st.columns(2)
    
    with col9:
        if st.button("📊 Export Technical Data"):
            # Prepare export data
            export_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if len(enhanced_data) >= 50:
                export_data = enhanced_data.tail(50)[export_columns].copy()
            else:
                export_data = enhanced_data[export_columns].copy()
                
            # Add technical indicators if available
            if 'RSI_14' in enhanced_data.columns:
                export_data['RSI'] = enhanced_data['RSI_14'].tail(len(export_data))
            if 'MACD_12_26_9' in enhanced_data.columns:
                export_data['MACD'] = enhanced_data['MACD_12_26_9'].tail(len(export_data))
            
            csv = export_data.to_csv()
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"{symbol}_technical_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
    
    with col10:
        if 'comprehensive_analysis' in st.session_state:
            if st.button("📄 Export Analysis Report"):
                report_text = st.session_state.comprehensive_analysis
                st.download_button(
                    label="📋 Download Report",
                    data=report_text,
                    file_name=f"{symbol}_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime='text/plain'
                )
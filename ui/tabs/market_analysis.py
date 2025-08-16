import streamlit as st
import plotly.graph_objects as go
import json

from config import get_config
from services.data_fetcher import get_market_data_yfinance
from analysis.technical import detect_candlestick_patterns
from analysis.predictive import detect_anomalies, generate_forecast
from analysis.reporting import generate_candlestick_analysis_report, generate_forecast_analysis_report
from services.audio_processing import AudioFrameBuffer, transcribe_audio_with_openai
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

def render():
    """Renders the Market Analysis tab."""
    st.markdown("## 📊 Real-Time Market Analysis")
    cfg = get_config()
    symbol = cfg['selected_symbol']
    llm = cfg['llm']

    market_data = get_market_data_yfinance(symbol)

    if market_data.empty:
        st.error(f"Could not load market data for {symbol}.")
        return

    # --- Main Chart ---
    fig = go.Figure(go.Candlestick(
        x=market_data.index, open=market_data['Open'], high=market_data['High'],
        low=market_data['Low'], close=market_data['Close'], name=symbol
    ))
    fig.update_layout(title=f"{symbol} Candlestick Chart", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- AI/ML Analysis Section ---
    st.markdown("---")
    st.markdown("### 🤖 GenAI & ML Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Chart Annotation & Analysis")
        if st.button("Analyze Chart with GenAI", disabled=(llm is None)):
            with st.spinner("AI is analyzing the chart..."):
                patterns = detect_candlestick_patterns(market_data)
                st.session_state.detected_patterns = patterns
                report = generate_candlestick_analysis_report(llm, symbol, patterns, market_data)
                st.session_state.chart_analysis_report = report
        
        if 'chart_analysis_report' in st.session_state:
            st.markdown(st.session_state.chart_analysis_report)

    with col2:
        st.markdown("#### Predictive Modeling")
        if st.button("Generate 30-Day Forecast"):
            with st.spinner("Forecasting with ARIMA model..."):
                plot_df, forecast_df = generate_forecast(market_data)
                st.session_state.forecast_plot_df = plot_df
                st.session_state.forecast_df = forecast_df

        if 'forecast_plot_df' in st.session_state:
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=st.session_state.forecast_plot_df.index, y=st.session_state.forecast_plot_df.iloc[:, 0], name='Historical'))
            fig_forecast.add_trace(go.Scatter(x=st.session_state.forecast_plot_df.index, y=st.session_state.forecast_plot_df.iloc[:, 1], name='Forecast', line=dict(dash='dot')))
            st.plotly_chart(fig_forecast, use_container_width=True)

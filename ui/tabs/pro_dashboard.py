import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

from config import config
from services.data_fetcher import pro_get_real_time_data, pro_get_intraday_data, pro_get_news
from analysis.technical import run_ma_crossover_backtest

WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    return ["AAPL.US", "MSFT.US", "TSLA.US"]

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f)

def render():
    """Renders the Pro Dashboard tab."""
    st.header("📊 Stock Dashboard with Backtesting")
    symbol = st.session_state.get('selected_symbol', 'AAPL.US')
    api_key = config.get_eodhd_api_key()

    if not api_key:
        st.warning("EODHD API Key required for this dashboard.")
        return

    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.subheader("📋 Watchlist")
        watchlist = load_watchlist()
        for ticker in watchlist:
            data = pro_get_real_time_data(ticker, api_key)
            if data and 'close' in data:
                st.metric(label=ticker, value=f"{data.get('close','N/A')}", delta=f"{data.get('change_p','N/A')}%")
        
        # Manage Watchlist
        new_ticker = st.text_input("Add Ticker")
        if st.button("Add"):
            if new_ticker and new_ticker.upper() not in watchlist:
                watchlist.append(new_ticker.upper())
                save_watchlist(watchlist)
                st.rerun()

    with right_col:
        st.subheader(f"📌 Detailed View: {symbol}")
        
        # --- Backtesting Section ---
        st.markdown("---")
        st.subheader("🧪 Code-Free Strategy Backtesting")
        
        with st.form("backtest_form"):
            c1, c2, c3 = st.columns(3)
            start_date = c1.date_input("Start Date", datetime.now() - timedelta(days=365*2))
            end_date = c2.date_input("End Date", datetime.now())
            initial_capital = c3.number_input("Initial Capital", 1000, 1000000, 10000)
            
            c1, c2, _ = st.columns(3)
            short_window = c1.number_input("Short MA", 1, 100, 40)
            long_window = c2.number_input("Long MA", 2, 250, 100)

            if st.form_submit_button("🚀 Run Backtest"):
                with st.spinner("Running backtest..."):
                    results = run_ma_crossover_backtest(
                        api_key, symbol, start_date, end_date, initial_capital, short_window, long_window
                    )
                    st.session_state.backtest_results = results

        if 'backtest_results' in st.session_state:
            res = st.session_state.backtest_results
            if res.get("error"):
                st.error(res["error"])
            else:
                st.metric("Final Portfolio Value", f"${res['final_value']:,.2f}")
                st.metric("Total Return", f"{res['total_return_pct']:.2f}%")
                
                df_plot = res['plot_data']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], name='Price'))
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['portfolio_value'], name='Portfolio Value', yaxis='y2'))
                st.plotly_chart(fig, use_container_width=True)

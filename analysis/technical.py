import pandas as pd
import numpy as np
import random
from typing import List, Dict
import streamlit as st
import requests
from datetime import datetime

def detect_candlestick_patterns(data: pd.DataFrame) -> List[Dict]:
    """Simulates detection of candlestick patterns."""
    patterns = {
        "Doji": {"type": "Neutral", "description": "Indicates indecision."},
        "Bullish Engulfing": {"type": "Bullish", "description": "Potential reversal to the upside."},
        "Bearish Engulfing": {"type": "Bearish", "description": "Potential reversal to the downside."},
        "Hammer": {"type": "Bullish", "description": "Potential bottom."},
        "Shooting Star": {"type": "Bearish", "description": "Potential top."}
    }
    
    detected = []
    if len(data) < 60:
        return []
        
    for _ in range(random.randint(2, 4)):
        idx = random.randint(-60, -2)
        pattern_name = random.choice(list(patterns.keys()))
        pattern_info = patterns[pattern_name]
        
        if data.index[idx] in data.index:
            detected.append({
                "date": data.index[idx],
                "name": pattern_name,
                "type": pattern_info["type"],
                "description": pattern_info["description"]
            })
    
    return detected

def detect_fibonacci_levels(data: pd.DataFrame) -> dict:
    """Detects Fibonacci retracement levels."""
    if data.empty:
        return {}
    
    max_price = data['high'].max()
    min_price = data['low'].min()
    price_range = max_price - min_price

    if price_range == 0:
        return {}

    return {
        'Level 100%': max_price,
        'Level 61.8%': max_price - 0.382 * price_range,
        'Level 50.0%': max_price - 0.50 * price_range,
        'Level 38.2%': max_price - 0.618 * price_range,
        'Level 0%': min_price,
    }

@st.cache_data(ttl=600)
def run_ma_crossover_backtest(_api_key, ticker, start_date, end_date, initial_capital, short_window, long_window):
    """Runs a Moving Average Crossover backtest."""
    from_date_str = start_date.strftime('%Y-%m-%d')
    to_date_str = end_date.strftime('%Y-%m-%d')
    url = f"https://eodhd.com/api/eod/{ticker}?from={from_date_str}&to={to_date_str}&api_token={_api_key}&period=d&fmt=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
             return {"error": "No data returned from API."}
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        return {"error": f"Failed to fetch historical data: {e}"}
        
    df['short_mavg'] = df['close'].rolling(window=short_window).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window).mean()

    df['signal'] = 0.0
    df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1.0, 0.0)
    df['positions'] = df['signal'].diff()
    
    cash = float(initial_capital)
    shares = 0.0
    portfolio_history = []
    trade_log = []
    position_open = False

    for i in range(len(df)):
        price = df['close'][i]
        if df['positions'][i] == 1 and not position_open: # Buy
            shares_to_buy = cash / price
            shares += shares_to_buy
            cash = 0
            position_open = True
            trade_log.append({'type': 'BUY', 'date': df.index[i], 'price': price, 'shares': shares_to_buy})
        elif df['positions'][i] == -1 and position_open: # Sell
            cash += shares * price
            shares = 0
            position_open = False
            # Update last trade log with exit info
            if trade_log and trade_log[-1]['type'] == 'BUY':
                trade_log[-1].update({'exit_date': df.index[i], 'exit_price': price})

        portfolio_history.append(cash + (shares * price))

    df['portfolio_value'] = portfolio_history
    final_value = df['portfolio_value'].iloc[-1]
    total_return = ((final_value / initial_capital) - 1) * 100
    
    return {
        "final_value": final_value,
        "total_return_pct": total_return,
        "plot_data": df,
        "trade_log": pd.DataFrame(trade_log),
        "fib_levels": detect_fibonacci_levels(df),
        "error": None
    }

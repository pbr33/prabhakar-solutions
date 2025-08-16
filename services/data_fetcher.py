import yfinance as yf
import pandas as pd
import requests
import time
import streamlit as st
from datetime import datetime

# --- yfinance Data Fetcher ---
def get_market_data_yfinance(symbol: str, period: str = '1y') -> pd.DataFrame:
    """Fetch real market data using yfinance and add technical indicators."""
    try:
        import pandas_ta as ta
        cleaned_symbol = symbol.split('.')[0]
        ticker = yf.Ticker(cleaned_symbol)
        data = ticker.history(period=period, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data for ticker: {cleaned_symbol}")
        
        # Add technical indicators
        data.ta.rsi(append=True)
        data.ta.macd(append=True)
        return data
    except Exception as e:
        st.warning(f"yfinance failed for {symbol}: {e}. Check ticker or network.")
        return pd.DataFrame()

# --- EODHD API Functions ---
def pro_get_real_time_data(ticker, api_key):
    """Fetches real-time data from EODHD."""
    url = f"https://eodhd.com/api/real-time/{ticker}?api_token={api_key}&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}

def pro_get_fundamental_data(ticker, api_key):
    """Fetches fundamental data from EODHD."""
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_key}&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}

def pro_get_historical_data(ticker, api_key):
    """Fetches historical end-of-day data from EODHD."""
    url = f"https://eodhd.com/api/eod/{ticker}?api_token={api_key}&period=d&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')
    except Exception:
        return pd.DataFrame()

def pro_get_intraday_data(ticker, api_key, interval='1m'):
    """Fetches intraday data from EODHD."""
    url = f"https://eodhd.com/api/intraday/{ticker}?interval={interval}&api_token={api_key}&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.set_index('datetime')
    except Exception:
        return pd.DataFrame()

def pro_get_news(ticker, api_key):
    """Fetches news from EODHD."""
    url = f"https://eodhd.com/api/news?s={ticker}&api_token={api_key}&limit=5&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception:
        return []

def fetch_all_tickers(api_key: str):
    """Fetches all tickers from all available exchanges via EODHD."""
    # Fetch exchanges
    exchanges_url = f"https://eodhd.com/api/exchanges-list/?api_token={api_key}&fmt=json"
    try:
        exchanges = requests.get(exchanges_url).json()
    except Exception:
        return []

    all_tickers = []
    progress_bar = st.progress(0, text="Fetching tickers...")
    for i, exchange in enumerate(exchanges):
        code = exchange.get('Code')
        if not code: continue
        
        progress_text = f"Fetching: {exchange.get('Name', code)} ({i+1}/{len(exchanges)})"
        progress_bar.progress((i + 1) / len(exchanges), text=progress_text)
        
        tickers_url = f"https://eodhd.com/api/exchange-symbol-list/{code}?api_token={api_key}&fmt=json"
        try:
            tickers = requests.get(tickers_url).json()
            all_tickers.extend(tickers)
        except Exception:
            continue # Skip exchange on error
        time.sleep(0.2) # API politeness

    progress_bar.empty()
    return all_tickers
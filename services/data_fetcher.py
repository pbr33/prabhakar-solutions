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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}

def pro_get_fundamental_data(ticker, api_key):
    """Fetches fundamental data from EODHD."""
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_key}&fmt=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}

def pro_get_historical_data(ticker, api_key, from_date=None, to_date=None):
    """Fetches historical end-of-day data from EODHD.

    Args:
        ticker:    Stock symbol, e.g. 'AAPL' or 'AAPL.US'.
        api_key:   EODHD API key.
        from_date: Optional start date as 'YYYY-MM-DD' string.
        to_date:   Optional end date as 'YYYY-MM-DD' string.

    Returns:
        DataFrame indexed by date, or empty DataFrame on failure.
    """
    url = f"https://eodhd.com/api/eod/{ticker}?api_token={api_key}&period=d&fmt=json"
    if from_date:
        url += f"&from={from_date}"
    if to_date:
        url += f"&to={to_date}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 'unknown'
        if status == 404:
            return pd.DataFrame()  # Ticker not found – silent fail is OK
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def search_tickers_eodhd(query: str, api_key: str, limit: int = 20) -> list:
    """Search for tickers / companies via the EODHD search endpoint.

    Args:
        query:   Partial company name or ticker symbol (e.g. 'Apple', 'AAPL').
        api_key: EODHD API key.
        limit:   Maximum results to return (default 20).

    Returns:
        List of result dicts with keys: Code, Name, Country, Exchange,
        Currency, Type.  Empty list on failure or missing key.
    """
    if not api_key or not query or not query.strip():
        return []
    url = (
        f"https://eodhd.com/api/search/{query.strip()}"
        f"?api_token={api_key}&limit={limit}&fmt=json"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json()
        if isinstance(results, list):
            return results
        return []
    except Exception:
        return []

def pro_get_intraday_data(ticker, api_key, interval='1m'):
    """Fetches intraday data from EODHD."""
    url = f"https://eodhd.com/api/intraday/{ticker}?interval={interval}&api_token={api_key}&fmt=json"
    try:
        response = requests.get(url, timeout=10)
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
    """
    Fetches news from EODHD for a given ticker.

    EODHD news API works most reliably with the bare ticker symbol (no exchange
    suffix).  We try the bare symbol first (e.g. 'AAPL'), then fall back to
    the full symbol (e.g. 'AAPL.US') if no results are returned.
    """
    # Bare symbol (strip exchange suffix like '.US', '.LSE', etc.)
    bare = ticker.split('.')[0]
    candidates = [bare] if bare != ticker else [ticker]
    if bare != ticker:
        candidates.append(ticker)   # fallback: try full symbol too

    for sym in candidates:
        url = (
            f"https://eodhd.com/api/news"
            f"?s={sym}&offset=0&limit=20"
            f"&api_token={api_key}&fmt=json"
        )
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            # EODHD returns a list on success; a dict on error
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception:
            continue

    return []

def fetch_all_tickers(api_key: str):
    """Fetches all tickers from all available exchanges via EODHD."""
    if not api_key:
        st.warning("EODHD API key is required to fetch tickers.")
        return []
        
    # Fetch exchanges
    exchanges_url = f"https://eodhd.com/api/exchanges-list/?api_token={api_key}&fmt=json"
    try:
        response = requests.get(exchanges_url, timeout=30)
        response.raise_for_status()
        exchanges = response.json()
    except Exception as e:
        st.error(f"Failed to fetch exchanges: {e}")
        return []

    all_tickers = []
    progress_bar = st.progress(0, text="Fetching tickers...")
    
    # Limit to major exchanges to avoid timeout
    major_exchanges = ['US', 'NYSE', 'NASDAQ', 'LSE', 'TSE', 'XETRA']
    filtered_exchanges = [ex for ex in exchanges if ex.get('Code') in major_exchanges]
    
    if not filtered_exchanges:
        # If no major exchanges found, use first 10 exchanges
        filtered_exchanges = exchanges[:10]
    
    for i, exchange in enumerate(filtered_exchanges):
        code = exchange.get('Code')
        if not code: 
            continue
        
        progress_text = f"Fetching: {exchange.get('Name', code)} ({i+1}/{len(filtered_exchanges)})"
        progress_bar.progress((i + 1) / len(filtered_exchanges), text=progress_text)
        
        tickers_url = f"https://eodhd.com/api/exchange-symbol-list/{code}?api_token={api_key}&fmt=json"
        try:
            response = requests.get(tickers_url, timeout=15)
            response.raise_for_status()
            tickers = response.json()
            if isinstance(tickers, list):
                all_tickers.extend(tickers)
        except Exception as e:
            st.warning(f"Failed to fetch tickers for {code}: {e}")
            continue # Skip exchange on error
        
        time.sleep(0.1) # API politeness

    progress_bar.empty()
    return all_tickers
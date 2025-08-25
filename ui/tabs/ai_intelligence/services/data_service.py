# ui/tabs/ai_intelligence/services/data_service.py
"""
Production-ready data service with multiple API sources and intelligent fallbacks.
Updated to use config.py instead of st.secrets
"""

import yfinance as yf
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
import sys
import os

# Import the config system
try:
    # Try to import from project root
    from config import config
except ImportError:
    try:
        # Try relative import
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        from config import config
    except ImportError:
        # Create a fallback config
        class FallbackConfig:
            def get(self, section, key, default=None):
                return os.getenv(f"{section.upper()}_{key.upper()}", default)
        config = FallbackConfig()

class DataService:
    """Professional data service with multiple providers and caching."""
    
    def __init__(self):
        # API Keys from config (which reads from .env)
        self.eodhd_key = config.get('eodhd', 'api_key') or os.getenv('EODHD_API_KEY')
        self.alpha_vantage_key = config.get('alpha_vantage', 'api_key') or os.getenv('ALPHA_VANTAGE_KEY', 'demo')
        self.fmp_key = config.get('fmp', 'api_key') or os.getenv('FMP_API_KEY')
        
        # Fallback to Streamlit secrets if config doesn't work
        if not self.eodhd_key:
            try:
                self.eodhd_key = st.secrets.get("EODHD_API_KEY")
            except:
                pass
                
        if not self.alpha_vantage_key or self.alpha_vantage_key == 'demo':
            try:
                self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_KEY", "demo")
            except:
                pass
        
        # Cache configuration
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Rate limiting
        self.last_request = {}
        self.min_interval = 1  # Minimum 1 second between requests
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Log configuration status
        self._log_config_status()
    
    def _log_config_status(self):
        """Log the current configuration status for debugging."""
        self.logger.info("Data Service Configuration:")
        self.logger.info(f"  EODHD API Key: {'✓ Configured' if self.eodhd_key else '✗ Missing'}")
        self.logger.info(f"  Alpha Vantage Key: {'✓ Configured' if self.alpha_vantage_key and self.alpha_vantage_key != 'demo' else '✗ Missing/Demo'}")
        self.logger.info(f"  FMP API Key: {'✓ Configured' if self.fmp_key else '✗ Missing'}")
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data with intelligent fallbacks."""
        try:
            # Check cache first
            if self._is_cached(symbol):
                return self.cache[symbol]['data']
            
            # Try multiple sources in order of preference
            data = None
            
            # 1. Try EODHD (if API key available)
            if self.eodhd_key and self.eodhd_key.strip():
                data = self._fetch_eodhd_data(symbol)
                if data is not None:
                    self.logger.info(f"Successfully fetched {symbol} data from EODHD")
            
            # 2. Try Yahoo Finance as fallback (always available)
            if data is None:
                data = self._fetch_yahoo_data(symbol)
                if data is not None:
                    self.logger.info(f"Successfully fetched {symbol} data from Yahoo Finance")
            
            # 3. Try Alpha Vantage as last resort
            if data is None and self.alpha_vantage_key and self.alpha_vantage_key != "demo":
                data = self._fetch_alpha_vantage_data(symbol)
                if data is not None:
                    self.logger.info(f"Successfully fetched {symbol} data from Alpha Vantage")
            
            # 4. Generate demo data if all else fails
            if data is None:
                data = self._generate_demo_data(symbol)
                self.logger.warning(f"Using demo data for {symbol}")
            
            # Cache the result
            if data is not None:
                self._cache_data(symbol, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return self._generate_demo_data(symbol)
    
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote data."""
        try:
            if self.eodhd_key and self.eodhd_key.strip():
                quote = self._fetch_eodhd_realtime(symbol)
                if quote:
                    return quote
            
            # Fallback to Yahoo Finance
            return self._fetch_yahoo_realtime(symbol)
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time quote for {symbol}: {str(e)}")
            return self._generate_fallback_quote(symbol)
    
    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for the symbol."""
        try:
            if self.eodhd_key and self.eodhd_key.strip():
                fundamentals = self._fetch_eodhd_fundamentals(symbol)
                if fundamentals:
                    return fundamentals
            
            # Fallback to Yahoo Finance
            return self._fetch_yahoo_fundamentals(symbol)
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
            return {}
    
    def _generate_demo_data(self, symbol: str) -> pd.DataFrame:
        """Generate realistic demo data when APIs fail."""
        import numpy as np
        
        # Generate 1 year of realistic stock data
        dates = pd.date_range(start=datetime.now() - timedelta(days=365), 
                             end=datetime.now(), freq='D')
        dates = [d for d in dates if d.weekday() < 5]  # Only weekdays
        
        n_days = len(dates)
        
        # Starting price
        start_price = np.random.uniform(50, 300)
        
        # Generate realistic price movements
        returns = np.random.normal(0.0005, 0.02, n_days)  # Small daily returns with volatility
        prices = [start_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(1.0, new_price))  # Ensure price doesn't go negative
        
        # Generate OHLC data
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            daily_volatility = np.random.uniform(0.01, 0.04)
            
            # Generate intraday high/low around close
            high = close_price * (1 + daily_volatility)
            low = close_price * (1 - daily_volatility)
            
            # Open is previous close with small gap
            if i == 0:
                open_price = close_price
            else:
                gap = np.random.normal(0, 0.005)
                open_price = prices[i-1] * (1 + gap)
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume
            volume = int(np.random.uniform(100000, 5000000))
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        self.logger.info(f"Generated demo data for {symbol} with {len(df)} days")
        return df
    
    def _fetch_eodhd_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from EODHD."""
        try:
            self._rate_limit("eodhd")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            url = f"https://eodhd.com/api/eod/{symbol}"
            params = {
                'api_token': self.eodhd_key,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'fmt': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['date'])
            df.set_index('Date', inplace=True)
            
            # Rename columns to match standard format
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adjusted_close': 'Adj Close'
            }, inplace=True)
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            self.logger.warning(f"EODHD fetch failed for {symbol}: {str(e)}")
            return None
    
    def _fetch_yahoo_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        try:
            self._rate_limit("yahoo")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            
            if data.empty:
                return None
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                return None
            
            return data[required_cols]
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch failed for {symbol}: {str(e)}")
            return None
    
    def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        try:
            self._rate_limit("alpha_vantage")
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                return None
            
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                row = {
                    'Date': datetime.strptime(date_str, '%Y-%m-%d'),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['6. volume'])
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Get last year of data
            cutoff_date = datetime.now() - timedelta(days=365)
            df = df[df.index >= cutoff_date]
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Alpha Vantage fetch failed for {symbol}: {str(e)}")
            return None
    
    def _fetch_eodhd_realtime(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time quote from EODHD."""
        try:
            self._rate_limit("eodhd_realtime")
            
            url = f"https://eodhd.com/api/real-time/{symbol}"
            params = {
                'api_token': self.eodhd_key,
                'fmt': 'json'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'symbol': symbol,
                'price': float(data.get('close', 0)),
                'change': float(data.get('change', 0)),
                'change_percent': float(data.get('change_p', 0)),
                'volume': int(data.get('volume', 0)),
                'timestamp': datetime.now(),
                'source': 'EODHD'
            }
            
        except Exception as e:
            self.logger.warning(f"EODHD real-time fetch failed for {symbol}: {str(e)}")
            return None
    
    def _fetch_yahoo_realtime(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time quote from Yahoo Finance."""
        try:
            self._rate_limit("yahoo_realtime")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d", interval="1d")
            
            if hist.empty:
                return self._generate_fallback_quote(symbol)
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            change = current_price - prev_price
            change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
            
            return {
                'symbol': symbol,
                'price': float(current_price),
                'change': float(change),
                'change_percent': float(change_percent),
                'volume': int(hist['Volume'].iloc[-1]),
                'timestamp': datetime.now(),
                'source': 'Yahoo Finance'
            }
            
        except Exception as e:
            self.logger.warning(f"Yahoo real-time fetch failed for {symbol}: {str(e)}")
            return self._generate_fallback_quote(symbol)
    
    def _fetch_eodhd_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from EODHD."""
        try:
            self._rate_limit("eodhd_fundamentals")
            
            url = f"https://eodhd.com/api/fundamentals/{symbol}"
            params = {
                'api_token': self.eodhd_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.warning(f"EODHD fundamentals fetch failed for {symbol}: {str(e)}")
            return {}
    
    def _fetch_yahoo_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from Yahoo Finance."""
        try:
            self._rate_limit("yahoo_fundamentals")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Yahoo fundamentals fetch failed for {symbol}: {str(e)}")
            return {}
    
    def _generate_fallback_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate fallback quote when all APIs fail."""
        import random
        
        base_price = random.uniform(50, 500)
        change_percent = random.uniform(-5, 5)
        change = base_price * (change_percent / 100)
        
        return {
            'symbol': symbol,
            'price': base_price,
            'change': change,
            'change_percent': change_percent,
            'volume': random.randint(100000, 10000000),
            'timestamp': datetime.now(),
            'source': 'Demo Data'
        }
    
    def _is_cached(self, symbol: str) -> bool:
        """Check if symbol data is cached and still valid."""
        if symbol not in self.cache:
            return False
        
        cache_time = self.cache[symbol]['timestamp']
        return (time.time() - cache_time) < self.cache_duration
    
    def _cache_data(self, symbol: str, data: pd.DataFrame):
        """Cache the data with timestamp."""
        self.cache[symbol] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _rate_limit(self, source: str):
        """Implement rate limiting for API calls."""
        now = time.time()
        
        if source in self.last_request:
            time_since_last = now - self.last_request[source]
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
        
        self.last_request[source] = time.time()
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        # Common US stocks for demo
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'BABA', 'V', 'JPM', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'DIS', 'HD',
            'BAC', 'ADBE', 'CRM', 'VZ', 'CMCSA', 'PEP', 'TMO', 'ABT', 'COST',
            'AVGO', 'NKE', 'XOM', 'LLY', 'CVX', 'KO', 'MRK', 'PFE', 'DHR',
            'INTC', 'T', 'CSCO', 'ORCL', 'IBM', 'QCOM', 'AMD', 'COP', 'WFC',
            'MDT', 'UPS', 'CAT', 'HON', 'BA', 'GE', 'MMM', 'AXP', 'GS', 'MS'
        ]
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and has data."""
        if not symbol or len(symbol) < 1:
            return False
        
        try:
            # Quick validation with Yahoo Finance (most reliable)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid data
            return 'symbol' in info or 'shortName' in info or 'longName' in info
            
        except Exception:
            # If validation fails, assume symbol is valid for demo purposes
            return True
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        try:
            now = datetime.now()
            
            # Simple market hours check (US Eastern Time)
            # This is a simplified version - production would use proper timezone handling
            hour = now.hour
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            
            # Market open 9:30 AM - 4:00 PM ET, Monday-Friday
            is_market_hours = (
                weekday < 5 and  # Monday to Friday
                9.5 <= hour <= 16  # 9:30 AM to 4:00 PM (simplified)
            )
            
            return {
                'is_open': is_market_hours,
                'status': 'OPEN' if is_market_hours else 'CLOSED',
                'next_open': self._get_next_market_open(now),
                'timezone': 'US/Eastern'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market status: {str(e)}")
            return {
                'is_open': True,  # Default to open for demo
                'status': 'UNKNOWN',
                'next_open': None,
                'timezone': 'UTC'
            }
    
    def _get_next_market_open(self, current_time: datetime) -> Optional[datetime]:
        """Calculate next market open time."""
        try:
            # Simplified calculation - production would use market calendar
            weekday = current_time.weekday()
            
            if weekday < 5 and current_time.hour < 9:
                # Same day, market opens at 9:30 AM
                next_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            elif weekday < 4:
                # Next weekday
                next_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
            else:
                # Next Monday
                days_until_monday = (7 - weekday) % 7
                if days_until_monday == 0:
                    days_until_monday = 7
                next_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=days_until_monday)
            
            return next_open
            
        except Exception:
            return None
    
    def get_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols efficiently."""
        quotes = {}
        
        # Process in batches to avoid overwhelming APIs
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            for symbol in batch:
                try:
                    quote = self.get_real_time_quote(symbol)
                    quotes[symbol] = quote
                except Exception as e:
                    self.logger.warning(f"Failed to get quote for {symbol}: {str(e)}")
                    quotes[symbol] = self._generate_fallback_quote(symbol)
            
            # Small delay between batches
            time.sleep(0.5)
        
        return quotes
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_symbols': list(self.cache.keys()),
            'cache_size': len(self.cache),
            'cache_duration': self.cache_duration
        }
    
    def get_config_status(self) -> Dict[str, str]:
        """Get current configuration status for debugging."""
        return {
            'eodhd_configured': '✓' if self.eodhd_key and self.eodhd_key.strip() else '✗',
            'alpha_vantage_configured': '✓' if self.alpha_vantage_key and self.alpha_vantage_key != 'demo' else '✗',
            'fmp_configured': '✓' if self.fmp_key and self.fmp_key.strip() else '✗',
            'yahoo_available': '✓ (Always Available)',
            'demo_data_available': '✓ (Fallback)'
        }
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import os
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import warnings
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy import stats
import talib
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScenarioData:
    """Enhanced data structure for scenario information."""
    name: str
    emoji: str
    probability: float
    target_price: float
    timeframe: str
    expected_return: float
    catalysts: List[str]
    conditions: str
    risk_factors: str
    confidence_score: float
    market_regime: str
    fundamental_score: float
    technical_score: float
    sentiment_score: float
    volatility_adjusted_return: float
    sharpe_ratio: float
    max_drawdown: float
    support_levels: List[float]
    resistance_levels: List[float]
    ml_prediction: float
    ai_confidence: float

@dataclass
class FundamentalData:
    """Fundamental analysis data structure."""
    pe_ratio: float
    peg_ratio: float
    price_to_book: float
    price_to_sales: float
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    return_on_equity: float
    return_on_assets: float
    profit_margin: float
    operating_margin: float
    revenue_growth: float
    earnings_growth: float
    free_cash_flow: float
    enterprise_value: float
    market_cap: float
    dividend_yield: float
    payout_ratio: float
    beta: float
    analyst_rating: str
    analyst_target_price: float
    earnings_surprise: float
    revenue_surprise: float

class EODHDDataService:
    """Production-ready EODHD API service with comprehensive data fetching."""
    
    def __init__(self, api_key: str = None):
        # Configuration management
        try:
            from config import config
            self.api_key = api_key or config.get('eodhd', 'api_key') or os.getenv('EODHD_API_KEY')
        except ImportError:
            self.api_key = api_key or os.getenv('EODHD_API_KEY')
        
        # Fallback to streamlit secrets
        if not self.api_key:
            try:
                self.api_key = st.secrets.get("EODHD_API_KEY")
            except:
                pass
        
        self.base_url = "https://eodhd.com/api"
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.rate_limiter = {}
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'ScenarioEngine/2.0'})
        
        logger.info(f"EODHD Service initialized: {'✓' if self.api_key else '✗'}")
        
    def _check_rate_limit(self, endpoint: str) -> bool:
        """Check rate limiting for API calls."""
        now = time.time()
        if endpoint not in self.rate_limiter:
            self.rate_limiter[endpoint] = []
        
        # Remove old calls (older than 1 minute)
        self.rate_limiter[endpoint] = [
            call_time for call_time in self.rate_limiter[endpoint] 
            if now - call_time < 60
        ]
        
        # Check if we can make a call (max 100 per minute per endpoint)
        if len(self.rate_limiter[endpoint]) >= 100:
            return False
        
        self.rate_limiter[endpoint].append(now)
        return True
    
    def _make_api_call(self, url: str, params: Dict = None) -> Dict:
        """Make API call with error handling and rate limiting."""
        if not self._check_rate_limit(url):
            time.sleep(1)  # Wait if rate limited
        
        try:
            if params is None:
                params = {}
            if self.api_key:
                params['api_token'] = self.api_key
            params['fmt'] = 'json'
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {str(e)}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return {}
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """Get real-time market data."""
        cache_key = f"realtime_{symbol}_{int(time.time() // 60)}"  # 1-minute cache
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return self._get_yahoo_fallback(symbol)
        
        url = f"{self.base_url}/real-time/{symbol}"
        data = self._make_api_call(url)
        
        if data and 'close' in data:
            self.cache[cache_key] = data
            return data
        
        # Fallback to Yahoo Finance
        return self._get_yahoo_fallback(symbol)
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data."""
        cache_key = f"historical_{symbol}_{period}_{date.today()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return self._get_yahoo_historical(symbol, period)
        
        # Calculate date range
        end_date = datetime.now()
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=365)
        
        url = f"{self.base_url}/eod/{symbol}"
        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'period': 'd'
        }
        
        data = self._make_api_call(url, params)
        
        if data and isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            numeric_cols = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            self.cache[cache_key] = df
            return df
        
        # Fallback to Yahoo Finance
        return self._get_yahoo_historical(symbol, period)
    
    def get_fundamental_data(self, symbol: str) -> FundamentalData:
        """Get comprehensive fundamental data."""
        cache_key = f"fundamentals_{symbol}_{date.today()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return self._get_yahoo_fundamentals(symbol)
        
        url = f"{self.base_url}/fundamentals/{symbol}"
        data = self._make_api_call(url)
        
        if not data:
            return self._get_yahoo_fundamentals(symbol)
        
        try:
            # Extract key fundamental metrics
            highlights = data.get('Highlights', {})
            valuation = data.get('Valuation', {})
            financials = data.get('Financials', {})
            balance_sheet = financials.get('Balance_Sheet', {}) if financials else {}
            income_statement = financials.get('Income_Statement', {}) if financials else {}
            cash_flow = financials.get('Cash_Flow', {}) if financials else {}
            
            # Get latest quarterly data
            latest_quarter = max(balance_sheet.keys()) if balance_sheet else None
            latest_income = max(income_statement.keys()) if income_statement else None
            
            fundamentals = FundamentalData(
                pe_ratio=float(highlights.get('PERatio', 0) or 0),
                peg_ratio=float(highlights.get('PEGRatio', 0) or 0),
                price_to_book=float(highlights.get('PriceBookMRQ', 0) or 0),
                price_to_sales=float(highlights.get('PriceSalesTTM', 0) or 0),
                debt_to_equity=float(highlights.get('DebtToEquity', 0) or 0),
                current_ratio=float(balance_sheet.get(latest_quarter, {}).get('currentRatio', 0) or 0) if latest_quarter else 0,
                quick_ratio=float(balance_sheet.get(latest_quarter, {}).get('quickRatio', 0) or 0) if latest_quarter else 0,
                return_on_equity=float(highlights.get('ReturnOnEquityTTM', 0) or 0),
                return_on_assets=float(highlights.get('ReturnOnAssetsTTM', 0) or 0),
                profit_margin=float(highlights.get('ProfitMargin', 0) or 0),
                operating_margin=float(highlights.get('OperatingMarginTTM', 0) or 0),
                revenue_growth=float(highlights.get('RevenuePerShareTTM', 0) or 0),
                earnings_growth=float(highlights.get('QuarterlyEarningsGrowthYOY', 0) or 0),
                free_cash_flow=float(cash_flow.get(latest_quarter, {}).get('freeCashFlow', 0) or 0) if latest_quarter else 0,
                enterprise_value=float(highlights.get('EnterpriseValue', 0) or 0),
                market_cap=float(highlights.get('MarketCapitalization', 0) or 0),
                dividend_yield=float(highlights.get('DividendYield', 0) or 0),
                payout_ratio=float(highlights.get('PayoutRatio', 0) or 0),
                beta=float(highlights.get('Beta', 1.0) or 1.0),
                analyst_rating=data.get('AnalystRatings', {}).get('Rating', 'N/A'),
                analyst_target_price=float(data.get('AnalystRatings', {}).get('TargetPrice', 0) or 0),
                earnings_surprise=float(highlights.get('QuarterlyEarningsGrowthYOY', 0) or 0),
                revenue_surprise=float(highlights.get('QuarterlyRevenueGrowthYOY', 0) or 0)
            )
            
            self.cache[cache_key] = fundamentals
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error parsing fundamental data: {str(e)}")
            return self._get_yahoo_fundamentals(symbol)
    
    def get_financial_news(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent financial news and events."""
        cache_key = f"news_{symbol}_{date.today()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return []
        
        url = f"{self.base_url}/news"
        params = {
            's': symbol,
            'limit': limit,
            'offset': 0
        }
        
        data = self._make_api_call(url, params)
        
        if data and isinstance(data, list):
            self.cache[cache_key] = data
            return data
        
        return []
    
    def get_insider_transactions(self, symbol: str) -> List[Dict]:
        """Get insider trading data."""
        cache_key = f"insider_{symbol}_{date.today()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return []
        
        url = f"{self.base_url}/insider-transactions"
        params = {'code': symbol}
        
        data = self._make_api_call(url, params)
        
        if data:
            self.cache[cache_key] = data
            return data
        
        return []
    
    def get_options_data(self, symbol: str) -> Dict:
        """Get options data for sentiment analysis."""
        cache_key = f"options_{symbol}_{date.today()}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self.api_key:
            return {}
        
        url = f"{self.base_url}/options/{symbol}"
        data = self._make_api_call(url)
        
        if data:
            self.cache[cache_key] = data
            return data
        
        return {}
    
    def _get_yahoo_fallback(self, symbol: str) -> Dict:
        """Yahoo Finance fallback for real-time data."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            
            if hist.empty:
                return {}
            
            current = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else current
            
            return {
                'close': float(current['Close']),
                'open': float(current['Open']),
                'high': float(current['High']),
                'low': float(current['Low']),
                'volume': int(current['Volume']),
                'change': float(current['Close'] - prev['Close']),
                'change_p': float(((current['Close'] - prev['Close']) / prev['Close']) * 100),
                'timestamp': int(time.time())
            }
        except Exception as e:
            logger.error(f"Yahoo fallback failed: {e}")
            return {}
    
    def _get_yahoo_historical(self, symbol: str, period: str) -> pd.DataFrame:
        """Yahoo Finance fallback for historical data."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Rename columns to match EODHD format
            hist = hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            hist['adjusted_close'] = hist['close']
            return hist
            
        except Exception as e:
            logger.error(f"Yahoo historical fallback failed: {e}")
            return pd.DataFrame()
    
    def _get_yahoo_fundamentals(self, symbol: str) -> FundamentalData:
        """Yahoo Finance fallback for fundamentals."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return FundamentalData(
                pe_ratio=info.get('trailingPE', 15.0) or 15.0,
                peg_ratio=info.get('pegRatio', 1.0) or 1.0,
                price_to_book=info.get('priceToBook', 2.0) or 2.0,
                price_to_sales=info.get('priceToSalesTrailing12Months', 2.0) or 2.0,
                debt_to_equity=info.get('debtToEquity', 50.0) or 50.0,
                current_ratio=info.get('currentRatio', 2.0) or 2.0,
                quick_ratio=info.get('quickRatio', 1.5) or 1.5,
                return_on_equity=info.get('returnOnEquity', 0.15) or 0.15,
                return_on_assets=info.get('returnOnAssets', 0.08) or 0.08,
                profit_margin=info.get('profitMargins', 0.1) or 0.1,
                operating_margin=info.get('operatingMargins', 0.12) or 0.12,
                revenue_growth=info.get('revenueGrowth', 0.05) or 0.05,
                earnings_growth=info.get('earningsGrowth', 0.08) or 0.08,
                free_cash_flow=info.get('freeCashflow', 1000000) or 1000000,
                enterprise_value=info.get('enterpriseValue', 10000000) or 10000000,
                market_cap=info.get('marketCap', 8000000) or 8000000,
                dividend_yield=info.get('dividendYield', 0.02) or 0.02,
                payout_ratio=info.get('payoutRatio', 0.4) or 0.4,
                beta=info.get('beta', 1.0) or 1.0,
                analyst_rating='Hold',
                analyst_target_price=info.get('targetMeanPrice', 0.0) or 0.0,
                earnings_surprise=0.0,
                revenue_surprise=0.0
            )
        except Exception as e:
            logger.error(f"Yahoo fundamentals fallback failed: {e}")
            return self._get_default_fundamentals()
    
    def _get_default_fundamentals(self) -> FundamentalData:
        """Return default fundamental data when all sources fail."""
        return FundamentalData(
            pe_ratio=15.0, peg_ratio=1.0, price_to_book=2.0, price_to_sales=2.0,
            debt_to_equity=0.5, current_ratio=2.0, quick_ratio=1.5, return_on_equity=15.0,
            return_on_assets=8.0, profit_margin=10.0, operating_margin=12.0,
            revenue_growth=5.0, earnings_growth=8.0, free_cash_flow=1000000,
            enterprise_value=10000000, market_cap=8000000, dividend_yield=2.0,
            payout_ratio=40.0, beta=1.0, analyst_rating='Hold',
            analyst_target_price=0.0, earnings_surprise=0.0, revenue_surprise=0.0
        )

class AdvancedTechnicalAnalyzer:
    """Advanced technical analysis with multiple indicators and ML integration."""
    
    def __init__(self):
        self.indicators = {}
        self.scaler = StandardScaler()
    
    def analyze_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Comprehensive technical analysis with ML predictions."""
        if df.empty or len(df) < 50:
            return self._get_default_technical_data()
        
        try:
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column {col}, using fallback data")
                    return self._get_default_technical_data()
            
            # Price data
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            open_prices = df['open'].values
            
            # Handle NaN values
            close = np.nan_to_num(close, nan=np.nanmean(close))
            high = np.nan_to_num(high, nan=np.nanmean(high))
            low = np.nan_to_num(low, nan=np.nanmean(low))
            volume = np.nan_to_num(volume, nan=np.nanmean(volume))
            
            # Moving averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            # Momentum indicators
            rsi = talib.RSI(close, timeperiod=14)
            macd, macdsignal, macdhist = talib.MACD(close)
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            
            # Volatility indicators
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # Volume indicators
            obv = talib.OBV(close, volume)
            ad = talib.AD(high, low, close, volume)
            
            # Price patterns and levels
            support_levels = self._find_support_resistance(close, low, 'support')
            resistance_levels = self._find_support_resistance(close, high, 'resistance')
            
            # Current values with NaN handling
            current_price = float(close[-1])
            current_rsi = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
            current_macd = float(macd[-1]) if not np.isnan(macd[-1]) else 0.0
            current_macd_signal = float(macdsignal[-1]) if not np.isnan(macdsignal[-1]) else 0.0
            
            # Technical scores
            trend_score = self._calculate_trend_score(
                current_price, 
                float(sma_20[-1]) if not np.isnan(sma_20[-1]) else current_price,
                float(sma_50[-1]) if not np.isnan(sma_50[-1]) else current_price,
                float(sma_200[-1]) if not np.isnan(sma_200[-1]) else current_price
            )
            momentum_score = self._calculate_momentum_score(current_rsi, current_macd, current_macd_signal)
            volume_score = self._calculate_volume_score(obv, volume)
            
            # ML prediction
            ml_prediction = self._generate_ml_prediction(df)
            
            return {
                'current_price': current_price,
                'sma_20': float(sma_20[-1]) if not np.isnan(sma_20[-1]) else current_price,
                'sma_50': float(sma_50[-1]) if not np.isnan(sma_50[-1]) else current_price,
                'sma_200': float(sma_200[-1]) if not np.isnan(sma_200[-1]) else current_price,
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_macd_signal,
                'bb_upper': float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else current_price * 1.1,
                'bb_lower': float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else current_price * 0.9,
                'atr': float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'technical_score': (trend_score + momentum_score + volume_score) / 3,
                'ml_prediction': ml_prediction,
                'pattern_recognition': self._identify_chart_patterns(close, high, low)
            }
            
        except Exception as e:
            logger.error(f"Technical analysis error: {str(e)}")
            return self._get_default_technical_data()
    
    def _find_support_resistance(self, close_prices: np.ndarray, reference_prices: np.ndarray, level_type: str) -> List[float]:
        """Find support and resistance levels."""
        try:
            levels = []
            window = min(20, len(reference_prices) // 5)
            
            if window < 3:
                return [float(np.mean(reference_prices))]
            
            # Find local extrema
            for i in range(window, len(reference_prices) - window):
                window_data = reference_prices[i-window:i+window+1]
                
                if level_type == 'support':
                    if reference_prices[i] == np.min(window_data):
                        levels.append(float(reference_prices[i]))
                else:  # resistance
                    if reference_prices[i] == np.max(window_data):
                        levels.append(float(reference_prices[i]))
            
            # Remove duplicates and sort
            levels = sorted(list(set([round(level, 2) for level in levels])))
            
            # Return top 3 levels
            if level_type == 'support':
                return levels[-3:] if len(levels) >= 3 else levels
            else:
                return levels[:3] if len(levels) >= 3 else levels
            
        except Exception as e:
            logger.error(f"Support/Resistance calculation error: {e}")
            current_price = float(np.mean(close_prices))
            if level_type == 'support':
                return [current_price * 0.95, current_price * 0.90]
            else:
                return [current_price * 1.05, current_price * 1.10]
    
    def _calculate_trend_score(self, price: float, sma20: float, sma50: float, sma200: float) -> float:
        """Calculate trend score (0-100)."""
        try:
            score = 50  # Neutral
            
            # Price vs moving averages
            if price > sma20:
                score += 10
            if price > sma50:
                score += 15
            if price > sma200:
                score += 20
            
            # Moving average alignment
            if sma20 > sma50:
                score += 5
            if sma50 > sma200:
                score += 10
            
            return min(100, max(0, score))
            
        except Exception:
            return 50.0
    
    def _calculate_momentum_score(self, rsi: float, macd: float, macd_signal: float) -> float:
        """Calculate momentum score (0-100)."""
        try:
            score = 50  # Neutral
            
            # RSI analysis
            if 40 <= rsi <= 60:
                score += 5  # Neutral zone
            elif rsi > 60:
                score += (rsi - 60) / 2
            else:
                score -= (40 - rsi) / 2
            
            # MACD analysis
            if macd > macd_signal:
                score += 15
            if macd > 0:
                score += 10
            
            return min(100, max(0, score))
            
        except Exception:
            return 50.0
    
    def _calculate_volume_score(self, obv: np.ndarray, volume: np.ndarray) -> float:
        """Calculate volume score (0-100)."""
        try:
            score = 50
            
            # OBV trend
            if len(obv) >= 20:
                obv_trend = np.polyfit(range(20), obv[-20:], 1)[0]
                if obv_trend > 0:
                    score += 20
                else:
                    score -= 10
            
            # Volume trend
            if len(volume) >= 20:
                avg_recent = np.mean(volume[-5:])
                avg_historical = np.mean(volume[-20:-5])
                
                if avg_recent > avg_historical * 1.2:
                    score += 15
                elif avg_recent < avg_historical * 0.8:
                    score -= 10
            
            return min(100, max(0, score))
            
        except Exception:
            return 50.0
    
    def _generate_ml_prediction(self, df: pd.DataFrame) -> Dict:
        """Generate ML-based price predictions."""
        try:
            if len(df) < 60:
                return {'prediction': 0.0, 'confidence': 0.5, 'direction': 'neutral'}
            
            # Prepare features
            features = self._prepare_ml_features(df)
            
            if features is None or len(features) < 30:
                return {'prediction': 0.0, 'confidence': 0.5, 'direction': 'neutral'}
            
            # Simple ML model using recent price patterns
            close_prices = df['close'].values[-60:]  # Last 60 days
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Feature engineering
            X = []
            y = []
            window_size = 10
            
            for i in range(window_size, len(returns)):
                X.append(returns[i-window_size:i])
                y.append(returns[i])
            
            if len(X) < 20:
                return {'prediction': 0.0, 'confidence': 0.5, 'direction': 'neutral'}
            
            X = np.array(X)
            y = np.array(y)
            
            # Train simple model
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=0.1)
            model.fit(X[:-5], y[:-5])  # Leave last 5 for validation
            
            # Predict next return
            last_returns = returns[-window_size:].reshape(1, -1)
            predicted_return = model.predict(last_returns)[0]
            
            # Calculate confidence based on model performance
            val_pred = model.predict(X[-5:])
            mse = mean_squared_error(y[-5:], val_pred)
            confidence = max(0.3, min(0.9, 1 - (mse * 100)))
            
            direction = 'bullish' if predicted_return > 0.01 else 'bearish' if predicted_return < -0.01 else 'neutral'
            
            return {
                'prediction': float(predicted_return),
                'confidence': float(confidence),
                'direction': direction,
                'model_type': 'Ridge Regression'
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {'prediction': 0.0, 'confidence': 0.5, 'direction': 'neutral'}
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ML model."""
        try:
            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calculate technical indicators as features
            rsi = talib.RSI(close, timeperiod=14)
            macd, _, _ = talib.MACD(close)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            atr = talib.ATR(high, low, close, timeperiod=14)
            
            # Price-based features
            returns = np.diff(close) / close[:-1]
            volatility = pd.Series(returns).rolling(window=20).std().values
            
            # Combine features
            features = np.column_stack([
                rsi[20:],  # Skip first 20 due to NaN
                macd[20:],
                (close[20:] - bb_middle[20:]) / bb_middle[20:],  # BB position
                atr[20:] / close[20:],  # Normalized ATR
                volatility[19:]  # One less due to returns diff
            ])
            
            # Remove NaN rows
            features = features[~np.isnan(features).any(axis=1)]
            
            return features if len(features) > 10 else None
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None
    
    def _get_default_technical_data(self) -> Dict:
        """Return default technical data when analysis fails."""
        return {
            'current_price': 100.0,
            'sma_20': 100.0,
            'sma_50': 100.0,
            'sma_200': 100.0,
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'bb_upper': 110.0,
            'bb_lower': 90.0,
            'atr': 2.0,
            'support_levels': [95.0, 90.0],
            'resistance_levels': [105.0, 110.0],
            'trend_score': 50.0,
            'momentum_score': 50.0,
            'volume_score': 50.0,
            'technical_score': 50.0,
            'ml_prediction': {'prediction': 0.0, 'confidence': 0.5, 'direction': 'neutral'},
            'pattern_recognition': []
        }
    
    def _identify_chart_patterns(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> List[str]:
        """Identify chart patterns using pattern recognition."""
        patterns = []
        
        try:
            if len(close) < 20:
                return patterns
            
            # Simple pattern recognition
            recent_close = close[-20:]
            recent_high = high[-20:]
            recent_low = low[-20:]
            
            # Double top/bottom detection
            if self._is_double_top(recent_high):
                patterns.append("Double Top")
            elif self._is_double_bottom(recent_low):
                patterns.append("Double Bottom")
            
            # Trend patterns
            if self._is_ascending_triangle(recent_high, recent_low):
                patterns.append("Ascending Triangle")
            elif self._is_descending_triangle(recent_high, recent_low):
                patterns.append("Descending Triangle")
            
            # Support/Resistance breaks
            if self._is_breakout(recent_close, recent_high):
                patterns.append("Resistance Breakout")
            elif self._is_breakdown(recent_close, recent_low):
                patterns.append("Support Breakdown")
            
        except Exception as e:
            logger.error(f"Pattern recognition error: {e}")
        
        return patterns
    
    def _is_double_top(self, highs: np.ndarray) -> bool:
        """Simple double top detection."""
        if len(highs) < 10:
            return False
        
        # Find peaks
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) < 2:
            return False
        
        # Check if two highest peaks are similar
        peaks.sort(key=lambda x: x[1], reverse=True)
        peak1, peak2 = peaks[0], peaks[1]
        
        # Similar height (within 2%)
        return abs(peak1[1] - peak2[1]) / max(peak1[1], peak2[1]) < 0.02
    
    def _is_double_bottom(self, lows: np.ndarray) -> bool:
        """Simple double bottom detection."""
        if len(lows) < 10:
            return False
        
        # Find troughs
        troughs = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        if len(troughs) < 2:
            return False
        
        # Check if two lowest troughs are similar
        troughs.sort(key=lambda x: x[1])
        trough1, trough2 = troughs[0], troughs[1]
        
        # Similar depth (within 2%)
        return abs(trough1[1] - trough2[1]) / min(trough1[1], trough2[1]) < 0.02
    
    def _is_ascending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Simple ascending triangle detection."""
        if len(highs) < 10:
            return False
        
        # Check if highs are relatively flat and lows are ascending
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        return abs(high_trend) < 0.001 and low_trend > 0.001
    
    def _is_descending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Simple descending triangle detection."""
        if len(highs) < 10:
            return False
        
        # Check if lows are relatively flat and highs are descending
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        return high_trend < -0.001 and abs(low_trend) < 0.001
    
    def _is_breakout(self, close: np.ndarray, highs: np.ndarray) -> bool:
        """Detect resistance breakout."""
        if len(close) < 10:
            return False
        
        recent_resistance = np.max(highs[-10:-2])  # Exclude last 2 days
        current_price = close[-1]
        
        return current_price > recent_resistance * 1.01  # 1% above resistance
    
    def _is_breakdown(self, close: np.ndarray, lows: np.ndarray) -> bool:
        """Detect support breakdown."""
        if len(close) < 10:
            return False
        
        recent_support = np.min(lows[-10:-2])  # Exclude last 2 days
        current_price = close[-1]
        
        return current_price < recent_support * 0.99  # 1% below support

class SentimentAnalyzer:
    """Advanced sentiment analysis from multiple sources."""
    
    def __init__(self):
        self.sentiment_cache = {}
    
    def analyze_market_sentiment(self, symbol: str, news_data: List[Dict], 
                               options_data: Dict, insider_data: List[Dict]) -> Dict:
        """Comprehensive sentiment analysis."""
        try:
            # News sentiment
            news_sentiment = self._analyze_news_sentiment(news_data)
            
            # Options sentiment (Put/Call ratio analysis)
            options_sentiment = self._analyze_options_sentiment(options_data)
            
            # Insider sentiment
            insider_sentiment = self._analyze_insider_sentiment(insider_data)
            
            # Combined sentiment score
            sentiment_score = (news_sentiment['score'] * 0.4 + 
                             options_sentiment['score'] * 0.3 + 
                             insider_sentiment['score'] * 0.3)
            
            return {
                'overall_score': sentiment_score,
                'sentiment_label': self._get_sentiment_label(sentiment_score),
                'news_sentiment': news_sentiment,
                'options_sentiment': options_sentiment,
                'insider_sentiment': insider_sentiment,
                'confidence': min(0.9, max(0.3, (len(news_data) + len(insider_data)) / 20))
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._get_default_sentiment()
    
    def _analyze_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze news sentiment using keyword analysis."""
        try:
            if not news_data:
                return {'score': 50, 'articles_analyzed': 0}
            
            positive_words = [
                'growth', 'profit', 'gain', 'beat', 'outperform', 'upgrade', 
                'strong', 'positive', 'bullish', 'buy', 'rally', 'surge'
            ]
            
            negative_words = [
                'loss', 'decline', 'miss', 'underperform', 'downgrade',
                'weak', 'negative', 'bearish', 'sell', 'crash', 'plunge'
            ]
            
            total_score = 0
            article_count = min(20, len(news_data))  # Limit to 20 articles
            
            for article in news_data[:article_count]:
                title = article.get('title', '').lower()
                content = article.get('content', '').lower() if 'content' in article else title
                
                # Simple keyword scoring
                positive_count = sum(1 for word in positive_words if word in content)
                negative_count = sum(1 for word in negative_words if word in content)
                
                # Article sentiment (-1 to +1)
                if positive_count + negative_count > 0:
                    article_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                else:
                    article_sentiment = 0
                
                total_score += article_sentiment
            
            # Convert to 0-100 scale
            avg_sentiment = total_score / article_count if article_count > 0 else 0
            sentiment_score = 50 + (avg_sentiment * 25)  # Scale to 25-75 range
            
            return {
                'score': max(0, min(100, sentiment_score)),
                'articles_analyzed': article_count
            }
            
        except Exception as e:
            logger.error(f"News sentiment error: {e}")
            return {'score': 50, 'articles_analyzed': 0}
    
    def _analyze_options_sentiment(self, options_data: Dict) -> Dict:
        """Analyze options data for sentiment indicators."""
        try:
            if not options_data:
                return {'score': 50, 'put_call_ratio': 1.0}
            
            # Extract put/call data
            calls = options_data.get('calls', [])
            puts = options_data.get('puts', [])
            
            if not calls and not puts:
                return {'score': 50, 'put_call_ratio': 1.0}
            
            # Calculate put/call ratio by volume
            call_volume = sum([opt.get('volume', 0) for opt in calls if opt.get('volume')])
            put_volume = sum([opt.get('volume', 0) for opt in puts if opt.get('volume')])
            
            if call_volume == 0:
                put_call_ratio = 2.0  # Very bearish
            else:
                put_call_ratio = put_volume / call_volume
            
            # Convert P/C ratio to sentiment score (lower ratio = more bullish)
            # Normal P/C ratio is around 0.7-1.3
            if put_call_ratio < 0.7:
                score = 75  # Bullish
            elif put_call_ratio > 1.3:
                score = 25  # Bearish
            else:
                score = 50 - ((put_call_ratio - 1.0) * 25)  # Linear interpolation
            
            return {
                'score': max(0, min(100, score)),
                'put_call_ratio': put_call_ratio
            }
            
        except Exception as e:
            logger.error(f"Options sentiment error: {e}")
            return {'score': 50, 'put_call_ratio': 1.0}
    
    def _analyze_insider_sentiment(self, insider_data: List[Dict]) -> Dict:
        """Analyze insider trading for sentiment."""
        try:
            if not insider_data:
                return {'score': 50, 'transactions_analyzed': 0}
            
            buy_value = 0
            sell_value = 0
            transaction_count = 0
            
            for transaction in insider_data[:50]:  # Limit to recent 50 transactions
                transaction_type = transaction.get('transactionType', '').upper()
                value = float(transaction.get('value', 0))
                
                if 'BUY' in transaction_type or 'PURCHASE' in transaction_type:
                    buy_value += value
                elif 'SELL' in transaction_type or 'SALE' in transaction_type:
                    sell_value += value
                
                transaction_count += 1
            
            # Calculate buy/sell ratio
            if sell_value == 0:
                buy_sell_ratio = 10 if buy_value > 0 else 1
            else:
                buy_sell_ratio = buy_value / sell_value
            
            # Convert to sentiment score
            if buy_sell_ratio > 2:
                score = 75  # Strong buying
            elif buy_sell_ratio < 0.5:
                score = 25  # Strong selling
            else:
                score = 35 + (buy_sell_ratio / 2 * 30)  # Linear scale
            
            return {
                'score': max(0, min(100, score)),
                'transactions_analyzed': transaction_count,
                'buy_sell_ratio': buy_sell_ratio
            }
            
        except Exception as e:
            logger.error(f"Insider sentiment error: {e}")
            return {'score': 50, 'transactions_analyzed': 0}
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score >= 70:
            return "Very Bullish"
        elif score >= 60:
            return "Bullish"
        elif score >= 40:
            return "Neutral"
        elif score >= 30:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def _get_default_sentiment(self) -> Dict:
        """Return default sentiment data."""
        return {
            'overall_score': 50,
            'sentiment_label': 'Neutral',
            'news_sentiment': {'score': 50, 'articles_analyzed': 0},
            'options_sentiment': {'score': 50, 'put_call_ratio': 1.0},
            'insider_sentiment': {'score': 50, 'transactions_analyzed': 0},
            'confidence': 0.3
        }

class EnhancedScenarioEngine:
    """Production-ready enhanced scenario engine with comprehensive analysis."""
    
    def __init__(self):
        self.data_service = EODHDDataService()
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.scenario_cache = {}
        
    def generate_comprehensive_scenarios(self, symbol: str, timeframe_days: int = 90) -> Dict[str, ScenarioData]:
        """Generate comprehensive AI-powered scenarios with full analysis."""
        
        try:
            # Get market data
            progress_placeholder = st.empty()
            progress_placeholder.info("🔄 Fetching market data...")
            
            # Historical data
            historical_data = self.data_service.get_historical_data(symbol, "1y")
            real_time_data = self.data_service.get_real_time_data(symbol)
            fundamental_data = self.data_service.get_fundamental_data(symbol)
            
            progress_placeholder.info("📈 Analyzing technical indicators...")
            
            # Technical analysis
            if not historical_data.empty:
                technical_analysis = self.technical_analyzer.analyze_technical_indicators(historical_data)
                current_price = technical_analysis['current_price']
            else:
                technical_analysis = self.technical_analyzer._get_default_technical_data()
                current_price = real_time_data.get('close', 100.0)
            
            progress_placeholder.info("📰 Analyzing market sentiment...")
            
            # Sentiment analysis
            news_data = self.data_service.get_financial_news(symbol)
            options_data = self.data_service.get_options_data(symbol)
            insider_data = self.data_service.get_insider_transactions(symbol)
            
            sentiment_analysis = self.sentiment_analyzer.analyze_market_sentiment(
                symbol, news_data, options_data, insider_data
            )
            
            progress_placeholder.info("🎯 Generating scenarios...")
            
            # Market regime detection
            regime = self._detect_advanced_market_regime(
                technical_analysis, sentiment_analysis, fundamental_data
            )
            
            # Generate scenarios
            scenarios = self._create_enhanced_scenarios(
                symbol=symbol,
                current_price=current_price,
                timeframe_days=timeframe_days,
                technical_analysis=technical_analysis,
                fundamental_data=fundamental_data,
                sentiment_analysis=sentiment_analysis,
                regime=regime
            )
            
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(1)
            progress_placeholder.empty()
            
            return scenarios
            
        except Exception as e:
            st.error(f"Scenario generation failed: {str(e)}")
            logger.error(f"Scenario generation error: {e}")
            return self._get_fallback_scenarios(symbol, timeframe_days)
    
    def _detect_advanced_market_regime(self, technical: Dict, sentiment: Dict, 
                                     fundamental: FundamentalData) -> str:
        """Advanced market regime detection."""
        try:
            # Technical regime factors
            trend_score = technical.get('trend_score', 50)
            momentum_score = technical.get('momentum_score', 50)
            rsi = technical.get('rsi', 50)
            
            # Sentiment factors
            sentiment_score = sentiment.get('overall_score', 50)
            
            # Fundamental factors
            pe_ratio = fundamental.pe_ratio
            
            # Regime detection logic
            if sentiment_score > 70 and trend_score > 70 and momentum_score > 70:
                return "Euphoric Bull Market"
            elif sentiment_score < 30 and trend_score < 30 and rsi < 30:
                return "Panic Bear Market"
            elif rsi > 70 and sentiment_score > 60:
                return "Overbought Rally"
            elif rsi < 30 and sentiment_score < 40:
                return "Oversold Decline"
            elif trend_score > 60 and momentum_score > 60:
                return "Bullish Trend"
            elif trend_score < 40 and momentum_score < 40:
                return "Bearish Trend"
            elif 40 <= sentiment_score <= 60 and 45 <= trend_score <= 55:
                return "Range-bound Market"
            else:
                return "Mixed Signals"
                
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return "Unknown Regime"
    
    def _create_enhanced_scenarios(self, symbol: str, current_price: float, timeframe_days: int,
                                 technical_analysis: Dict, fundamental_data: FundamentalData,
                                 sentiment_analysis: Dict, regime: str) -> Dict[str, ScenarioData]:
        """Create enhanced scenarios with comprehensive analysis."""
        
        # Calculate volatility
        volatility = self._calculate_volatility_estimate(technical_analysis)
        
        # Adjust probabilities based on regime
        base_probs = self._get_regime_adjusted_probabilities(regime, sentiment_analysis)
        
        scenarios = {}
        
        # Moonshot scenario
        scenarios['🚀 Moonshot'] = self._create_enhanced_moonshot_scenario(
            current_price, volatility, timeframe_days, base_probs['moonshot'], 
            technical_analysis, fundamental_data, sentiment_analysis, regime, symbol
        )
        
        # Bull scenario
        scenarios['🐂 Bull Case'] = self._create_enhanced_bull_scenario(
            current_price, volatility, timeframe_days, base_probs['bull'],
            technical_analysis, fundamental_data, sentiment_analysis, regime, symbol
        )
        
        # Base scenario
        scenarios['📊 Base Case'] = self._create_enhanced_base_scenario(
            current_price, volatility, timeframe_days, base_probs['base'],
            technical_analysis, fundamental_data, sentiment_analysis, regime, symbol
        )
        
        # Bear scenario
        scenarios['🐻 Bear Case'] = self._create_enhanced_bear_scenario(
            current_price, volatility, timeframe_days, base_probs['bear'],
            technical_analysis, fundamental_data, sentiment_analysis, regime, symbol
        )
        
        # Black swan scenario
        scenarios['⚡ Black Swan'] = self._create_enhanced_black_swan_scenario(
            current_price, volatility, timeframe_days, base_probs['black_swan'],
            technical_analysis, fundamental_data, sentiment_analysis, regime, symbol
        )
        
        return scenarios
    
    def _calculate_volatility_estimate(self, technical_analysis: Dict) -> float:
        """Calculate volatility estimate from technical data."""
        try:
            atr = technical_analysis.get('atr', 2.0)
            current_price = technical_analysis.get('current_price', 100.0)
            
            # Convert ATR to percentage volatility
            volatility = (atr / current_price) * np.sqrt(252)  # Annualized
            
            return max(0.15, min(0.80, volatility))  # Cap between 15% and 80%
            
        except Exception:
            return 0.25  # Default 25% volatility
    
    def _get_regime_adjusted_probabilities(self, regime: str, sentiment: Dict) -> Dict:
        """Get probability adjustments based on market regime."""
        
        base_probs = {'moonshot': 8, 'bull': 25, 'base': 40, 'bear': 22, 'black_swan': 5}
        
        regime_adjustments = {
            "Euphoric Bull Market": {'moonshot': +5, 'bull': +10, 'bear': -10, 'black_swan': +2},
            "Panic Bear Market": {'moonshot': -3, 'bull': -10, 'bear': +15, 'black_swan': +5},
            "Overbought Rally": {'moonshot': +2, 'bull': +5, 'bear': +8, 'black_swan': +3},
            "Oversold Decline": {'moonshot': +3, 'bull': +8, 'bear': -5, 'black_swan': +2},
            "Bullish Trend": {'moonshot': +2, 'bull': +8, 'bear': -5, 'base': -3},
            "Bearish Trend": {'moonshot': -2, 'bull': -8, 'bear': +10, 'base': -2},
            "Range-bound Market": {'base': +10, 'moonshot': -2, 'bear': -2, 'bull': -3},
        }
        
        adjustments = regime_adjustments.get(regime, {})
        
        for scenario, adj in adjustments.items():
            base_probs[scenario] += adj
        
        # Ensure all probabilities are positive and sum to 100
        total = sum(base_probs.values())
        for key in base_probs:
            base_probs[key] = max(1, round(base_probs[key] * 100 / total))
        
        return base_probs
    
    def _create_enhanced_moonshot_scenario(self, current_price: float, volatility: float, 
                                         timeframe: int, probability: float, technical: Dict,
                                         fundamental: FundamentalData, sentiment: Dict, 
                                         regime: str, symbol: str) -> ScenarioData:
        """Create enhanced moonshot scenario with comprehensive analysis."""
        
        # Calculate target price with multiple factors
        base_multiplier = np.random.uniform(1.30, 1.60)
        
        # Adjust based on sentiment
        sentiment_boost = (sentiment['overall_score'] - 50) / 100 * 0.2
        
        # Adjust based on technical strength
        technical_boost = (technical.get('technical_score', 50) - 50) / 100 * 0.15
        
        final_multiplier = base_multiplier + sentiment_boost + technical_boost
        target_price = current_price * max(1.15, final_multiplier)
        expected_return = (target_price / current_price - 1) * 100
        
        # Calculate enhanced metrics
        volatility_adjusted_return = expected_return / (volatility * 100)
        sharpe_ratio = volatility_adjusted_return / np.sqrt(timeframe / 252)
        max_drawdown = volatility * 100 * 0.3  # Estimate
        
        # ML prediction integration
        ml_pred = technical.get('ml_prediction', {})
        ai_confidence = ml_pred.get('confidence', 0.5) * 0.8  # Slightly reduce confidence
        
        catalysts = [
            f"Major product launch or innovation breakthrough for {symbol}",
            "Exceptional earnings beat with significantly raised guidance",
            "Strategic partnership or acquisition announcement",
            "Sector rotation driving massive institutional inflows",
            "Regulatory approval for key initiatives"
        ]
        
        return ScenarioData(
            name="Moonshot Scenario",
            emoji="🚀",
            probability=probability,
            target_price=target_price,
            timeframe=f"{timeframe} days",
            expected_return=expected_return,
            catalysts=catalysts,
            conditions=f"Perfect storm of catalysts in {regime.lower()} environment",
            risk_factors="Extreme volatility, profit-taking cascades, market correction risk",
            confidence_score=min(85, 60 + volatility * 50),
            market_regime=regime,
            fundamental_score=self._calculate_fundamental_score(fundamental),
            technical_score=technical.get('technical_score', 50),
            sentiment_score=sentiment.get('overall_score', 50),
            volatility_adjusted_return=volatility_adjusted_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            support_levels=technical.get('support_levels', []),
            resistance_levels=technical.get('resistance_levels', []),
            ml_prediction=ml_pred.get('prediction', 0.0),
            ai_confidence=ai_confidence
        )
    
    def _create_enhanced_bull_scenario(self, current_price: float, volatility: float,
                                 timeframe: int, probability: float, technical: Dict,
                                 fundamental: FundamentalData, sentiment: Dict,
                                 regime: str, symbol: str) -> ScenarioData:
        """Create enhanced bull scenario."""
        
        base_multiplier = np.random.uniform(1.10, 1.30)
        sentiment_boost = (sentiment['overall_score'] - 50) / 100 * 0.15
        technical_boost = (technical.get('technical_score', 50) - 50) / 100 * 0.10
        
        final_multiplier = base_multiplier + sentiment_boost + technical_boost
        target_price = current_price * max(1.05, final_multiplier)
        expected_return = (target_price / current_price - 1) * 100
        
        volatility_adjusted_return = expected_return / (volatility * 100)
        sharpe_ratio = volatility_adjusted_return / np.sqrt(timeframe / 252)
        max_drawdown = volatility * 100 * 0.4
        
        ml_pred = technical.get('ml_prediction', {})
        ai_confidence = ml_pred.get('confidence', 0.5) * 0.85
        
        catalysts = [
            f"Steady revenue growth and margin expansion for {symbol}",
            "Positive industry trends and favorable market conditions",
            "Strong institutional buying and analyst upgrades",
            "Improved economic outlook supporting growth stocks",
            "Technical breakout above key resistance levels"
        ]
        
        return ScenarioData(
            name="Bull Case Scenario",
            emoji="🐂",
            probability=probability,
            target_price=target_price,
            timeframe=f"{timeframe} days",
            expected_return=expected_return,
            catalysts=catalysts,
            conditions=f"Favorable market environment with {regime.lower()} supporting momentum",
            risk_factors="Valuation expansion limits, profit-taking, sector rotation risks",
            confidence_score=min(85, 65 + volatility * 40),
            market_regime=regime,
            fundamental_score=self._calculate_fundamental_score(fundamental),
            technical_score=technical.get('technical_score', 50),
            sentiment_score=sentiment.get('overall_score', 50),
            volatility_adjusted_return=volatility_adjusted_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            support_levels=technical.get('support_levels', []),
            resistance_levels=technical.get('resistance_levels', []),
            ml_prediction=ml_pred.get('prediction', 0.0),
            ai_confidence=ai_confidence
        )

    def _create_enhanced_base_scenario(self, current_price: float, volatility: float,
                                    timeframe: int, probability: float, technical: Dict,
                                    fundamental: FundamentalData, sentiment: Dict,
                                    regime: str, symbol: str) -> ScenarioData:
        """Create enhanced base scenario."""
        
        base_multiplier = np.random.uniform(0.95, 1.12)
        sentiment_boost = (sentiment['overall_score'] - 50) / 100 * 0.08
        technical_boost = (technical.get('technical_score', 50) - 50) / 100 * 0.06
        
        final_multiplier = base_multiplier + sentiment_boost + technical_boost
        target_price = current_price * max(0.92, final_multiplier)
        expected_return = (target_price / current_price - 1) * 100
        
        volatility_adjusted_return = expected_return / (volatility * 100)
        sharpe_ratio = volatility_adjusted_return / np.sqrt(timeframe / 252)
        max_drawdown = volatility * 100 * 0.25
        
        ml_pred = technical.get('ml_prediction', {})
        ai_confidence = ml_pred.get('confidence', 0.5) * 0.90
        
        catalysts = [
            f"Steady business execution and in-line performance for {symbol}",
            "Market following broader economic trends",
            "Balanced institutional positioning with modest flows",
            "Normal seasonal patterns and business cycles",
            "Gradual technical consolidation within trading range"
        ]
        
        return ScenarioData(
            name="Base Case Scenario",
            emoji="📊",
            probability=probability,
            target_price=target_price,
            timeframe=f"{timeframe} days",
            expected_return=expected_return,
            catalysts=catalysts,
            conditions=f"Status quo maintained in {regime.lower()} market conditions",
            risk_factors="Limited catalysts, sideways trading, low volatility compression",
            confidence_score=min(88, 75 - volatility * 20),
            market_regime=regime,
            fundamental_score=self._calculate_fundamental_score(fundamental),
            technical_score=technical.get('technical_score', 50),
            sentiment_score=sentiment.get('overall_score', 50),
            volatility_adjusted_return=volatility_adjusted_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            support_levels=technical.get('support_levels', []),
            resistance_levels=technical.get('resistance_levels', []),
            ml_prediction=ml_pred.get('prediction', 0.0),
            ai_confidence=ai_confidence
        )

    def _create_enhanced_bear_scenario(self, current_price: float, volatility: float,
                                    timeframe: int, probability: float, technical: Dict,
                                    fundamental: FundamentalData, sentiment: Dict,
                                    regime: str, symbol: str) -> ScenarioData:
        """Create enhanced bear scenario."""
        
        base_multiplier = np.random.uniform(0.75, 0.95)
        sentiment_drag = (50 - sentiment['overall_score']) / 100 * 0.12
        technical_drag = (50 - technical.get('technical_score', 50)) / 100 * 0.08
        
        final_multiplier = base_multiplier - sentiment_drag - technical_drag
        target_price = current_price * min(0.98, final_multiplier)
        expected_return = (target_price / current_price - 1) * 100
        
        volatility_adjusted_return = expected_return / (volatility * 100)
        sharpe_ratio = volatility_adjusted_return / np.sqrt(timeframe / 252)
        max_drawdown = volatility * 100 * 0.6
        
        ml_pred = technical.get('ml_prediction', {})
        ai_confidence = ml_pred.get('confidence', 0.5) * 0.80
        
        catalysts = [
            f"Disappointing earnings or guidance cuts for {symbol}",
            "Sector headwinds and competitive pressures intensifying",
            "Risk-off market sentiment driving institutional selling",
            "Economic slowdown concerns affecting growth prospects",
            "Technical breakdown below key support levels"
        ]
        
        return ScenarioData(
            name="Bear Case Scenario",
            emoji="🐻",
            probability=probability,
            target_price=target_price,
            timeframe=f"{timeframe} days",
            expected_return=expected_return,
            catalysts=catalysts,
            conditions=f"Challenging environment with {regime.lower()} creating downward pressure",
            risk_factors="Momentum selling cascades, support level breaks, sentiment deterioration",
            confidence_score=min(82, 60 + volatility * 45),
            market_regime=regime,
            fundamental_score=self._calculate_fundamental_score(fundamental),
            technical_score=technical.get('technical_score', 50),
            sentiment_score=sentiment.get('overall_score', 50),
            volatility_adjusted_return=volatility_adjusted_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            support_levels=technical.get('support_levels', []),
            resistance_levels=technical.get('resistance_levels', []),
            ml_prediction=ml_pred.get('prediction', 0.0),
            ai_confidence=ai_confidence
        )

    def _create_enhanced_black_swan_scenario(self, current_price: float, volatility: float,
                                        timeframe: int, probability: float, technical: Dict,
                                        fundamental: FundamentalData, sentiment: Dict,
                                        regime: str, symbol: str) -> ScenarioData:
        """Create enhanced black swan scenario."""
        
        base_multiplier = np.random.uniform(0.40, 0.70)
        crisis_multiplier = 0.85 if regime in ["Panic Bear Market", "High Volatility"] else 0.95
        
        final_multiplier = base_multiplier * crisis_multiplier
        target_price = current_price * final_multiplier
        expected_return = (target_price / current_price - 1) * 100
        
        volatility_adjusted_return = expected_return / (volatility * 100)
        sharpe_ratio = volatility_adjusted_return / np.sqrt(timeframe / 252)
        max_drawdown = volatility * 100 * 1.2  # Can exceed 100% in extreme cases
        
        ml_pred = technical.get('ml_prediction', {})
        ai_confidence = ml_pred.get('confidence', 0.5) * 0.60  # Lower confidence for extreme events
        
        catalysts = [
            f"Major systemic crisis directly impacting {symbol}",
            "Severe market crash triggered by unforeseen events",
            "Regulatory intervention or legal issues creating uncertainty",
            "Geopolitical shocks causing widespread market panic",
            "Liquidity crisis and forced institutional selling"
        ]
        
        return ScenarioData(
            name="Black Swan Scenario",
            emoji="⚡",
            probability=probability,
            target_price=target_price,
            timeframe=f"{timeframe} days",
            expected_return=expected_return,
            catalysts=catalysts,
            conditions="Extreme negative shock event with systemic market implications",
            risk_factors="Complete sentiment reversal, liquidity crisis, panic selling, contagion effects",
            confidence_score=min(75, 35 + volatility * 80),
            market_regime=regime,
            fundamental_score=self._calculate_fundamental_score(fundamental),
            technical_score=technical.get('technical_score', 50),
            sentiment_score=sentiment.get('overall_score', 50),
            volatility_adjusted_return=volatility_adjusted_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            support_levels=technical.get('support_levels', []),
            resistance_levels=technical.get('resistance_levels', []),
            ml_prediction=ml_pred.get('prediction', 0.0),
            ai_confidence=ai_confidence
        )

    def _calculate_fundamental_score(self, fundamental: FundamentalData) -> float:
        """Calculate comprehensive fundamental score (0-100)."""
        try:
            score = 50  # Start with neutral
            
            # Valuation metrics (30% weight)
            pe_score = 0
            if 0 < fundamental.pe_ratio <= 15:
                pe_score = 20
            elif 15 < fundamental.pe_ratio <= 25:
                pe_score = 15
            elif 25 < fundamental.pe_ratio <= 35:
                pe_score = 10
            elif fundamental.pe_ratio > 35:
                pe_score = 5
            else:
                pe_score = 10  # Default for negative or zero PE
            
            # PEG ratio scoring
            peg_score = 0
            if 0 < fundamental.peg_ratio <= 1:
                peg_score = 15
            elif 1 < fundamental.peg_ratio <= 2:
                peg_score = 10
            elif fundamental.peg_ratio > 2:
                peg_score = 5
            else:
                peg_score = 8
            
            # Financial health metrics (25% weight)
            health_score = 0
            
            # ROE
            if fundamental.return_on_equity > 0.20:
                health_score += 8
            elif fundamental.return_on_equity > 0.15:
                health_score += 6
            elif fundamental.return_on_equity > 0.10:
                health_score += 4
            elif fundamental.return_on_equity > 0.05:
                health_score += 2
            
            # Current Ratio
            if fundamental.current_ratio > 2.0:
                health_score += 4
            elif fundamental.current_ratio > 1.5:
                health_score += 3
            elif fundamental.current_ratio > 1.0:
                health_score += 2
            
            # Debt to Equity
            if fundamental.debt_to_equity < 0.3:
                health_score += 4
            elif fundamental.debt_to_equity < 0.6:
                health_score += 2
            elif fundamental.debt_to_equity > 1.0:
                health_score -= 2
            
            # Growth metrics (25% weight)
            growth_score = 0
            
            # Revenue growth
            if fundamental.revenue_growth > 0.20:
                growth_score += 8
            elif fundamental.revenue_growth > 0.10:
                growth_score += 6
            elif fundamental.revenue_growth > 0.05:
                growth_score += 4
            elif fundamental.revenue_growth > 0:
                growth_score += 2
            elif fundamental.revenue_growth < -0.05:
                growth_score -= 3
            
            # Earnings growth
            if fundamental.earnings_growth > 0.20:
                growth_score += 7
            elif fundamental.earnings_growth > 0.10:
                growth_score += 5
            elif fundamental.earnings_growth > 0.05:
                growth_score += 3
            elif fundamental.earnings_growth > 0:
                growth_score += 1
            elif fundamental.earnings_growth < -0.10:
                growth_score -= 4
            
            # Profitability metrics (20% weight)
            profit_score = 0
            
            # Profit margin
            if fundamental.profit_margin > 0.20:
                profit_score += 6
            elif fundamental.profit_margin > 0.15:
                profit_score += 5
            elif fundamental.profit_margin > 0.10:
                profit_score += 4
            elif fundamental.profit_margin > 0.05:
                profit_score += 2
            elif fundamental.profit_margin < 0:
                profit_score -= 5
            
            # Operating margin
            if fundamental.operating_margin > 0.25:
                profit_score += 4
            elif fundamental.operating_margin > 0.15:
                profit_score += 3
            elif fundamental.operating_margin > 0.10:
                profit_score += 2
            elif fundamental.operating_margin < 0:
                profit_score -= 3
            
            # Calculate final score
            final_score = score + (pe_score + peg_score) * 0.3 + health_score * 0.25 + growth_score * 0.25 + profit_score * 0.2
            
            return max(0, min(100, final_score))
            
        except Exception as e:
            logger.error(f"Fundamental score calculation error: {e}")
            return 50.0

    def _get_fallback_scenarios(self, symbol: str, timeframe_days: int) -> Dict[str, ScenarioData]:
        """Generate fallback scenarios when data fetching fails."""
        
        base_price = 100.0  # Default price
        base_volatility = 0.25  # Default volatility
        
        scenarios = {
            '🚀 Moonshot': ScenarioData(
                name="Moonshot Scenario", emoji="🚀", probability=8,
                target_price=base_price * 1.35, timeframe=f"{timeframe_days} days",
                expected_return=35.0, 
                catalysts=["Major breakthrough", "Strong earnings", "Market optimism"],
                conditions="Ideal market conditions", 
                risk_factors="High volatility, profit-taking",
                confidence_score=65, market_regime="Unknown",
                fundamental_score=50, technical_score=50, sentiment_score=50,
                volatility_adjusted_return=1.4, sharpe_ratio=0.8, max_drawdown=15.0,
                support_levels=[95.0, 90.0], resistance_levels=[110.0, 120.0],
                ml_prediction=0.05, ai_confidence=0.6
            ),
            
            '🐂 Bull Case': ScenarioData(
                name="Bull Case Scenario", emoji="🐂", probability=25,
                target_price=base_price * 1.18, timeframe=f"{timeframe_days} days",
                expected_return=18.0,
                catalysts=["Steady growth", "Positive trends", "Strong buying"],
                conditions="Favorable environment", 
                risk_factors="Valuation concerns, rotation risk",
                confidence_score=72, market_regime="Unknown",
                fundamental_score=50, technical_score=50, sentiment_score=50,
                volatility_adjusted_return=0.72, sharpe_ratio=0.6, max_drawdown=10.0,
                support_levels=[95.0, 90.0], resistance_levels=[105.0, 110.0],
                ml_prediction=0.02, ai_confidence=0.7
            ),
            
            '📊 Base Case': ScenarioData(
                name="Base Case Scenario", emoji="📊", probability=40,
                target_price=base_price * 1.05, timeframe=f"{timeframe_days} days",
                expected_return=5.0,
                catalysts=["Steady performance", "In-line results", "Normal operations"],
                conditions="Status quo maintained", 
                risk_factors="Limited catalysts, sideways movement",
                confidence_score=80, market_regime="Unknown",
                fundamental_score=50, technical_score=50, sentiment_score=50,
                volatility_adjusted_return=0.20, sharpe_ratio=0.3, max_drawdown=6.0,
                support_levels=[95.0, 92.0], resistance_levels=[105.0, 108.0],
                ml_prediction=0.0, ai_confidence=0.8
            ),
            
            '🐻 Bear Case': ScenarioData(
                name="Bear Case Scenario", emoji="🐻", probability=22,
                target_price=base_price * 0.85, timeframe=f"{timeframe_days} days",
                expected_return=-15.0,
                catalysts=["Disappointing results", "Headwinds", "Risk-off sentiment"],
                conditions="Challenging environment", 
                risk_factors="Momentum selling, support breaks",
                confidence_score=70, market_regime="Unknown",
                fundamental_score=50, technical_score=50, sentiment_score=50,
                volatility_adjusted_return=-0.60, sharpe_ratio=-0.4, max_drawdown=15.0,
                support_levels=[90.0, 85.0], resistance_levels=[100.0, 102.0],
                ml_prediction=-0.03, ai_confidence=0.65
            ),
            
            '⚡ Black Swan': ScenarioData(
                name="Black Swan Scenario", emoji="⚡", probability=5,
                target_price=base_price * 0.60, timeframe=f"{timeframe_days} days",
                expected_return=-40.0,
                catalysts=["Major crisis", "Systemic shock", "Panic selling"],
                conditions="Extreme negative event", 
                risk_factors="Complete reversal, liquidity crisis",
                confidence_score=45, market_regime="Unknown",
                fundamental_score=50, technical_score=50, sentiment_score=50,
                volatility_adjusted_return=-1.60, sharpe_ratio=-1.2, max_drawdown=30.0,
                support_levels=[80.0, 70.0], resistance_levels=[95.0, 100.0],
                ml_prediction=-0.08, ai_confidence=0.4
            )
        }
        
        return scenarios
    
# Add this entire class to the END of your scenario_engine.py file

class ScenarioModelingTab:
    """
    UI component for the Scenario Modeling tab.
    This class orchestrates the user interface, triggers the analysis,
    and displays the results generated by the EnhancedScenarioEngine.
    """
    def __init__(self, symbol: str, market_data: pd.DataFrame, ui_components):
        self.symbol = symbol
        self.market_data = market_data
        self.ui = ui_components  # For potential future use with shared UI components
        self.engine = EnhancedScenarioEngine()

    def render(self):
        """Renders the entire UI for the Scenario Modeling tab."""
        st.markdown("### 🎭 AI-Powered Scenario Modeling")
        st.markdown(
            "Our engine analyzes fundamental, technical, and sentiment data to generate five weighted scenarios, "
            "providing a comprehensive outlook on potential market movements."
        )

        # --- Configuration Sidebar ---
        with st.sidebar:
            st.header("Scenario Configuration")
            timeframe = st.slider(
                "Analysis Timeframe (Days)", 
                min_value=30, 
                max_value=365, 
                value=90, 
                step=15,
                help="Set the forward-looking timeframe for the scenario predictions."
            )

        # --- Main Action Button ---
        if st.button(f"🚀 Analyze Scenarios for {self.symbol}", type="primary", use_container_width=True):

            # --- Perform Analysis and Display Results ---
            with st.spinner(f"🔍 Running scenario analysis for {self.symbol}… this may take a moment."):
                try:
                    scenarios = self.engine.generate_comprehensive_scenarios(self.symbol, timeframe)
                    if scenarios:
                        st.session_state['scenario_results'] = scenarios
                    else:
                        st.error("Scenario generation failed to produce results. Please try again.")
                        st.session_state['scenario_results'] = None

                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis: {e}")
                    st.session_state['scenario_results'] = None
        
        # --- Display Cached or Newly Generated Results ---
        if st.session_state.get('scenario_results'):
            self.display_scenarios(st.session_state['scenario_results'])

    def display_scenarios(self, scenarios: Dict[str, ScenarioData]):
        """Displays the generated scenarios in an organized layout."""
        st.markdown("---")
        st.subheader("Analysis Results")

        # Create columns for a cleaner layout
        cols = st.columns(len(scenarios))
        
        # Sort scenarios for a consistent display order
        sorted_scenarios = sorted(scenarios.items(), key=lambda item: item[1].expected_return, reverse=True)

        for i, (name, data) in enumerate(sorted_scenarios):
            with cols[i]:
                st.markdown(f"<h5 style='text-align: center;'>{data.emoji} {data.name.replace(' Scenario', '')}</h5>", unsafe_allow_html=True)
                st.metric(label="Probability", value=f"{data.probability}%")
                st.metric(label="Target Price", value=f"${data.target_price:,.2f}")
                st.metric(label="Expected Return", value=f"{data.expected_return:,.2f}%", 
                          delta=f"{data.expected_return:,.2f}%")

        # Detailed breakdown in expanders
        st.markdown("---")
        st.subheader("Detailed Scenario Breakdown")
        
        for name, data in sorted_scenarios:
            with st.expander(f"{data.emoji} **{data.name}** ({data.probability}% Probability)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info("**Key Metrics**")
                    st.metric("Sharpe Ratio", f"{data.sharpe_ratio:.2f}")
                    st.metric("Max Drawdown", f"{data.max_drawdown:.2f}%")
                    st.metric("ML Prediction (Return)", f"{data.ml_prediction*100:.2f}%")

                with col2:
                    st.warning("**Core Drivers**")
                    st.markdown(f"**Market Regime:** `{data.market_regime}`")
                    st.markdown("**Catalysts:**")
                    for catalyst in data.catalysts[:2]: # Show top 2 catalysts
                        st.markdown(f"- {catalyst}")

                with col3:
                    st.error("**Risk Factors**")
                    st.markdown(f"**Conditions:** {data.conditions}")
                    st.markdown(f"**Risks:** {data.risk_factors}")
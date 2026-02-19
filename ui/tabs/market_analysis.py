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
import io
import warnings
warnings.filterwarnings('ignore')

from config import config

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

def generate_comprehensive_analysis_report(llm, symbol, patterns_detected, 
                                         enhanced_data, fundamental_data, 
                                         sentiment_data, fibonacci_levels):
    """Generate a comprehensive analysis report using all available data."""
    
    if not llm:
        return "LLM not initialized."
    
    # Format patterns
    patterns_str = "\n".join([
        f"- **{p['name']}** ({p['type']}) on {p['date'].strftime('%Y-%m-%d')}: {p['description']} (Confidence: {p.get('confidence', 'N/A')}%)"
        for p in patterns_detected
    ]) if patterns_detected else "No significant patterns detected."
    
    # Get latest indicators
    latest = enhanced_data.iloc[-1]
    
    # Format technical indicators with safe value checking
    def safe_format(value, decimals=2):
        """Safely format numeric values."""
        if isinstance(value, (int, float)) and not np.isnan(value):
            return f"{value:.{decimals}f}"
        return "N/A"
    
    technical_str = f"""
**Trend & Momentum:**
- RSI (14): {safe_format(latest.get('RSI_14'))}
- MACD Line: {safe_format(latest.get('MACD_12_26_9'))}
- MACD Signal: {safe_format(latest.get('MACDs_12_26_9'))}
- ADX (14): {safe_format(latest.get('ADX_14'))}
- Stochastic %K: {safe_format(latest.get('STOCHk_14_3_3'))}

**Volatility:**
- ATR (14): {safe_format(latest.get('ATR_14'))}
"""
    
    # Add Bollinger Band position if available
    bb_upper = latest.get('BBU_20_2.0')
    bb_lower = latest.get('BBL_20_2.0')
    current_close = latest.get('Close')
    
    if all(isinstance(x, (int, float)) and not np.isnan(x) for x in [bb_upper, bb_lower, current_close]) and bb_upper != bb_lower:
        bb_position = ((current_close - bb_lower) / (bb_upper - bb_lower) * 100)
        technical_str += f"- Bollinger Band Position: {bb_position:.1f}%\n"
    
    # Add volume indicators
    technical_str += f"""
**Volume:**
- OBV: {safe_format(latest.get('OBV'), 0) if isinstance(latest.get('OBV'), (int, float)) else 'N/A'}
- CMF (20): {safe_format(latest.get('CMF_20'))}
- A/D Line: {safe_format(latest.get('AD'), 0) if isinstance(latest.get('AD'), (int, float)) else 'N/A'}
"""
    
    # Format fundamental data
    fundamental_str = f"""
**Company Profile:**
- Sector: {fundamental_data.get('sector', 'N/A')}
- Industry: {fundamental_data.get('industry', 'N/A')}
- Market Cap: {fundamental_data.get('market_cap', 'N/A')}

**Valuation Metrics:**
- P/E Ratio: {fundamental_data.get('pe_ratio', 'N/A')}
- P/B Ratio: {fundamental_data.get('price_to_book', 'N/A')}
- P/S Ratio: {fundamental_data.get('price_to_sales', 'N/A')}
- EPS: {fundamental_data.get('eps', 'N/A')}
- Dividend Yield: {fundamental_data.get('dividend_yield', 'N/A')}%
- Debt/Equity: {fundamental_data.get('debt_to_equity', 'N/A')}
""" if fundamental_data else "**Fundamental data not available**"
    
    # Format sentiment data
    if sentiment_data and isinstance(sentiment_data, list):
        # Extract sentiment from news list
        headlines = [item.get('title', '') for item in sentiment_data[:5]]
        sentiments = [item.get('sentiment', 'neutral') for item in sentiment_data[:5]]
        
        # Calculate overall sentiment
        positive_count = sum(1 for s in sentiments if s == 'positive')
        negative_count = sum(1 for s in sentiments if s == 'negative')
        
        if positive_count > negative_count:
            overall_sentiment = "Positive"
        elif negative_count > positive_count:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        sentiment_str = f"""
**Market Sentiment:**
- Overall Sentiment: {overall_sentiment}
- Positive News: {positive_count}/{len(sentiments)}
- Negative News: {negative_count}/{len(sentiments)}
- Recent Headlines:
{chr(10).join([f"  • {headline[:80]}..." if len(headline) > 80 else f"  • {headline}" for headline in headlines])}
"""
    else:
        sentiment_str = "**Sentiment data not available**"
    
    # Format Fibonacci levels
    fib_str = "\n".join([f"- {level}: ${price:.2f}" for level, price in fibonacci_levels.items()]) if fibonacci_levels else "Not calculated"
    
    # Recent price action
    recent_data = enhanced_data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    prompt = f"""
**Act as a Senior Quantitative Analyst at a top-tier institutional investment firm.**

**COMPREHENSIVE ANALYSIS REQUEST**
**Security:** {symbol}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}
**Analysis Type:** Multi-Factor Technical, Fundamental & Sentiment Analysis

**SECTION 1: TECHNICAL ANALYSIS**

**Detected Candlestick Patterns:**
{patterns_str}

**Technical Indicators:**
{technical_str}

**Key Fibonacci Retracement Levels:**
{fib_str}

**Recent 5-Day Price Action:**
{recent_data.to_string()}

**SECTION 2: FUNDAMENTAL ANALYSIS**
{fundamental_str}

**SECTION 3: SENTIMENT ANALYSIS**
{sentiment_str}

**ANALYSIS REQUIREMENTS:**

1. **Technical Synopsis:** Synthesize the candlestick patterns, technical indicators, and price action into a cohesive technical outlook. Identify any confluences or divergences between different technical signals.

2. **Fundamental Context:** Assess how the fundamental metrics support or contradict the technical signals. Consider valuation levels relative to historical norms and sector peers.

3. **Sentiment Integration:** Analyze how current market sentiment aligns with technical and fundamental indicators. Identify any potential sentiment-driven catalysts or risks.

4. **Risk Assessment:** Identify key technical support/resistance levels, fundamental risks, and sentiment-driven volatility factors.

5. **Strategic Recommendations:** Provide specific actionable insights including:
   - Primary trading bias (bullish/bearish/neutral)
   - Key entry/exit levels
   - Risk management parameters
   - Time horizon considerations

6. **Confidence Assessment:** Rate your overall confidence in the analysis and identify the strongest supporting factors.

**OUTPUT FORMAT:**
Please structure your response with clear sections and conclude with a JSON summary.

**FINAL SUMMARY (JSON):**
At the very end, provide:
{{"signal": "Bullish/Bearish/Neutral", "confidence": XX, "time_horizon": "Short/Medium/Long", "key_catalyst": "Primary driver", "risk_level": "Low/Medium/High"}}
"""
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Comprehensive analysis failed: {e}"

def detect_fibonacci_levels(data: pd.DataFrame):
    """Detects Fibonacci retracement levels."""
    if data.empty:
        return {}
    
    max_price = data['High'].max()
    min_price = data['Low'].min()
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

def calculate_fibonacci_levels(data: pd.DataFrame, period: int = 50):
    """Calculate Fibonacci retracement levels based on recent high/low."""
    if len(data) < period:
        period = len(data)
    
    recent_data = data.tail(period)
    high = recent_data['High'].max()
    low = recent_data['Low'].min()
    diff = high - low
    
    if diff == 0:
        return {}
    
    levels = {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.500 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff,
        '100.0%': low
    }
    
    return levels

# ========================= EODHD API CLIENT =========================

class EODHDClient:
    """Enhanced EODHD API client with comprehensive real-time capabilities"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
        self.session = requests.Session()
        
    def _make_request(self, endpoint, params=None):
        """Make API request with proper error handling"""
        if params is None:
            params = {}
        params['api_token'] = self.api_key
        params['fmt'] = 'json'
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"EODHD API Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Data processing error: {str(e)}")
            return None
    
    def get_real_time_data(self, symbol):
        """Get real-time price data"""
        data = self._make_request(f"real-time/{symbol}")
        if data:
            return {
                'symbol': symbol,
                'price': data.get('close', 0),
                'change': data.get('change', 0),
                'change_p': data.get('change_p', 0),
                'volume': data.get('volume', 0),
                'timestamp': data.get('timestamp', int(time.time()))
            }
        return None
    
    def get_historical_data(self, symbol, period='1y', interval='d'):
        """Get historical data with flexible periods"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate start date based on period
        if period == '1d':
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            interval = '1m'
        elif period == '1w':
            start_date = (datetime.now() - timedelta(weeks=1)).strftime('%Y-%m-%d')
            interval = '1h'
        elif period == '1m':
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        elif period == '3m':
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        elif period == '6m':
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        elif period == '1y':
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        elif period == '2y':
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        params = {
            'from': start_date,
            'to': end_date,
            'period': interval
        }
        
        data = self._make_request(f"eod/{symbol}", params)
        if data:
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Adjusted_close', 'Volume']
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return pd.DataFrame()
    
    def get_fundamentals(self, symbol):
        """Get fundamental data"""
        data = self._make_request(f"fundamentals/{symbol}")
        if data:
            highlights = data.get('Highlights', {})
            valuation  = data.get('Valuation', {})
            technicals = data.get('Technicals', {})
            general    = data.get('General', {})

            def _float(val):
                try:
                    return float(val or 0)
                except (TypeError, ValueError):
                    return 0.0

            # Revenue and net income from most-recent yearly income statement
            revenue = net_income = 0.0
            yearly_is = data.get('Financials', {}).get('Income_Statement', {}).get('yearly', {})
            if yearly_is:
                try:
                    latest = max(yearly_is.keys())
                    revenue    = _float(yearly_is[latest].get('totalRevenue'))
                    net_income = _float(yearly_is[latest].get('netIncome'))
                except Exception:
                    pass

            # Debt-to-equity from most-recent yearly balance sheet
            debt_to_equity = 0.0
            yearly_bs = data.get('Financials', {}).get('Balance_Sheet', {}).get('yearly', {})
            if yearly_bs:
                try:
                    latest = max(yearly_bs.keys())
                    liab   = _float(yearly_bs[latest].get('totalLiabilities'))
                    equity = _float(yearly_bs[latest].get('totalStockholderEquity')) or 1
                    debt_to_equity = liab / equity
                except Exception:
                    pass

            return {
                'sector':          general.get('Sector', 'N/A') or 'N/A',
                'industry':        general.get('Industry', 'N/A') or 'N/A',
                'market_cap':      _float(highlights.get('MarketCapitalization')),
                'pe_ratio':        _float(highlights.get('PERatio')),
                'peg_ratio':       _float(highlights.get('PEGRatio')),
                'eps':             _float(highlights.get('EarningsShare')),
                'dividend_yield':  _float(highlights.get('DividendYield')) * 100,
                'revenue':         revenue,
                'net_income':      net_income,
                'ebitda':          _float(highlights.get('EBITDA')),
                'profit_margin':   _float(highlights.get('ProfitMargin')) * 100,
                'roe':             _float(highlights.get('ReturnOnEquityTTM')) * 100,
                'roa':             _float(highlights.get('ReturnOnAssetsTTM')) * 100,
                'price_to_book':   _float(valuation.get('PriceBookMRQ')),
                'price_to_sales':  _float(valuation.get('PriceSalesTTM')),
                'beta':            _float(technicals.get('Beta')),
                '52_week_high':    _float(technicals.get('52WeekHigh')),
                '52_week_low':     _float(technicals.get('52WeekLow')),
                'debt_to_equity':  debt_to_equity,
            }
        return {}
    
    def get_news(self, symbol, limit=50):
        """Get news data"""
        try:
            params = {'s': symbol, 'limit': limit}
            data = self._make_request("news", params)
            
            if data and isinstance(data, list):
                news_list = []
                for item in data:
                    if isinstance(item, dict):
                        news_list.append({
                            'title': str(item.get('title', '')),
                            'content': str(item.get('content', '')),
                            'date': str(item.get('date', '')),
                            'sentiment': str(item.get('sentiment', 'neutral'))
                        })
                return news_list
            return []
        except Exception as e:
            print(f"News API error: {str(e)}")
            return []

# ========================= ADVANCED PATTERN DETECTOR =========================

class AdvancedPatternDetector:
    """Comprehensive pattern detection with all major strategies"""
    
    @staticmethod
    def detect_candlestick_patterns(df):
        """Detect all major candlestick patterns"""
        patterns = []
        
        if len(df) < 5:
            return patterns
        
        for i in range(2, len(df) - 2):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            next1 = df.iloc[i+1] if i+1 < len(df) else current
            next2 = df.iloc[i+2] if i+2 < len(df) else current
            
            # Pattern detection logic
            patterns.extend(AdvancedPatternDetector._detect_single_candle_patterns(current, prev, df.index[i]))
            patterns.extend(AdvancedPatternDetector._detect_multi_candle_patterns(prev2, prev, current, next1, next2, df.index[i]))
        
        return patterns
    
    @staticmethod
    def _detect_single_candle_patterns(candle, prev_candle, date):
        """Detect single candlestick patterns"""
        patterns = []
        
        body = abs(candle['Close'] - candle['Open'])
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        total_range = candle['High'] - candle['Low']
        
        if total_range == 0:
            return patterns
        
        # Doji
        if body < (total_range * 0.1):
            patterns.append({
                'name': 'Doji',
                'type': 'Neutral',
                'date': date,
                'confidence': 85,
                'description': 'Indecision in the market, potential reversal signal'
            })
        
        # Hammer
        if (lower_shadow > body * 2 and upper_shadow < body * 0.5 and 
            candle['Close'] > candle['Open']):
            patterns.append({
                'name': 'Hammer',
                'type': 'Bullish',
                'date': date,
                'confidence': 80,
                'description': 'Bullish reversal pattern, buyers stepping in'
            })
        
        # Shooting Star
        if (upper_shadow > body * 2 and lower_shadow < body * 0.5 and 
            candle['Close'] < candle['Open']):
            patterns.append({
                'name': 'Shooting Star',
                'type': 'Bearish',
                'date': date,
                'confidence': 80,
                'description': 'Bearish reversal pattern, selling pressure'
            })
        
        # Marubozu (Bullish)
        if (body > total_range * 0.9 and candle['Close'] > candle['Open']):
            patterns.append({
                'name': 'Bullish Marubozu',
                'type': 'Bullish',
                'date': date,
                'confidence': 90,
                'description': 'Strong bullish sentiment, continuous buying'
            })
        
        # Marubozu (Bearish)
        if (body > total_range * 0.9 and candle['Close'] < candle['Open']):
            patterns.append({
                'name': 'Bearish Marubozu',
                'type': 'Bearish',
                'date': date,
                'confidence': 90,
                'description': 'Strong bearish sentiment, continuous selling'
            })
        
        # Spinning Top
        if (body < total_range * 0.3 and upper_shadow > body and lower_shadow > body):
            patterns.append({
                'name': 'Spinning Top',
                'type': 'Neutral',
                'date': date,
                'confidence': 70,
                'description': 'Market indecision, potential trend reversal'
            })
        
        return patterns
    
    @staticmethod
    def _detect_multi_candle_patterns(c1, c2, c3, c4, c5, date):
        """Detect multi-candlestick patterns"""
        patterns = []
        
        # Bullish Engulfing
        if (c2['Close'] < c2['Open'] and c3['Close'] > c3['Open'] and
            c3['Open'] < c2['Close'] and c3['Close'] > c2['Open']):
            patterns.append({
                'name': 'Bullish Engulfing',
                'type': 'Bullish',
                'date': date,
                'confidence': 85,
                'description': 'Strong bullish reversal, buyers taking control'
            })
        
        # Bearish Engulfing
        if (c2['Close'] > c2['Open'] and c3['Close'] < c3['Open'] and
            c3['Open'] > c2['Close'] and c3['Close'] < c2['Open']):
            patterns.append({
                'name': 'Bearish Engulfing',
                'type': 'Bearish',
                'date': date,
                'confidence': 85,
                'description': 'Strong bearish reversal, sellers taking control'
            })
        
        # Morning Star
        if (c1['Close'] < c1['Open'] and  # First candle bearish
            abs(c2['Close'] - c2['Open']) < (c1['High'] - c1['Low']) * 0.3 and  # Second candle small
            c3['Close'] > c3['Open'] and  # Third candle bullish
            c3['Close'] > (c1['Open'] + c1['Close']) / 2):  # Third closes above midpoint of first
            patterns.append({
                'name': 'Morning Star',
                'type': 'Bullish',
                'date': date,
                'confidence': 90,
                'description': 'Three-candle bullish reversal pattern'
            })
        
        # Evening Star
        if (c1['Close'] > c1['Open'] and  # First candle bullish
            abs(c2['Close'] - c2['Open']) < (c1['High'] - c1['Low']) * 0.3 and  # Second candle small
            c3['Close'] < c3['Open'] and  # Third candle bearish
            c3['Close'] < (c1['Open'] + c1['Close']) / 2):  # Third closes below midpoint of first
            patterns.append({
                'name': 'Evening Star',
                'type': 'Bearish',
                'date': date,
                'confidence': 90,
                'description': 'Three-candle bearish reversal pattern'
            })
        
        # Three White Soldiers
        if (c1['Close'] > c1['Open'] and c2['Close'] > c2['Open'] and c3['Close'] > c3['Open'] and
            c2['Close'] > c1['Close'] and c3['Close'] > c2['Close']):
            patterns.append({
                'name': 'Three White Soldiers',
                'type': 'Bullish',
                'date': date,
                'confidence': 88,
                'description': 'Strong bullish continuation pattern'
            })
        
        # Three Black Crows
        if (c1['Close'] < c1['Open'] and c2['Close'] < c2['Open'] and c3['Close'] < c3['Open'] and
            c2['Close'] < c1['Close'] and c3['Close'] < c2['Close']):
            patterns.append({
                'name': 'Three Black Crows',
                'type': 'Bearish',
                'date': date,
                'confidence': 88,
                'description': 'Strong bearish continuation pattern'
            })
        
        # Piercing Pattern
        if (c2['Close'] < c2['Open'] and c3['Close'] > c3['Open'] and
            c3['Open'] < c2['Low'] and c3['Close'] > (c2['Open'] + c2['Close']) / 2):
            patterns.append({
                'name': 'Piercing Pattern',
                'type': 'Bullish',
                'date': date,
                'confidence': 75,
                'description': 'Bullish reversal pattern'
            })
        
        # Dark Cloud Cover
        if (c2['Close'] > c2['Open'] and c3['Close'] < c3['Open'] and
            c3['Open'] > c2['High'] and c3['Close'] < (c2['Open'] + c2['Close']) / 2):
            patterns.append({
                'name': 'Dark Cloud Cover',
                'type': 'Bearish',
                'date': date,
                'confidence': 75,
                'description': 'Bearish reversal pattern'
            })
        
        return patterns
    
    @staticmethod
    def detect_chart_patterns(df):
        """Detect chart patterns (support, resistance, trends)"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Find support and resistance levels
        highs = df['High'].rolling(window=5, center=True).max()
        lows = df['Low'].rolling(window=5, center=True).min()
        
        resistance_levels = df[df['High'] == highs]['High'].dropna()
        support_levels = df[df['Low'] == lows]['Low'].dropna()
        
        # Add support/resistance patterns
        for date, level in resistance_levels.tail(5).items():
            patterns.append({
                'name': 'Resistance Level',
                'type': 'Neutral',
                'date': date,
                'confidence': 75,
                'price_level': level,
                'description': f'Resistance at ${level:.2f}'
            })
        
        for date, level in support_levels.tail(5).items():
            patterns.append({
                'name': 'Support Level',
                'type': 'Neutral',
                'date': date,
                'confidence': 75,
                'price_level': level,
                'description': f'Support at ${level:.2f}'
            })
        
        # Trend analysis
        recent_data = df.tail(20)
        if len(recent_data) >= 10:
            slope = np.polyfit(range(len(recent_data)), recent_data['Close'], 1)[0]
            if slope > 0.5:
                patterns.append({
                    'name': 'Uptrend',
                    'type': 'Bullish',
                    'date': recent_data.index[-1],
                    'confidence': 80,
                    'description': 'Strong upward trend detected'
                })
            elif slope < -0.5:
                patterns.append({
                    'name': 'Downtrend',
                    'type': 'Bearish',
                    'date': recent_data.index[-1],
                    'confidence': 80,
                    'description': 'Strong downward trend detected'
                })
        
        return patterns

# ========================= TECHNICAL ANALYSIS =========================

def calculate_advanced_indicators(df):
    """Calculate comprehensive technical indicators"""
    if df.empty:
        return df
    
    try:
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df

# ========================= CLEAN PATTERN VISUALIZATION =========================

def create_clean_pattern_chart(df, patterns, symbol, real_time_data=None):
    """Create a separate, clean chart specifically for pattern visualization"""
    
    # Create subplots for clean pattern display
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'{symbol} - Price Action with Detected Patterns',
            'Pattern Timeline & Confidence Scores'
        ),
        row_heights=[0.7, 0.3]
    )
    
    # Main candlestick chart (cleaner version)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing_line_color='#00FF88',
            decreasing_line_color='#FF4444',
            increasing_fillcolor='rgba(0, 255, 136, 0.8)',
            decreasing_fillcolor='rgba(255, 68, 68, 0.8)'
        ),
        row=1, col=1
    )
    
    # Add only essential moving averages for context
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                name='SMA 20',
                line=dict(color='#FFD700', width=2, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                name='SMA 50',
                line=dict(color='#4ECDC4', width=2, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Pattern visualization with clean organization
    pattern_colors = {
        'Bullish': '#00FF88',
        'Bearish': '#FF4444',
        'Neutral': '#FFD700'
    }
    
    pattern_shapes = {
        'Bullish': 'triangle-up',
        'Bearish': 'triangle-down',
        'Neutral': 'circle'
    }
    
    # Group patterns by type for better organization
    patterns_by_type = {'Bullish': [], 'Bearish': [], 'Neutral': []}
    for pattern in patterns:
        pattern_type = pattern.get('type', 'Neutral')
        if pattern_type in patterns_by_type:
            patterns_by_type[pattern_type].append(pattern)
    
    # Add pattern markers to main chart (clean version)
    y_offset_multiplier = {'Bullish': 1.03, 'Bearish': 0.97, 'Neutral': 1.01}
    
    for pattern_type, type_patterns in patterns_by_type.items():
        if not type_patterns:
            continue
        
        # Limit to most recent and highest confidence patterns
        sorted_patterns = sorted(type_patterns, 
                               key=lambda x: (x['date'], x.get('confidence', 0)), 
                               reverse=True)[:8]  # Show max 8 patterns per type
        
        dates = []
        prices = []
        hover_texts = []
        pattern_names = []
        
        for pattern in sorted_patterns:
            if pattern['date'] in df.index:
                base_price = df.loc[pattern['date'], 'High'] if pattern_type == 'Bullish' else df.loc[pattern['date'], 'Low']
                adjusted_price = base_price * y_offset_multiplier[pattern_type]
                
                dates.append(pattern['date'])
                prices.append(adjusted_price)
                pattern_names.append(pattern['name'])
                
                hover_text = (f"<b>{pattern['name']}</b><br>"
                            f"Type: {pattern['type']}<br>"
                            f"Date: {pattern['date'].strftime('%Y-%m-%d')}<br>"
                            f"Confidence: {pattern.get('confidence', 'N/A')}%<br>"
                            f"Price: ${base_price:.2f}")
                hover_texts.append(hover_text)
        
        if dates:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=prices,
                    mode='markers',
                    marker=dict(
                        size=14,
                        color=pattern_colors[pattern_type],
                        symbol=pattern_shapes[pattern_type],
                        line=dict(width=2, color='white'),
                        opacity=0.9
                    ),
                    name=f'{pattern_type} Patterns ({len(dates)})',
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts,
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Pattern timeline chart (bottom subplot)
    all_patterns_for_timeline = []
    for pattern_type, type_patterns in patterns_by_type.items():
        for pattern in type_patterns:
            all_patterns_for_timeline.append({
                'date': pattern['date'],
                'type': pattern_type,
                'name': pattern['name'],
                'confidence': pattern.get('confidence', 50),
                'color': pattern_colors[pattern_type]
            })
    
    # Sort by date for timeline
    all_patterns_for_timeline.sort(key=lambda x: x['date'])
    
    if all_patterns_for_timeline:
        # Create timeline scatter plot
        timeline_dates = [p['date'] for p in all_patterns_for_timeline]
        timeline_confidence = [p['confidence'] for p in all_patterns_for_timeline]
        timeline_colors = [p['color'] for p in all_patterns_for_timeline]
        timeline_names = [p['name'] for p in all_patterns_for_timeline]
        timeline_types = [p['type'] for p in all_patterns_for_timeline]
        
        fig.add_trace(
            go.Scatter(
                x=timeline_dates,
                y=timeline_confidence,
                mode='markers+lines',
                marker=dict(
                    size=12,
                    color=timeline_colors,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                line=dict(width=1, color='rgba(255,255,255,0.3)', dash='dot'),
                name='Pattern Confidence Timeline',
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Date: %{x}<br>'
                    'Confidence: %{y}%<br>'
                    'Type: %{customdata}<br>'
                    '<extra></extra>'
                ),
                text=timeline_names,
                customdata=timeline_types,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add confidence level lines
        fig.add_hline(y=80, line_dash="dash", line_color="green", 
                     annotation_text="High Confidence", row=2, col=1)
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Confidence", row=2, col=1)
        fig.add_hline(y=40, line_dash="dash", line_color="red", 
                     annotation_text="Low Confidence", row=2, col=1)
    
    # Add real-time data point if available
    if real_time_data:
        fig.add_trace(
            go.Scatter(
                x=[datetime.now()],
                y=[real_time_data['price']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='yellow',
                    symbol='star',
                    line=dict(width=3, color='black')
                ),
                name='Real-Time Price',
                hovertemplate=f'<b>Live Price</b><br>Price: ${real_time_data["price"]:.2f}<br>Change: {real_time_data["change_p"]:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Update layout for clean appearance
    fig.update_layout(
        title={
            'text': f"{symbol} - Clean Pattern Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        xaxis2_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="Confidence (%)",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(0,0,0,0.9)',
        font=dict(color='white', size=12),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # Customize axes
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.1)',
        showgrid=True,
        zeroline=False
    )
    
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.1)',
        showgrid=True,
        zeroline=False
    )
    
    # Set y-axis range for confidence chart
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    return fig

def create_pattern_summary_table(patterns):
    """Create a clean summary table of detected patterns"""
    
    if not patterns:
        return pd.DataFrame()
    
    # Process patterns for table display
    pattern_data = []
    for pattern in patterns:
        pattern_data.append({
            'Date': pattern['date'].strftime('%Y-%m-%d'),
            'Pattern Name': pattern['name'],
            'Type': pattern['type'],
            'Confidence': f"{pattern.get('confidence', 0)}%",
            'Description': pattern['description'][:50] + "..." if len(pattern['description']) > 50 else pattern['description']
        })
    
    # Create DataFrame and sort by date (most recent first)
    df = pd.DataFrame(pattern_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=False)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    return df

# ========================= DATA INTEGRATION =========================

def integrate_uploaded_data(symbol, uploaded_files):
    """Integrate uploaded data relevant to the selected symbol"""
    integrated_data = {}
    
    if not uploaded_files:
        return integrated_data
    
    try:
        for file_info in uploaded_files:
            file_name = file_info['name'].lower()
            content = file_info['content']
            
            # Check if file is relevant to the symbol
            if symbol.lower().replace('.us', '').replace('.', '') in file_name:
                file_type = file_info['type']
                
                # Process CSV files
                if file_name.endswith('.csv'):
                    try:
                        df = pd.read_csv(io.BytesIO(content))
                        if 'price' in df.columns.str.lower().tolist() or 'close' in df.columns.str.lower().tolist():
                            integrated_data['price_data'] = df
                        if 'volume' in df.columns.str.lower().tolist():
                            integrated_data['volume_data'] = df
                        if 'earnings' in file_name or 'revenue' in file_name:
                            integrated_data['financial_data'] = df
                    except Exception as e:
                        st.warning(f"Could not process CSV file {file_name}: {str(e)}")
                
                # Process Excel files
                elif file_name.endswith(('.xlsx', '.xls')):
                    try:
                        df = pd.read_excel(io.BytesIO(content))
                        if 'financial' in file_name or 'earnings' in file_name:
                            integrated_data['financial_data'] = df
                        elif 'analysis' in file_name or 'research' in file_name:
                            integrated_data['research_data'] = df
                    except Exception as e:
                        st.warning(f"Could not process Excel file {file_name}: {str(e)}")
                
                # Process text files for news/sentiment
                elif file_name.endswith('.txt'):
                    try:
                        text_content = content.decode('utf-8')
                        integrated_data['text_analysis'] = {
                            'content': text_content,
                            'length': len(text_content),
                            'file_name': file_name
                        }
                    except Exception as e:
                        st.warning(f"Could not process text file {file_name}: {str(e)}")
        
        return integrated_data
        
    except Exception as e:
        st.error(f"Error integrating uploaded data: {str(e)}")
        return {}

# ========================= MAIN CHART CREATION =========================

def create_main_technical_chart(df, symbol, real_time_data=None):
    """Create the main technical analysis chart without pattern clutter"""
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{symbol} - Price Action & Technical Indicators',
            'RSI & Stochastic',
            'MACD & Signal',
            'Volume Analysis'
        ),
        row_heights=[0.5, 0.2, 0.15, 0.15]
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
            increasing_line_color='#00FF88',
            decreasing_line_color='#FF4444',
            increasing_fillcolor='#00FF88',
            decreasing_fillcolor='#FF4444'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    colors = ['#FFD700', '#FF6B35', '#4ECDC4', '#45B7D1']
    mas = ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26']
    
    for i, ma in enumerate(mas):
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ma],
                    name=ma,
                    line=dict(color=colors[i], width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )
    
    # Add Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color='gray', dash='dash', width=1),
                opacity=0.6
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color='gray', dash='dash', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                opacity=0.6
            ),
            row=1, col=1
        )
    
    # Add real-time data point if available
    if real_time_data:
        fig.add_trace(
            go.Scatter(
                x=[datetime.now()],
                y=[real_time_data['price']],
                mode='markers',
                marker=dict(
                    size=15,
                    color='yellow',
                    symbol='star',
                    line=dict(width=3, color='black')
                ),
                name='Real-Time Price',
                hovertemplate=f'<b>Live Price</b><br>Price: ${real_time_data["price"]:.2f}<br>Change: {real_time_data["change_p"]:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Technical indicators subplot
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", row=2, col=1)
    
    if 'Stoch_K' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Stoch_K'], name='Stoch %K', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Stoch_D'], name='Stoch %D', line=dict(color='blue')),
            row=2, col=1
        )
    
    # MACD subplot
    if 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram', 
                   marker_color=colors, opacity=0.6),
            row=3, col=1
        )
    
    # Volume subplot
    volume_colors = ['green' if close >= open_price else 'red' 
                    for close, open_price in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', 
               marker_color=volume_colors, opacity=0.7),
        row=4, col=1
    )
    
    if 'Volume_SMA' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Volume_SMA'], name='Volume SMA', 
                      line=dict(color='orange', width=2)),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Comprehensive Technical Analysis",
        height=1000,
        showlegend=True,
        xaxis4_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="RSI / Stoch",
        yaxis3_title="MACD",
        yaxis4_title="Volume",
        template="plotly_dark",
        font=dict(size=12)
    )
    
    # Remove range slider for cleaner look
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig

# ========================= CLEAN PATTERN ANALYSIS COMPONENT =========================

def render_clean_pattern_analysis(enhanced_data, symbol, real_time_data=None):
    """Render clean pattern analysis with separate chart"""

    st.markdown("### 🎯 Clean Pattern Detection & Analysis")

    if st.button("🔍 Detect & Show Patterns (Clean Chart)", key="clean_pattern_detection"):
        with st.spinner("🎯 Detecting patterns and creating clean visualization..."):
            try:
                # Detect patterns
                candlestick_patterns = AdvancedPatternDetector.detect_candlestick_patterns(enhanced_data)
                chart_patterns = AdvancedPatternDetector.detect_chart_patterns(enhanced_data)
                all_patterns = candlestick_patterns + chart_patterns

                if all_patterns:
                    st.session_state.detected_patterns = all_patterns

                    # Create clean pattern chart
                    clean_fig = create_clean_pattern_chart(enhanced_data, all_patterns, symbol, real_time_data)

                    # Display the clean chart
                    st.plotly_chart(clean_fig, use_container_width=True)

                    # Pattern statistics
                    pattern_counts = {'Bullish': 0, 'Bearish': 0, 'Neutral': 0}
                    high_confidence_patterns = 0

                    for pattern in all_patterns:
                        pattern_counts[pattern['type']] += 1
                        if pattern.get('confidence', 0) >= 80:
                            high_confidence_patterns += 1

                    # Display statistics
                    st.markdown("#### 📊 Pattern Detection Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("🟢 Bullish", pattern_counts['Bullish'])

                    with col2:
                        st.metric("🔴 Bearish", pattern_counts['Bearish'])

                    with col3:
                        st.metric("🟡 Neutral", pattern_counts['Neutral'])

                    with col4:
                        st.metric("⭐ High Confidence", high_confidence_patterns)

                    # Pattern summary table
                    st.markdown("#### 📋 Pattern Details")

                    pattern_table = create_pattern_summary_table(all_patterns)
                    if not pattern_table.empty:
                        st.dataframe(
                            pattern_table,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Type": st.column_config.TextColumn(
                                    width="small",
                                ),
                                "Confidence": st.column_config.TextColumn(
                                    width="small",
                                ),
                                "Description": st.column_config.TextColumn(
                                    width="large",
                                )
                            }
                        )

                    st.success(f"✅ Detected {len(all_patterns)} patterns with clean visualization!")

                else:
                    st.info("ℹ️ No significant patterns detected in the current time frame.")

            except Exception as e:
                st.error(f"❌ Error detecting patterns: {str(e)}")

# ========================= MAIN RENDER FUNCTION =========================

def render():
    """Main render function for enhanced market analysis"""
    st.markdown("## 📊 Enhanced Market Analysis with Real-Time Data")
    
    # Get configuration
    api_key = config.get_eodhd_api_key()
    symbol = st.session_state.get('selected_symbol', 'AAPL.US')
    
    if not api_key:
        st.error("❌ EODHD API key not configured. Please check your .env file.")
        st.info("💡 Add EODHD_API_KEY to your environment variables to enable real-time features.")
        return
    
    # Initialize EODHD client
    eodhd = EODHDClient(api_key)

    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        period = st.selectbox(
            "Time Period",
            ["1d", "1w", "1m", "3m", "6m", "1y", "2y"],
            index=5,
            key="enhanced_period"
        )
    
    with col2:
        show_patterns = st.checkbox("Show Patterns", value=True, key="show_patterns_enhanced")
    
    with col3:
        show_real_time = st.checkbox("Real-Time Data", value=False, key="show_realtime_enhanced")
    
    with col4:
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False, key="auto_refresh_enhanced")
    
    # Auto refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Load data with progress tracking
    with st.spinner("🔄 Loading market data and analysis..."):
        progress_bar = st.progress(0, text="Fetching historical data...")
        
        # Get historical data
        market_data = eodhd.get_historical_data(symbol, period)
        progress_bar.progress(25, text="Calculating technical indicators...")
        
        if market_data.empty:
            st.error(f"❌ Could not load data for {symbol}. Please check the symbol.")
            return
        
        # Calculate technical indicators
        enhanced_data = calculate_advanced_indicators(market_data)
        fibonacci_levels = calculate_fibonacci_levels(enhanced_data)
        progress_bar.progress(50, text="Fetching real-time data...")
        
        # Real-time data
        real_time_data = None
        if show_real_time:
            real_time_data = eodhd.get_real_time_data(symbol)
        
        progress_bar.progress(100, text="Complete!")
        progress_bar.empty()
    
    # Display real-time info
    if real_time_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            change_color = "green" if real_time_data['change'] >= 0 else "red"
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #1e3c72, #2a5298); padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0;">💰 Live Price</h3>
                <h2 style="color: white; margin: 0.5rem 0;">${real_time_data['price']:.2f}</h2>
                <p style="color: {change_color}; margin: 0; font-weight: bold;">
                    {real_time_data['change']:+.2f} ({real_time_data['change_p']:+.2f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #134e5e, #71b280); padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0;">📊 Volume</h3>
                <h2 style="color: white; margin: 0.5rem 0;">{real_time_data['volume']:,.0f}</h2>
                <p style="color: #ccc; margin: 0;">Last updated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            latest_data = enhanced_data.iloc[-1]
            rsi = latest_data.get('RSI', 0)
            rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "orange"
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0;">📈 RSI</h3>
                <h2 style="color: white; margin: 0.5rem 0;">{rsi:.1f}</h2>
                <p style="color: {rsi_color}; margin: 0; font-weight: bold;">
                    {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            macd = latest_data.get('MACD', 0)
            macd_signal = latest_data.get('MACD_Signal', 0)
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            macd_color = "green" if macd > macd_signal else "red"
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #f093fb, #f5576c); padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0;">🎯 MACD</h3>
                <h2 style="color: white; margin: 0.5rem 0;">{macd:.3f}</h2>
                <p style="color: {macd_color}; margin: 0; font-weight: bold;">{macd_trend}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main chart
    st.markdown("### 📈 Interactive Technical Analysis Chart")
    
    if not enhanced_data.empty:
        fig = create_main_technical_chart(enhanced_data, symbol, real_time_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Clean pattern analysis section
    if show_patterns:
        st.markdown("---")
        render_clean_pattern_analysis(enhanced_data, symbol, real_time_data)
    
    # Technical analysis summary
    st.markdown("---")
    st.markdown("### 📊 Technical Analysis Summary")
    
    latest_data = enhanced_data.iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Moving Averages")
        
        price = latest_data['Close']
        sma_20 = latest_data.get('SMA_20', 0)
        sma_50 = latest_data.get('SMA_50', 0)
        ema_12 = latest_data.get('EMA_12', 0)
        
        ma_signals = []
        if price > sma_20:
            ma_signals.append("🟢 Price above SMA 20")
        else:
            ma_signals.append("🔴 Price below SMA 20")
        
        if price > sma_50:
            ma_signals.append("🟢 Price above SMA 50")
        else:
            ma_signals.append("🔴 Price below SMA 50")
        
        if sma_20 > sma_50:
            ma_signals.append("🟢 SMA 20 above SMA 50")
        else:
            ma_signals.append("🔴 SMA 20 below SMA 50")
        
        for signal in ma_signals:
            st.markdown(f"- {signal}")
    
    with col2:
        st.markdown("#### 🎯 Oscillators")
        
        rsi = latest_data.get('RSI', 50)
        stoch_k = latest_data.get('Stoch_K', 50)
        
        oscillator_signals = []
        
        if rsi > 70:
            oscillator_signals.append("🔴 RSI Overbought")
        elif rsi < 30:
            oscillator_signals.append("🟢 RSI Oversold")
        else:
            oscillator_signals.append("🟡 RSI Neutral")
        
        if stoch_k > 80:
            oscillator_signals.append("🔴 Stochastic Overbought")
        elif stoch_k < 20:
            oscillator_signals.append("🟢 Stochastic Oversold")
        else:
            oscillator_signals.append("🟡 Stochastic Neutral")
        
        for signal in oscillator_signals:
            st.markdown(f"- {signal}")
    
    # Additional data sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Fundamental data
        if st.button("📊 Load Fundamental Data", key="load_fundamentals"):
            with st.spinner("Loading fundamental data..."):
                fundamentals = eodhd.get_fundamentals(symbol)
                if fundamentals:
                    st.markdown("#### 💼 Fundamental Data")

                    def _fmt_large(v):
                        if v >= 1e12: return f"${v/1e12:.2f}T"
                        if v >= 1e9:  return f"${v/1e9:.2f}B"
                        if v >= 1e6:  return f"${v/1e6:.2f}M"
                        return f"${v:,.0f}"

                    mc  = fundamentals.get('market_cap', 0)
                    rev = fundamentals.get('revenue', 0)
                    ni  = fundamentals.get('net_income', 0)
                    ebi = fundamentals.get('ebitda', 0)

                    fund_df = pd.DataFrame([
                        {"Metric": "Sector",         "Value": fundamentals.get('sector', 'N/A')},
                        {"Metric": "Industry",        "Value": fundamentals.get('industry', 'N/A')},
                        {"Metric": "Market Cap",      "Value": _fmt_large(mc) if mc else "N/A"},
                        {"Metric": "Revenue (TTM)",   "Value": _fmt_large(rev) if rev else "N/A"},
                        {"Metric": "Net Income",      "Value": _fmt_large(ni) if ni else "N/A"},
                        {"Metric": "EBITDA",          "Value": _fmt_large(ebi) if ebi else "N/A"},
                        {"Metric": "P/E Ratio",       "Value": f"{fundamentals.get('pe_ratio', 0):.2f}"},
                        {"Metric": "PEG Ratio",       "Value": f"{fundamentals.get('peg_ratio', 0):.2f}"},
                        {"Metric": "P/B Ratio",       "Value": f"{fundamentals.get('price_to_book', 0):.2f}"},
                        {"Metric": "P/S Ratio",       "Value": f"{fundamentals.get('price_to_sales', 0):.2f}"},
                        {"Metric": "EPS",             "Value": f"${fundamentals.get('eps', 0):.2f}"},
                        {"Metric": "Dividend Yield",  "Value": f"{fundamentals.get('dividend_yield', 0):.2f}%"},
                        {"Metric": "Profit Margin",   "Value": f"{fundamentals.get('profit_margin', 0):.2f}%"},
                        {"Metric": "ROE",             "Value": f"{fundamentals.get('roe', 0):.2f}%"},
                        {"Metric": "ROA",             "Value": f"{fundamentals.get('roa', 0):.2f}%"},
                        {"Metric": "Debt/Equity",     "Value": f"{fundamentals.get('debt_to_equity', 0):.2f}"},
                        {"Metric": "Beta",            "Value": f"{fundamentals.get('beta', 0):.2f}"},
                        {"Metric": "52W High",        "Value": f"${fundamentals.get('52_week_high', 0):.2f}"},
                        {"Metric": "52W Low",         "Value": f"${fundamentals.get('52_week_low', 0):.2f}"},
                    ])

                    st.dataframe(fund_df, hide_index=True, use_container_width=True)
                else:
                    st.info("Fundamental data not available for this symbol.")
    
    with col2:
        # News and sentiment
        if st.button("📰 Load News & Sentiment", key="load_news"):
            with st.spinner("Loading news and sentiment data..."):
                try:
                    news_data = eodhd.get_news(symbol, limit=20)
                    if news_data and isinstance(news_data, list):
                        st.markdown("#### 📰 Latest News")
                        
                        # Show recent headlines
                        for i, article in enumerate(news_data[:5]):
                            if isinstance(article, dict):
                                # Safely get sentiment and title
                                sentiment = article.get('sentiment', 'neutral')
                                title = article.get('title', 'No title available')
                                
                                # Handle sentiment icon
                                if sentiment == 'positive':
                                    sentiment_icon = '🟢'
                                elif sentiment == 'negative':
                                    sentiment_icon = '🔴'
                                else:
                                    sentiment_icon = '🟡'
                                
                                # Display the headline
                                display_title = title[:100] + "..." if len(title) > 100 else title
                                st.markdown(f"{sentiment_icon} {display_title}")
                            else:
                                st.markdown(f"🟡 {str(article)[:100]}...")
                    else:
                        st.info("News data not available for this symbol.")
                except Exception as news_error:
                    st.error(f"Could not load news data: {str(news_error)}")
                    if st.checkbox("Show news error details", key="show_news_error"):
                        st.code(str(news_error))
    
    # Uploaded data integration
    uploaded_files = st.session_state.get('uploaded_files', [])
    if uploaded_files:
        st.markdown("---")
        st.markdown("### 📁 Uploaded Data Integration")
        
        integrated_data = integrate_uploaded_data(symbol, uploaded_files)
        
        if integrated_data:
            st.success(f"✅ Found {len(integrated_data)} relevant data sources for {symbol}")
            
            for data_type, data_content in integrated_data.items():
                with st.expander(f"📊 {data_type.replace('_', ' ').title()}", expanded=False):
                    if isinstance(data_content, pd.DataFrame):
                        st.dataframe(data_content.head(10), use_container_width=True)
                        st.info(f"Showing first 10 rows of {len(data_content)} total rows")
                    elif isinstance(data_content, dict):
                        for key, value in data_content.items():
                            st.markdown(f"**{key}:** {value}")
                    else:
                        st.text(str(data_content)[:500] + "..." if len(str(data_content)) > 500 else str(data_content))
        else:
            st.info(f"No uploaded files found relevant to {symbol}")
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 📤 Export Data & Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Technical Data", key="export_tech"):
            with st.spinner("Preparing technical data export…"):
                csv_data = enhanced_data.tail(100).to_csv()
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name=f"{symbol}_technical_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col2:
        patterns = st.session_state.get('detected_patterns', [])
        if patterns and st.button("🎯 Export Patterns", key="export_patterns"):
            with st.spinner("Preparing patterns export…"):
                pattern_df = pd.DataFrame(patterns)
                csv_data = pattern_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Patterns",
                data=csv_data,
                file_name=f"{symbol}_patterns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col3:
        if st.button("📈 Export Chart", key="export_chart"):
            st.info("Chart export functionality - would generate PNG/PDF of the current chart")

# ========================= DEMO DATA GENERATOR =========================

def generate_demo_data():
    """Generate demo data for fallback mode"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    demo_df = pd.DataFrame({
        'Open': prices + np.random.randn(len(dates)) * 0.1,
        'High': prices + np.abs(np.random.randn(len(dates)) * 0.5),
        'Low': prices - np.abs(np.random.randn(len(dates)) * 0.5),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return demo_df

def render_demo_fallback():
    """Render demo mode when APIs are unavailable"""
    st.markdown("### 📊 Demo Mode")
    st.info("Showing demo data while issues are resolved...")
    
    demo_df = generate_demo_data()
    demo_enhanced = calculate_advanced_indicators(demo_df)
    
    if not demo_enhanced.empty:
        # Show main chart
        fig = create_main_technical_chart(demo_enhanced, "DEMO")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show pattern detection demo
        st.markdown("---")
        st.markdown("### 🎯 Demo Pattern Detection")
        
        demo_patterns = AdvancedPatternDetector.detect_candlestick_patterns(demo_enhanced)
        if demo_patterns:
            pattern_fig = create_clean_pattern_chart(demo_enhanced, demo_patterns, "DEMO")
            st.plotly_chart(pattern_fig, use_container_width=True)
        
        st.warning("⚠️ This is demo data. Configure your EODHD API key for real market data.")

def render_error_info(error_msg, show_details_key="show_error_details_main"):
    """Render error information with troubleshooting tips"""
    st.error(f"❌ {error_msg}")
    
    if st.checkbox("Show detailed error information", key=show_details_key):
        st.code(str(error_msg))
    
    st.markdown("""
    ### 🔧 Troubleshooting Tips:
    1. **Check API Configuration**: Ensure EODHD_API_KEY is properly set in your .env file
    2. **Verify Symbol Format**: Use correct format (e.g., AAPL.US for US stocks)
    3. **Network Connection**: Ensure stable internet connection
    4. **API Limits**: Check if you've exceeded API rate limits
    5. **Symbol Availability**: Verify symbol exists on EODHD
    6. **Restart Application**: Sometimes a restart resolves configuration issues
    """)

# ========================= MAIN RENDER FUNCTION (NO RECURSION) =========================

def render():
    """Main render function for enhanced market analysis - NO RECURSION"""
    
    try:
        st.markdown("## 📊 Enhanced Market Analysis with Real-Time Data")
        
        # Get configuration
        api_key = config.get_eodhd_api_key()
        symbol = st.session_state.get('selected_symbol', 'AAPL.US')
        
        # Check API key first
        if not api_key:
            st.error("❌ EODHD API key not configured.")
            st.info("💡 Add EODHD_API_KEY to your environment variables to enable real-time features.")
            
            with st.expander("🔧 Configuration Help", expanded=True):
                st.markdown("""
                **Steps to configure EODHD API:**
                1. Create a `.env` file in your project root
                2. Add this line: `EODHD_API_KEY=your_api_key_here`
                3. Restart the application
                4. Your paid EODHD API key will enable all real-time features
                """)
            
            # Show demo mode
            render_demo_fallback()
            return
        
        # Initialize EODHD client
        eodhd = EODHDClient(api_key)
        # Control panel
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            period = st.selectbox(
                "Time Period",
                ["1d", "1w", "1m", "3m", "6m", "1y", "2y"],
                index=5,
                key="enhanced_period"
            )
        
        with col2:
            show_patterns = st.checkbox("Show Patterns", value=True, key="show_patterns_enhanced")
        
        with col3:
            show_real_time = st.checkbox("Real-Time Data", value=False, key="show_realtime_enhanced")
        
        with col4:
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=False, key="auto_refresh_enhanced")
        
        # Auto refresh functionality
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Load data with progress tracking
        try:
            with st.spinner("🔄 Loading market data and analysis..."):
                progress_bar = st.progress(0, text="Fetching historical data...")
                
                # Get historical data
                market_data = eodhd.get_historical_data(symbol, period)
                progress_bar.progress(25, text="Calculating technical indicators...")
                
                if market_data.empty:
                    progress_bar.empty()
                    st.error(f"❌ Could not load data for {symbol}. Please check the symbol.")
                    render_demo_fallback()
                    return
                
                from analysis.predictive import detect_anomalies, generate_forecast
                from analysis.reporting import generate_forecast_analysis_report
                # Calculate technical indicators
                enhanced_data = calculate_advanced_indicators(market_data)
                fibonacci_levels = calculate_fibonacci_levels(enhanced_data)
                progress_bar.progress(50, text="Fetching real-time data...")
                
                # Real-time data
                real_time_data = None
                if show_real_time:
                    try:
                        real_time_data = eodhd.get_real_time_data(symbol)
                    except Exception as rt_error:
                        st.warning(f"Could not fetch real-time data: {str(rt_error)}")
                
                progress_bar.progress(100, text="Complete!")
                progress_bar.empty()
        
        except Exception as data_error:
            render_error_info(f"Data loading error: {str(data_error)}", "show_data_error_details")
            render_demo_fallback()
            return
        
        # Display real-time info
        if real_time_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                change_color = "green" if real_time_data['change'] >= 0 else "red"
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #1e3c72, #2a5298); padding: 1rem; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">💰 Live Price</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">${real_time_data['price']:.2f}</h2>
                    <p style="color: {change_color}; margin: 0; font-weight: bold;">
                        {real_time_data['change']:+.2f} ({real_time_data['change_p']:+.2f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #134e5e, #71b280); padding: 1rem; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">📊 Volume</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">{real_time_data['volume']:,.0f}</h2>
                    <p style="color: #ccc; margin: 0;">Last updated</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                latest_data = enhanced_data.iloc[-1]
                rsi = latest_data.get('RSI', 0)
                rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "orange"
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">📈 RSI</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">{rsi:.1f}</h2>
                    <p style="color: {rsi_color}; margin: 0; font-weight: bold;">
                        {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                macd = latest_data.get('MACD', 0)
                macd_signal = latest_data.get('MACD_Signal', 0)
                macd_trend = "Bullish" if macd > macd_signal else "Bearish"
                macd_color = "green" if macd > macd_signal else "red"
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #f093fb, #f5576c); padding: 1rem; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">🎯 MACD</h3>
                    <h2 style="color: white; margin: 0.5rem 0;">{macd:.3f}</h2>
                    <p style="color: {macd_color}; margin: 0; font-weight: bold;">{macd_trend}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Main chart
        st.markdown("### 📈 Interactive Technical Analysis Chart")
        
        if not enhanced_data.empty:
            try:
                fig = create_main_technical_chart(enhanced_data, symbol, real_time_data)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as chart_error:
                st.error(f"Chart creation error: {str(chart_error)}")
                st.info("Showing simplified chart...")
                
                # Simple fallback chart
                simple_fig = go.Figure(data=go.Candlestick(
                    x=enhanced_data.index,
                    open=enhanced_data['Open'],
                    high=enhanced_data['High'],
                    low=enhanced_data['Low'],
                    close=enhanced_data['Close'],
                    name=symbol
                ))
                simple_fig.update_layout(title=f"{symbol} - Basic Chart", height=400)
                st.plotly_chart(simple_fig, use_container_width=True)
        
        # Clean pattern analysis section
        if show_patterns:
            st.markdown("---")
            try:
                render_clean_pattern_analysis(enhanced_data, symbol, real_time_data)
            except Exception as pattern_error:
                st.error(f"Pattern analysis error: {str(pattern_error)}")
                st.info("Pattern detection temporarily unavailable.")
        
        # Technical Indicator Dashboard
        st.markdown("---")
        st.markdown("### 📊 Technical Indicator Dashboard")
        
        if not enhanced_data.empty:
            latest = enhanced_data.iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI Gauge
                rsi = latest.get('RSI', 0)
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
                    macd = latest.get('MACD', 0)
                    macd_signal = latest.get('MACD_Signal', 0)
                    if macd != 0 and macd_signal != 0:
                        macd_trend = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
                        st.metric("MACD Signal", macd_trend, f"{macd:.3f}")
                    
                with col2b:
                    if 'ATR' in enhanced_data.columns:
                        atr = latest.get('ATR', 0)
                        current_price = latest.get('Close', 1)
                        if atr > 0 and current_price > 0:
                            vol_pct = (atr / current_price) * 100
                            st.metric("Volatility (ATR)", f"{vol_pct:.1f}%", f"${atr:.2f}")
            
            with col2:
                # Additional technical indicators
                st.markdown("#### 📈 Additional Indicators")
                
                # Bollinger Band Position
                if 'BB_Position' in enhanced_data.columns:
                    bb_pos = latest.get('BB_Position', 0.5) * 100
                    st.metric("Bollinger Band Position", f"{bb_pos:.1f}%")
                
                # Stochastic
                if 'Stoch_K' in enhanced_data.columns:
                    stoch_k = latest.get('Stoch_K', 50)
                    stoch_status = "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral"
                    st.metric("Stochastic %K", f"{stoch_k:.1f}", stoch_status)
                
                # Williams %R
                if 'Williams_R' in enhanced_data.columns:
                    williams_r = latest.get('Williams_R', -50)
                    st.metric("Williams %R", f"{williams_r:.1f}")
                
                # Volume Ratio
                if 'Volume_Ratio' in enhanced_data.columns:
                    vol_ratio = latest.get('Volume_Ratio', 1)
                    vol_status = "High" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.5 else "Normal"
                    st.metric("Volume Ratio", f"{vol_ratio:.2f}x", vol_status)

        # AI-Powered Comprehensive Analysis
        st.markdown("---")
        st.markdown("### 🧠 AI-Powered Comprehensive Analysis")
        
        llm = st.session_state.get('llm', None)
        if st.button("🎯 Generate AI Analysis", disabled=(llm is None), 
                    help="Generate comprehensive AI-powered market analysis"):
            
            with st.spinner("🤖 AI is analyzing market data... This may take a moment."):
                try:
                    # Get additional data
                    fundamental_data = eodhd.get_fundamentals(symbol) if api_key else {}
                    sentiment_data = eodhd.get_news(symbol, limit=10) if api_key else []
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

        # Predictive Analytics & Forecasting
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
                        # Simple anomaly detection using statistical methods
                        recent_data = enhanced_data.tail(60)
                        
                        # Price anomalies (using z-score)
                        price_changes = recent_data['Close'].pct_change()
                        price_mean = price_changes.mean()
                        price_std = price_changes.std()
                        price_z_scores = (price_changes - price_mean) / price_std
                        
                        # Volume anomalies
                        volume_mean = recent_data['Volume'].mean()
                        volume_std = recent_data['Volume'].std()
                        volume_z_scores = (recent_data['Volume'] - volume_mean) / volume_std
                        
                        # Find anomalies (z-score > 2 or < -2)
                        anomaly_mask = (np.abs(price_z_scores) > 2) | (np.abs(volume_z_scores) > 2)
                        anomalies = recent_data[anomaly_mask].copy()
                        
                        if not anomalies.empty:
                            anomalies['Price_Change'] = price_changes[anomaly_mask]
                            anomalies['Volume_Change'] = (anomalies['Volume'] / volume_mean - 1)
                            
                            st.session_state.anomalies = anomalies
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
        
        # Technical analysis summary
        st.markdown("---")
        st.markdown("### 📊 Technical Analysis Summary")
        
        try:
            latest_data = enhanced_data.iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Moving Averages")
                
                price = latest_data['Close']
                sma_20 = latest_data.get('SMA_20', 0)
                sma_50 = latest_data.get('SMA_50', 0)
                
                ma_signals = []
                if sma_20 > 0:
                    if price > sma_20:
                        ma_signals.append("🟢 Price above SMA 20")
                    else:
                        ma_signals.append("🔴 Price below SMA 20")
                
                if sma_50 > 0:
                    if price > sma_50:
                        ma_signals.append("🟢 Price above SMA 50")
                    else:
                        ma_signals.append("🔴 Price below SMA 50")
                
                if sma_20 > 0 and sma_50 > 0:
                    if sma_20 > sma_50:
                        ma_signals.append("🟢 SMA 20 above SMA 50")
                    else:
                        ma_signals.append("🔴 SMA 20 below SMA 50")
                
                for signal in ma_signals:
                    st.markdown(f"- {signal}")
            
            with col2:
                st.markdown("#### 🎯 Oscillators")
                
                rsi = latest_data.get('RSI', 50)
                stoch_k = latest_data.get('Stoch_K', 50)
                
                oscillator_signals = []
                
                if rsi > 70:
                    oscillator_signals.append("🔴 RSI Overbought")
                elif rsi < 30:
                    oscillator_signals.append("🟢 RSI Oversold")
                else:
                    oscillator_signals.append("🟡 RSI Neutral")
                
                if stoch_k > 80:
                    oscillator_signals.append("🔴 Stochastic Overbought")
                elif stoch_k < 20:
                    oscillator_signals.append("🟢 Stochastic Oversold")
                else:
                    oscillator_signals.append("🟡 Stochastic Neutral")
                
                for signal in oscillator_signals:
                    st.markdown(f"- {signal}")
        
        except Exception as summary_error:
            st.warning(f"Could not generate technical summary: {str(summary_error)}")
        
        # Additional data sections
        col1, col2 = st.columns(2)
        
        with col1:
            # Fundamental data
            if st.button("📊 Load Fundamental Data", key="load_fundamentals"):
                with st.spinner("Loading fundamental data..."):
                    try:
                        fundamentals = eodhd.get_fundamentals(symbol)
                        if fundamentals and any(fundamentals.values()):
                            st.markdown("#### 💼 Fundamental Data")

                            def _fmt_large(v):
                                if v >= 1e12: return f"${v/1e12:.2f}T"
                                if v >= 1e9:  return f"${v/1e9:.2f}B"
                                if v >= 1e6:  return f"${v/1e6:.2f}M"
                                return f"${v:,.0f}"

                            mc  = fundamentals.get('market_cap', 0)
                            rev = fundamentals.get('revenue', 0)
                            ni  = fundamentals.get('net_income', 0)
                            ebi = fundamentals.get('ebitda', 0)

                            fund_df = pd.DataFrame([
                                {"Metric": "Sector",         "Value": fundamentals.get('sector', 'N/A')},
                                {"Metric": "Industry",        "Value": fundamentals.get('industry', 'N/A')},
                                {"Metric": "Market Cap",      "Value": _fmt_large(mc) if mc else "N/A"},
                                {"Metric": "Revenue (TTM)",   "Value": _fmt_large(rev) if rev else "N/A"},
                                {"Metric": "Net Income",      "Value": _fmt_large(ni) if ni else "N/A"},
                                {"Metric": "EBITDA",          "Value": _fmt_large(ebi) if ebi else "N/A"},
                                {"Metric": "P/E Ratio",       "Value": f"{fundamentals.get('pe_ratio', 0):.2f}"},
                                {"Metric": "PEG Ratio",       "Value": f"{fundamentals.get('peg_ratio', 0):.2f}"},
                                {"Metric": "P/B Ratio",       "Value": f"{fundamentals.get('price_to_book', 0):.2f}"},
                                {"Metric": "P/S Ratio",       "Value": f"{fundamentals.get('price_to_sales', 0):.2f}"},
                                {"Metric": "EPS",             "Value": f"${fundamentals.get('eps', 0):.2f}"},
                                {"Metric": "Dividend Yield",  "Value": f"{fundamentals.get('dividend_yield', 0):.2f}%"},
                                {"Metric": "Profit Margin",   "Value": f"{fundamentals.get('profit_margin', 0):.2f}%"},
                                {"Metric": "ROE",             "Value": f"{fundamentals.get('roe', 0):.2f}%"},
                                {"Metric": "ROA",             "Value": f"{fundamentals.get('roa', 0):.2f}%"},
                                {"Metric": "Debt/Equity",     "Value": f"{fundamentals.get('debt_to_equity', 0):.2f}"},
                                {"Metric": "Beta",            "Value": f"{fundamentals.get('beta', 0):.2f}"},
                                {"Metric": "52W High",        "Value": f"${fundamentals.get('52_week_high', 0):.2f}"},
                                {"Metric": "52W Low",         "Value": f"${fundamentals.get('52_week_low', 0):.2f}"},
                            ])

                            st.dataframe(fund_df, hide_index=True, use_container_width=True)
                        else:
                            st.info("Fundamental data not available for this symbol.")
                    except Exception as fund_error:
                        st.error(f"Could not load fundamental data: {str(fund_error)}")
        
        with col2:
            # News and sentiment
            if st.button("📰 Load News & Sentiment", key="load_news"):
                with st.spinner("Loading news and sentiment data..."):
                    try:
                        news_data = eodhd.get_news(symbol, limit=20)
                        if news_data:
                            st.markdown("#### 📰 Latest News")
                            
                            # Show recent headlines
                            for i, article in enumerate(news_data[:5]):
                                sentiment_icon = {
                                    'positive': '🟢',
                                    'negative': '🔴',
                                    'neutral': '🟡'
                                }.get(article.get('sentiment', 'neutral'), '🟡')
                                
                                st.markdown(f"{sentiment_icon} {article['title'][:100]}...")
                        else:
                            st.info("News data not available for this symbol.")
                    except Exception as news_error:
                        st.error(f"Could not load news data: {str(news_error)}")
        
        # Uploaded data integration
        uploaded_files = st.session_state.get('uploaded_files', [])
        if uploaded_files:
            st.markdown("---")
            st.markdown("### 📁 Uploaded Data Integration")
            
            try:
                integrated_data = integrate_uploaded_data(symbol, uploaded_files)
                
                if integrated_data:
                    st.success(f"✅ Found {len(integrated_data)} relevant data sources for {symbol}")
                    
                    for data_type, data_content in integrated_data.items():
                        with st.expander(f"📊 {data_type.replace('_', ' ').title()}", expanded=False):
                            if isinstance(data_content, pd.DataFrame):
                                st.dataframe(data_content.head(10), use_container_width=True)
                                st.info(f"Showing first 10 rows of {len(data_content)} total rows")
                            elif isinstance(data_content, dict):
                                for key, value in data_content.items():
                                    st.markdown(f"**{key}:** {value}")
                            else:
                                st.text(str(data_content)[:500] + "..." if len(str(data_content)) > 500 else str(data_content))
                else:
                    st.info(f"No uploaded files found relevant to {symbol}")
            except Exception as upload_error:
                st.warning(f"Could not process uploaded data: {str(upload_error)}")
        
        # Export functionality
        st.markdown("---")
        st.markdown("### 📤 Export Data & Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Technical Data", key="export_tech"):
                try:
                    with st.spinner("Preparing technical data export…"):
                        csv_data = enhanced_data.tail(100).to_csv()
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=f"{symbol}_technical_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                except Exception as export_error:
                    st.error(f"Export error: {str(export_error)}")

        with col2:
            patterns = st.session_state.get('detected_patterns', [])
            if patterns and st.button("🎯 Export Patterns", key="export_patterns"):
                try:
                    with st.spinner("Preparing patterns export…"):
                        pattern_df = pd.DataFrame(patterns)
                        csv_data = pattern_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Patterns",
                        data=csv_data,
                        file_name=f"{symbol}_patterns_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                except Exception as pattern_export_error:
                    st.error(f"Pattern export error: {str(pattern_export_error)}")

        with col3:
            if st.button("📈 Export Chart", key="export_chart"):
                st.info("Chart export functionality - would generate PNG/PDF of the current chart")
    
    except Exception as main_error:
        render_error_info(f"Application error: {str(main_error)}", "show_main_error_details")
        render_demo_fallback()
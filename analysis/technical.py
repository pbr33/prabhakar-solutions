import pandas as pd
import numpy as np
import random
from typing import List, Dict
import streamlit as st
import requests
from datetime import datetime

def detect_real_candlestick_patterns(data: pd.DataFrame) -> List[Dict]:
    """
    Detects real candlestick patterns using TA-Lib or pandas-ta.
    Falls back to enhanced rule-based detection if TA-Lib is not available.
    """
    patterns_detected = []
    
    if len(data) < 10:
        return patterns_detected
    
    try:
        # Engulfing patterns
        if prev is not None:
            prev_body = abs(prev['Close'] - prev['Open'])
            
            # Bullish Engulfing
            if (prev['Close'] < prev['Open'] and  # Previous was bearish
                current['Close'] > current['Open'] and  # Current is bullish
                current['Open'] < prev['Close'] and  # Current opens below prev close
                current['Close'] > prev['Open'] and  # Current closes above prev open
                body_size > prev_body * 1.1):  # Current body is larger
                
                patterns.append({
                    "date": current.name,
                    "name": "Bullish Engulfing",
                    "type": "Bullish",
                    "description": "Strong bullish reversal signal",
                    "strength": 1,
                    "confidence": 80
                })
            
            # Bearish Engulfing
            elif (prev['Close'] > prev['Open'] and  # Previous was bullish
                  current['Close'] < current['Open'] and  # Current is bearish
                  current['Open'] > prev['Close'] and  # Current opens above prev close
                  current['Close'] < prev['Open'] and  # Current closes below prev open
                  body_size > prev_body * 1.1):  # Current body is larger
                
                patterns.append({
                    "date": current.name,
                    "name": "Bearish Engulfing",
                    "type": "Bearish",
                    "description": "Strong bearish reversal signal",
                    "strength": 1,
                    "confidence": 80
                })
    
    return patterns[-10:]  # Return last 10 patterns

def calculate_advanced_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced technical indicators using pandas-ta."""
    try:
        import pandas_ta as ta
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Basic indicators (if not already present)
        if 'RSI_14' not in df.columns:
            df.ta.rsi(length=14, append=True)
        if 'MACD_12_26_9' not in df.columns:
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Volatility indicators
        df.ta.bbands(length=20, std=2, append=True)  # Bollinger Bands
        df.ta.atr(length=14, append=True)  # Average True Range
        df.ta.kc(length=20, scalar=2, append=True)  # Keltner Channels
        
        # Volume indicators
        df.ta.obv(append=True)  # On-Balance Volume
        df.ta.ad(append=True)   # Accumulation/Distribution
        df.ta.cmf(length=20, append=True)  # Chaikin Money Flow
        
        # Trend indicators
        df.ta.ema(length=20, append=True)  # Exponential Moving Average
        df.ta.sma(length=50, append=True)  # Simple Moving Average
        df.ta.adx(length=14, append=True)  # Average Directional Index
        
        # Momentum indicators
        df.ta.stoch(k=14, d=3, append=True)  # Stochastic
        df.ta.cci(length=20, append=True)    # Commodity Channel Index
        df.ta.willr(length=14, append=True)  # Williams %R
        
        # Support and Resistance
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Resistance_1'] = 2 * df['Pivot'] - df['Low']
        df['Support_1'] = 2 * df['Pivot'] - df['High']
        df['Resistance_2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['Support_2'] = df['Pivot'] - (df['High'] - df['Low'])
        
        return df
        
    except ImportError:
        st.warning("pandas_ta not installed. Using basic indicators only.")
        return data

def calculate_fibonacci_levels(data: pd.DataFrame, period: int = 50) -> Dict:
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

def get_fundamental_data(symbol: str, api_key: str) -> Dict:
    """Fetch fundamental data from EODHD API."""
    if not api_key:
        return {}
    
    url = f"https://eodhd.com/api/fundamentals/{symbol}?api_token={api_key}&fmt=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract key fundamental metrics
        general = data.get('General', {})
        highlights = data.get('Highlights', {})
        valuation = data.get('Valuation', {})
        
        return {
            'sector': general.get('Sector', 'N/A'),
            'industry': general.get('Industry', 'N/A'),
            'market_cap': highlights.get('MarketCapitalization', 'N/A'),
            'pe_ratio': highlights.get('PERatio', 'N/A'),
            'eps': highlights.get('EarningsShare', 'N/A'),
            'dividend_yield': highlights.get('DividendYield', 'N/A'),
            'price_to_book': valuation.get('PriceBookMRQ', 'N/A'),
            'price_to_sales': valuation.get('PriceSalesTTM', 'N/A'),
            'debt_to_equity': highlights.get('DebtToEquity', 'N/A')
        }
    except Exception as e:
        st.warning(f"Could not fetch fundamental data: {e}")
        return {}

def get_sentiment_data(symbol: str, api_key: str) -> Dict:
    """Fetch news and sentiment data from EODHD API."""
    if not api_key:
        return {}
    
    url = f"https://eodhd.com/api/news?s={symbol}&api_token={api_key}&limit=10&fmt=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        
        # Simple sentiment analysis based on headline keywords
        positive_words = ['gain', 'rise', 'up', 'positive', 'growth', 'beat', 'exceed', 'strong', 'bull']
        negative_words = ['fall', 'drop', 'down', 'negative', 'decline', 'miss', 'weak', 'bear', 'loss']
        
        sentiment_score = 0
        total_articles = len(news_data)
        recent_headlines = []
        
        for article in news_data[:5]:  # Analyze top 5 articles
            title = article.get('title', '').lower()
            recent_headlines.append(article.get('title', ''))
            
            pos_count = sum(1 for word in positive_words if word in title)
            neg_count = sum(1 for word in negative_words if word in title)
            sentiment_score += (pos_count - neg_count)
        
        # Normalize sentiment score
        if total_articles > 0:
            sentiment_score = sentiment_score / total_articles
        
        sentiment_label = 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'recent_headlines': recent_headlines[:3],
            'total_articles': total_articles
        }
    except Exception as e:
        st.warning(f"Could not fetch sentiment data: {e}")
        return {}

def generate_comprehensive_analysis_report(llm, symbol: str, patterns_detected: List[Dict], 
                                         enhanced_data: pd.DataFrame, fundamental_data: Dict, 
                                         sentiment_data: Dict, fibonacci_levels: Dict):
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
    
    # Format technical indicators
    technical_str = f"""
**Trend & Momentum:**
- RSI (14): {latest.get('RSI_14', 'N/A'):.2f if isinstance(latest.get('RSI_14'), (int, float)) and not np.isnan(latest.get('RSI_14', np.nan)) else 'N/A'}
- MACD Line: {latest.get('MACD_12_26_9', 'N/A'):.2f if isinstance(latest.get('MACD_12_26_9'), (int, float)) and not np.isnan(latest.get('MACD_12_26_9', np.nan)) else 'N/A'}
- MACD Signal: {latest.get('MACDs_12_26_9', 'N/A'):.2f if isinstance(latest.get('MACDs_12_26_9'), (int, float)) and not np.isnan(latest.get('MACDs_12_26_9', np.nan)) else 'N/A'}
- ADX (14): {latest.get('ADX_14', 'N/A'):.2f if isinstance(latest.get('ADX_14'), (int, float)) and not np.isnan(latest.get('ADX_14', np.nan)) else 'N/A'}
- Stochastic %K: {latest.get('STOCHk_14_3_3', 'N/A'):.2f if isinstance(latest.get('STOCHk_14_3_3'), (int, float)) and not np.isnan(latest.get('STOCHk_14_3_3', np.nan)) else 'N/A'}

**Volatility:**
- ATR (14): {latest.get('ATR_14', 'N/A'):.2f if isinstance(latest.get('ATR_14'), (int, float)) and not np.isnan(latest.get('ATR_14', np.nan)) else 'N/A'}
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
- OBV: {latest.get('OBV', 'N/A'):,.0f if isinstance(latest.get('OBV'), (int, float)) and not np.isnan(latest.get('OBV', np.nan)) else 'N/A'}
- CMF (20): {latest.get('CMF_20', 'N/A'):.2f if isinstance(latest.get('CMF_20'), (int, float)) and not np.isnan(latest.get('CMF_20', np.nan)) else 'N/A'}
- A/D Line: {latest.get('AD', 'N/A'):,.0f if isinstance(latest.get('AD'), (int, float)) and not np.isnan(latest.get('AD', np.nan)) else 'N/A'}
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
    sentiment_str = f"""
**Market Sentiment:**
- Overall Sentiment: {sentiment_data.get('sentiment_label', 'N/A')}
- Sentiment Score: {sentiment_data.get('sentiment_score', 'N/A'):.2f if isinstance(sentiment_data.get('sentiment_score'), (int, float)) else 'N/A'}
- Recent Headlines:
{chr(10).join([f"  • {headline}" for headline in sentiment_data.get('recent_headlines', [])])}
""" if sentiment_data else "**Sentiment data not available**"
    
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

def detect_fibonacci_levels(data: pd.DataFrame) -> dict:
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

@st.cache_data(ttl=600)
def run_ma_crossover_backtest(_api_key, ticker, start_date, end_date, initial_capital, short_window, long_window):
    """Runs a Moving Average Crossover backtest."""
    from_date_str = start_date.strftime('%Y-%m-%d')
    to_date_str = end_date.strftime('%Y-%m-%d')
    url = f"https://eodhd.com/api/eod/{ticker}?from={from_date_str}&to={to_date_str}&api_token={_api_key}&period=d&fmt=json"
    
    try:
        response = requests.get(url, timeout=30)
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
        price = df['close'].iloc[i]
        if df['positions'].iloc[i] == 1 and not position_open: # Buy
            shares_to_buy = cash / price
            shares += shares_to_buy
            cash = 0
            position_open = True
            trade_log.append({'type': 'BUY', 'date': df.index[i], 'price': price, 'shares': shares_to_buy})
        elif df['positions'].iloc[i] == -1 and position_open: # Sell
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
    } Try to use TA-Lib for real pattern detection
        import talib
        
        # Get OHLC data
        open_prices = data['Open'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
        # Define TA-Lib pattern functions and their descriptions
        pattern_functions = {
            'CDLDOJI': ('Doji', 'Neutral', 'Market indecision - open and close prices are very close'),
            'CDLENGULFING': ('Engulfing', 'Variable', 'One candle completely engulfs the previous candle'),
            'CDLHAMMER': ('Hammer', 'Bullish', 'Potential reversal after downtrend - long lower shadow'),
            'CDLSHOOTINGSTAR': ('Shooting Star', 'Bearish', 'Potential reversal after uptrend - long upper shadow'),
            'CDLMORNINGSTAR': ('Morning Star', 'Bullish', 'Three-candle reversal pattern indicating potential uptrend'),
            'CDLEVENINGSTAR': ('Evening Star', 'Bearish', 'Three-candle reversal pattern indicating potential downtrend'),
            'CDLDRAGONFLYDOJI': ('Dragonfly Doji', 'Bullish', 'Doji with long lower shadow - potential reversal'),
            'CDLGRAVESTONEDOJI': ('Gravestone Doji', 'Bearish', 'Doji with long upper shadow - potential reversal'),
            'CDLPIERCING': ('Piercing Line', 'Bullish', 'Bullish reversal pattern in downtrend'),
            'CDLDARKCLOUDCOVER': ('Dark Cloud Cover', 'Bearish', 'Bearish reversal pattern in uptrend'),
            'CDLMARUBOZU': ('Marubozu', 'Variable', 'Strong directional move - no wicks'),
            'CDLSPINNINGTOP': ('Spinning Top', 'Neutral', 'Market indecision - small body with wicks'),
        }
        
        # Detect patterns
        for func_name, (pattern_name, pattern_type, description) in pattern_functions.items():
            try:
                pattern_func = getattr(talib, func_name)
                pattern_result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                
                # Find where patterns occur (non-zero values)
                pattern_indices = np.where(pattern_result != 0)[0]
                
                for idx in pattern_indices[-10:]:  # Get last 10 occurrences
                    # Determine if bullish or bearish based on pattern result value
                    signal_strength = pattern_result[idx]
                    if pattern_type == 'Variable':
                        actual_type = 'Bullish' if signal_strength > 0 else 'Bearish'
                    else:
                        actual_type = pattern_type
                    
                    patterns_detected.append({
                        "date": data.index[idx],
                        "name": pattern_name,
                        "type": actual_type,
                        "description": description,
                        "strength": abs(signal_strength),
                        "confidence": min(abs(signal_strength) * 20, 100)  # Convert to percentage
                    })
            except Exception as e:
                continue
                
    except ImportError:
        # Fallback to enhanced rule-based detection
        st.info("TA-Lib not available. Using enhanced rule-based pattern detection.")
        patterns_detected = detect_enhanced_rule_based_patterns(data)
    
    # Sort by date and return most recent patterns
    patterns_detected.sort(key=lambda x: x['date'], reverse=True)
    return patterns_detected[:8]  # Return top 8 most recent patterns

def detect_enhanced_rule_based_patterns(data: pd.DataFrame) -> List[Dict]:
    """Enhanced rule-based pattern detection when TA-Lib is not available."""
    patterns = []
    
    if len(data) < 5:
        return patterns
    
    for i in range(2, len(data)):
        current = data.iloc[i]
        prev = data.iloc[i-1]
        prev2 = data.iloc[i-2] if i >= 2 else None
        
        # Calculate candle properties
        body_size = abs(current['Close'] - current['Open'])
        upper_shadow = current['High'] - max(current['Open'], current['Close'])
        lower_shadow = min(current['Open'], current['Close']) - current['Low']
        total_range = current['High'] - current['Low']
        
        # Doji pattern
        if body_size < (total_range * 0.1) and total_range > 0:
            if lower_shadow > (total_range * 0.6):
                pattern_type = "Dragonfly Doji"
                signal = "Bullish"
            elif upper_shadow > (total_range * 0.6):
                pattern_type = "Gravestone Doji"
                signal = "Bearish"
            else:
                pattern_type = "Doji"
                signal = "Neutral"
            
            patterns.append({
                "date": current.name,
                "name": pattern_type,
                "type": signal,
                "description": f"Market indecision with {signal.lower()} bias",
                "strength": 1,
                "confidence": 70
            })
        
        # Hammer pattern
        elif (lower_shadow > body_size * 2 and 
              upper_shadow < body_size * 0.5 and
              i >= 5 and data.iloc[i-5:i]['Close'].mean() > current['Close']):
            patterns.append({
                "date": current.name,
                "name": "Hammer",
                "type": "Bullish",
                "description": "Potential reversal after downtrend",
                "strength": 1,
                "confidence": 75
            })
        
        # Shooting Star pattern
        elif (upper_shadow > body_size * 2 and 
              lower_shadow < body_size * 0.5 and
              i >= 5 and data.iloc[i-5:i]['Close'].mean() < current['Close']):
            patterns.append({
                "date": current.name,
                "name": "Shooting Star",
                "type": "Bearish",
                "description": "Potential reversal after uptrend",
                "strength": 1,
                "confidence": 75
            })
        
        
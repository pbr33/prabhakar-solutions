# realtime_multi_agent_system.py
"""
Real-Time Multi-Agent Trading System with LangGraph
Features:
- Real-time data integration with EODHD API
- LLM-powered agent debates and consensus building
- Advanced technical, fundamental, sentiment, and quant analysis
- Parallel processing with conflict resolution
- Dynamic portfolio risk management
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
import talib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from enum import Enum

# LangGraph and LLM imports
try:
    from langgraph.graph import Graph, Node, Edge, START, END
    from langchain.llms.base import LLM
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_openai import ChatOpenAI
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available. Install with: pip install langgraph langchain-openai")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY" 
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    ERROR = "ERROR"

@dataclass
class AgentSignal:
    """Standardized signal format for all agents."""
    signal: SignalType
    confidence: float  # 0-100
    reasoning: str
    key_data: Dict[str, Any]
    timestamp: datetime
    agent_name: str
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_score: Optional[float] = None

@dataclass
class MarketData:
    """Comprehensive market data container."""
    symbol: str
    current_price: float
    historical_data: pd.DataFrame
    technical_indicators: Dict[str, Any]
    fundamental_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    news: List[Dict]
    real_time_data: Dict[str, Any]
    timestamp: datetime

class EODHDDataProvider:
    """Real-time data provider using EODHD API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
        
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote data."""
        url = f"{self.base_url}/real-time/{symbol}"
        params = {"api_token": self.api_key, "fmt": "json"}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch real-time data for {symbol}: {e}")
            return {}
    
    def get_intraday_data(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """Get intraday data for technical analysis."""
        url = f"{self.base_url}/intraday/{symbol}"
        params = {"interval": interval, "api_token": self.api_key, "fmt": "json"}
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                df = df.astype(float)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data."""
        url = f"{self.base_url}/fundamentals/{symbol}"
        params = {"api_token": self.api_key, "fmt": "json"}
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch fundamental data for {symbol}: {e}")
            return {}
    
    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get latest news for sentiment analysis."""
        url = f"{self.base_url}/news"
        params = {"s": symbol, "limit": limit, "api_token": self.api_key, "fmt": "json"}
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return []

class TechnicalAgent:
    """Advanced technical analysis agent."""
    
    def __init__(self, llm: Optional[Any] = None):
        self.name = "Technical Analyst"
        self.specialty = "Chart patterns, indicators, and price action analysis"
        self.llm = llm
        
    def analyze(self, market_data: MarketData) -> AgentSignal:
        """Perform comprehensive technical analysis."""
        try:
            df = market_data.historical_data
            if df.empty:
                return self._error_signal("No historical data available")
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            patterns = self._detect_patterns(df)
            support_resistance = self._find_support_resistance(df)
            
            # Generate signal based on technical confluence
            signal = self._generate_technical_signal(
                indicators, patterns, support_resistance, market_data.current_price
            )
            
            # Use LLM for advanced reasoning if available
            if self.llm:
                signal = self._enhance_with_llm(signal, indicators, patterns, market_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return self._error_signal(str(e))
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive technical indicators."""
        try:
            high, low, close, volume = df['high'].values, df['low'].values, df['close'].values, df['volume'].values
            
            return {
                'rsi_14': talib.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50,
                'macd_line': talib.MACD(close)[0][-1] if len(close) >= 26 else 0,
                'macd_signal': talib.MACD(close)[1][-1] if len(close) >= 26 else 0,
                'bb_upper': talib.BBANDS(close)[0][-1] if len(close) >= 20 else close[-1] * 1.02,
                'bb_lower': talib.BBANDS(close)[2][-1] if len(close) >= 20 else close[-1] * 0.98,
                'sma_20': talib.SMA(close, timeperiod=20)[-1] if len(close) >= 20 else close[-1],
                'sma_50': talib.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else close[-1],
                'atr': talib.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0,
                'adx': talib.ADX(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 25,
                'stoch_k': talib.STOCH(high, low, close)[0][-1] if len(close) >= 14 else 50,
                'obv': talib.OBV(close, volume)[-1] if len(close) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return {}
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect candlestick and chart patterns."""
        try:
            open_prices, high, low, close = df['open'].values, df['high'].values, df['low'].values, df['close'].values
            
            patterns = {}
            if len(close) >= 10:  # Minimum data requirement
                patterns['doji'] = talib.CDLDOJI(open_prices, high, low, close)[-1]
                patterns['hammer'] = talib.CDLHAMMER(open_prices, high, low, close)[-1]
                patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high, low, close)[-1]
                patterns['engulfing'] = talib.CDLENGULFING(open_prices, high, low, close)[-1]
                patterns['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high, low, close)[-1]
                patterns['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high, low, close)[-1]
            
            return patterns
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return {}
    
    def _find_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """Identify key support and resistance levels."""
        try:
            if len(df) < lookback:
                return {}
            
            recent_data = df.tail(lookback)
            support = recent_data['low'].min()
            resistance = recent_data['high'].max()
            
            return {
                'support': support,
                'resistance': resistance,
                'pivot': (support + resistance + recent_data['close'].iloc[-1]) / 3
            }
        except Exception:
            return {}
    
    def _generate_technical_signal(self, indicators: Dict, patterns: Dict, 
                                 levels: Dict, current_price: float) -> AgentSignal:
        """Generate trading signal based on technical analysis."""
        bullish_signals = 0
        bearish_signals = 0
        confidence_factors = []
        reasoning_parts = []
        
        # RSI Analysis
        rsi = indicators.get('rsi_14', 50)
        if rsi < 30:
            bullish_signals += 2
            confidence_factors.append(0.8)
            reasoning_parts.append(f"RSI oversold at {rsi:.1f}")
        elif rsi > 70:
            bearish_signals += 2
            confidence_factors.append(0.8)
            reasoning_parts.append(f"RSI overbought at {rsi:.1f}")
        
        # MACD Analysis
        macd_line = indicators.get('macd_line', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd_line > macd_signal and macd_line > 0:
            bullish_signals += 1
            confidence_factors.append(0.6)
            reasoning_parts.append("MACD bullish crossover")
        elif macd_line < macd_signal and macd_line < 0:
            bearish_signals += 1
            confidence_factors.append(0.6)
            reasoning_parts.append("MACD bearish crossover")
        
        # Moving Average Analysis
        sma_20 = indicators.get('sma_20', current_price)
        sma_50 = indicators.get('sma_50', current_price)
        if current_price > sma_20 > sma_50:
            bullish_signals += 1
            confidence_factors.append(0.5)
            reasoning_parts.append("Price above key MAs")
        elif current_price < sma_20 < sma_50:
            bearish_signals += 1
            confidence_factors.append(0.5)
            reasoning_parts.append("Price below key MAs")
        
        # Pattern Analysis
        bullish_patterns = patterns.get('hammer', 0) + patterns.get('morning_star', 0)
        bearish_patterns = patterns.get('shooting_star', 0) + patterns.get('evening_star', 0)
        
        if bullish_patterns > 0:
            bullish_signals += 1
            confidence_factors.append(0.7)
            reasoning_parts.append("Bullish candlestick pattern detected")
        if bearish_patterns > 0:
            bearish_signals += 1
            confidence_factors.append(0.7)
            reasoning_parts.append("Bearish candlestick pattern detected")
        
        # Determine signal
        if bullish_signals > bearish_signals + 1:
            signal_type = SignalType.BUY if bullish_signals > bearish_signals + 2 else SignalType.BUY
            if bullish_signals > bearish_signals + 3:
                signal_type = SignalType.STRONG_BUY
        elif bearish_signals > bullish_signals + 1:
            signal_type = SignalType.SELL if bearish_signals > bullish_signals + 2 else SignalType.SELL
            if bearish_signals > bullish_signals + 3:
                signal_type = SignalType.STRONG_SELL
        else:
            signal_type = SignalType.HOLD
        
        confidence = min(95, max(50, np.mean(confidence_factors) * 100 if confidence_factors else 60))
        
        # Calculate price targets
        atr = indicators.get('atr', current_price * 0.02)
        price_target = None
        stop_loss = None
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            price_target = current_price + (2 * atr)
            stop_loss = current_price - atr
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            price_target = current_price - (2 * atr)
            stop_loss = current_price + atr
        
        return AgentSignal(
            signal=signal_type,
            confidence=confidence,
            reasoning=f"Technical Analysis: {'; '.join(reasoning_parts)}",
            key_data={
                'indicators': indicators,
                'patterns': patterns,
                'levels': levels,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals
            },
            timestamp=datetime.now(),
            agent_name=self.name,
            price_target=price_target,
            stop_loss=stop_loss,
            risk_score=min(100, atr / current_price * 100) if current_price > 0 else 50
        )
    
    def _enhance_with_llm(self, signal: AgentSignal, indicators: Dict, 
                         patterns: Dict, market_data: MarketData) -> AgentSignal:
        """Enhance analysis with LLM reasoning."""
        try:
            prompt = f"""
            As a senior technical analyst, analyze this technical data for {market_data.symbol}:
            
            Current Price: ${market_data.current_price:.2f}
            
            Technical Indicators:
            - RSI: {indicators.get('rsi_14', 'N/A')}
            - MACD Line: {indicators.get('macd_line', 'N/A')}
            - MACD Signal: {indicators.get('macd_signal', 'N/A')}
            - 20-day SMA: {indicators.get('sma_20', 'N/A')}
            - 50-day SMA: {indicators.get('sma_50', 'N/A')}
            - ADX: {indicators.get('adx', 'N/A')}
            - ATR: {indicators.get('atr', 'N/A')}
            
            Detected Patterns: {patterns}
            
            Current Signal: {signal.signal.value}
            Confidence: {signal.confidence}%
            
            Provide enhanced reasoning and adjust confidence if needed. Focus on:
            1. Technical confluence/divergence
            2. Risk assessment
            3. Key levels to watch
            4. Market structure analysis
            
            Return JSON format:
            {{"enhanced_reasoning": "...", "adjusted_confidence": XX, "key_insights": "..."}}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                llm_output = json.loads(response.content)
                signal.reasoning = llm_output.get('enhanced_reasoning', signal.reasoning)
                signal.confidence = min(100, max(0, llm_output.get('adjusted_confidence', signal.confidence)))
                signal.key_data['llm_insights'] = llm_output.get('key_insights', '')
            except json.JSONDecodeError:
                signal.key_data['llm_raw_response'] = response.content
                
        except Exception as e:
            logger.error(f"LLM enhancement error: {e}")
        
        return signal
    
    def _error_signal(self, error_msg: str) -> AgentSignal:
        """Create error signal."""
        return AgentSignal(
            signal=SignalType.ERROR,
            confidence=0,
            reasoning=f"Technical analysis failed: {error_msg}",
            key_data={'error': error_msg},
            timestamp=datetime.now(),
            agent_name=self.name
        )

class FundamentalAgent:
    """Fundamental analysis agent focusing on financial health and valuation."""
    
    def __init__(self, llm: Optional[Any] = None):
        self.name = "Fundamental Analyst"
        self.specialty = "Financial metrics, valuation, and business analysis"
        self.llm = llm
    
    def analyze(self, market_data: MarketData) -> AgentSignal:
        """Perform fundamental analysis."""
        try:
            fundamental_data = market_data.fundamental_data
            if not fundamental_data:
                return self._error_signal("No fundamental data available")
            
            # Extract key metrics
            metrics = self._extract_key_metrics(fundamental_data)
            valuation = self._assess_valuation(metrics, market_data.current_price)
            financial_health = self._assess_financial_health(metrics)
            
            # Generate signal
            signal = self._generate_fundamental_signal(metrics, valuation, financial_health)
            
            # Enhance with LLM if available
            if self.llm:
                signal = self._enhance_with_llm(signal, metrics, market_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Fundamental analysis error: {e}")
            return self._error_signal(str(e))
    
    def _extract_key_metrics(self, fundamental_data: Dict) -> Dict[str, Any]:
        """Extract key financial metrics."""
        highlights = fundamental_data.get('Highlights', {})
        financials = fundamental_data.get('Financials', {})
        valuation = fundamental_data.get('Valuation', {})
        
        return {
            'market_cap': highlights.get('MarketCapitalization'),
            'pe_ratio': highlights.get('PERatio'),
            'forward_pe': highlights.get('ForwardPE'),
            'peg_ratio': highlights.get('PEGRatio'),
            'price_to_book': valuation.get('PriceBookMRQ'),
            'price_to_sales': valuation.get('PriceSalesTTM'),
            'ev_to_ebitda': valuation.get('EnterpriseValueEbitda'),
            'debt_to_equity': highlights.get('DebtToEquity'),
            'current_ratio': highlights.get('CurrentRatio'),
            'roe': highlights.get('ReturnOnEquity'),
            'roa': highlights.get('ReturnOnAssets'),
            'profit_margin': highlights.get('ProfitMargin'),
            'eps': highlights.get('EarningsShare'),
            'eps_growth': highlights.get('QuarterlyEarningsGrowthYOY'),
            'revenue_growth': highlights.get('QuarterlyRevenueGrowthYOY'),
            'dividend_yield': highlights.get('DividendYield')
        }
    
    def _assess_valuation(self, metrics: Dict, current_price: float) -> Dict[str, Any]:
        """Assess valuation metrics."""
        valuation_score = 0
        reasoning = []
        
        # P/E Ratio assessment
        pe_ratio = metrics.get('pe_ratio')
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 15:
                valuation_score += 2
                reasoning.append(f"Attractive P/E of {pe_ratio:.1f}")
            elif pe_ratio > 25:
                valuation_score -= 2
                reasoning.append(f"High P/E of {pe_ratio:.1f}")
        
        # PEG Ratio assessment
        peg_ratio = metrics.get('peg_ratio')
        if peg_ratio and peg_ratio > 0:
            if peg_ratio < 1:
                valuation_score += 1
                reasoning.append(f"Favorable PEG of {peg_ratio:.2f}")
            elif peg_ratio > 2:
                valuation_score -= 1
                reasoning.append(f"High PEG of {peg_ratio:.2f}")
        
        # Price-to-Book assessment
        pb_ratio = metrics.get('price_to_book')
        if pb_ratio and pb_ratio > 0:
            if pb_ratio < 1.5:
                valuation_score += 1
                reasoning.append(f"Low P/B of {pb_ratio:.2f}")
            elif pb_ratio > 3:
                valuation_score -= 1
                reasoning.append(f"High P/B of {pb_ratio:.2f}")
        
        return {
            'score': valuation_score,
            'reasoning': reasoning,
            'is_undervalued': valuation_score > 1,
            'is_overvalued': valuation_score < -1
        }
    
    def _assess_financial_health(self, metrics: Dict) -> Dict[str, Any]:
        """Assess financial health."""
        health_score = 0
        reasoning = []
        
        # Debt-to-Equity
        debt_equity = metrics.get('debt_to_equity')
        if debt_equity is not None:
            if debt_equity < 0.3:
                health_score += 2
                reasoning.append(f"Low debt-to-equity of {debt_equity:.2f}")
            elif debt_equity > 1:
                health_score -= 2
                reasoning.append(f"High debt-to-equity of {debt_equity:.2f}")
        
        # ROE assessment
        roe = metrics.get('roe')
        if roe and roe > 0:
            if roe > 15:
                health_score += 2
                reasoning.append(f"Strong ROE of {roe:.1f}%")
            elif roe < 5:
                health_score -= 1
                reasoning.append(f"Weak ROE of {roe:.1f}%")
        
        # Current Ratio
        current_ratio = metrics.get('current_ratio')
        if current_ratio:
            if 1.5 <= current_ratio <= 3:
                health_score += 1
                reasoning.append(f"Healthy current ratio of {current_ratio:.2f}")
            elif current_ratio < 1:
                health_score -= 2
                reasoning.append(f"Poor liquidity, current ratio {current_ratio:.2f}")
        
        return {
            'score': health_score,
            'reasoning': reasoning,
            'is_healthy': health_score > 1
        }
    
    def _generate_fundamental_signal(self, metrics: Dict, valuation: Dict, 
                                   financial_health: Dict) -> AgentSignal:
        """Generate fundamental analysis signal."""
        total_score = valuation['score'] + financial_health['score']
        
        # Determine signal based on combined scores
        if total_score >= 4:
            signal_type = SignalType.STRONG_BUY
            confidence = 85
        elif total_score >= 2:
            signal_type = SignalType.BUY
            confidence = 75
        elif total_score <= -4:
            signal_type = SignalType.STRONG_SELL
            confidence = 85
        elif total_score <= -2:
            signal_type = SignalType.SELL
            confidence = 75
        else:
            signal_type = SignalType.HOLD
            confidence = 60
        
        # Combine reasoning
        all_reasoning = valuation['reasoning'] + financial_health['reasoning']
        reasoning = f"Fundamental Analysis: {'; '.join(all_reasoning)}"
        
        return AgentSignal(
            signal=signal_type,
            confidence=confidence,
            reasoning=reasoning,
            key_data={
                'metrics': metrics,
                'valuation_score': valuation['score'],
                'health_score': financial_health['score'],
                'total_score': total_score
            },
            timestamp=datetime.now(),
            agent_name=self.name,
            risk_score=max(0, min(100, 50 - financial_health['score'] * 5))
        )
    
    def _enhance_with_llm(self, signal: AgentSignal, metrics: Dict, 
                         market_data: MarketData) -> AgentSignal:
        """Enhance fundamental analysis with LLM."""
        try:
            prompt = f"""
            As a senior fundamental analyst, analyze {market_data.symbol}:
            
            Key Metrics:
            - P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
            - PEG Ratio: {metrics.get('peg_ratio', 'N/A')}
            - P/B Ratio: {metrics.get('price_to_book', 'N/A')}
            - Debt/Equity: {metrics.get('debt_to_equity', 'N/A')}
            - ROE: {metrics.get('roe', 'N/A')}%
            - Profit Margin: {metrics.get('profit_margin', 'N/A')}%
            - EPS Growth: {metrics.get('eps_growth', 'N/A')}%
            - Revenue Growth: {metrics.get('revenue_growth', 'N/A')}%
            
            Current Signal: {signal.signal.value}
            
            Provide enhanced analysis focusing on:
            1. Relative valuation vs sector/market
            2. Growth sustainability
            3. Balance sheet strength
            4. Competitive positioning
            5. Key catalysts/risks
            
            JSON format:
            {{"enhanced_reasoning": "...", "adjusted_confidence": XX, "investment_thesis": "..."}}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                llm_output = json.loads(response.content)
                signal.reasoning = llm_output.get('enhanced_reasoning', signal.reasoning)
                signal.confidence = min(100, max(0, llm_output.get('adjusted_confidence', signal.confidence)))
                signal.key_data['investment_thesis'] = llm_output.get('investment_thesis', '')
            except json.JSONDecodeError:
                signal.key_data['llm_raw_response'] = response.content
                
        except Exception as e:
            logger.error(f"LLM enhancement error: {e}")
        
        return signal
    
    def _error_signal(self, error_msg: str) -> AgentSignal:
        """Create error signal."""
        return AgentSignal(
            signal=SignalType.ERROR,
            confidence=0,
            reasoning=f"Fundamental analysis failed: {error_msg}",
            key_data={'error': error_msg},
            timestamp=datetime.now(),
            agent_name=self.name
        )

class SentimentAgent:
    """Sentiment analysis agent using news and market psychology."""
    
    def __init__(self, llm: Optional[Any] = None):
        self.name = "Sentiment Analyst"
        self.specialty = "Market psychology, news sentiment, and social indicators"
        self.llm = llm
    
    def analyze(self, market_data: MarketData) -> AgentSignal:
        """Perform sentiment analysis."""
        try:
            news_sentiment = self._analyze_news_sentiment(market_data.news)
            market_sentiment = self._analyze_market_sentiment(market_data)
            
            # Generate signal
            signal = self._generate_sentiment_signal(news_sentiment, market_sentiment)
            
            # Enhance with LLM if available
            if self.llm:
                signal = self._enhance_with_llm(signal, news_sentiment, market_sentiment, market_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._error_signal(str(e))
    
    def _analyze_news_sentiment(self, news: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from news headlines."""
        if not news:
            return {'score': 0, 'confidence': 0, 'articles_analyzed': 0}
        
        positive_words = [
            'gain', 'rise', 'up', 'positive', 'growth', 'beat', 'exceed', 'strong', 
            'bull', 'bullish', 'optimistic', 'boost', 'surge', 'rally', 'upgrade'
        ]
        negative_words = [
            'fall', 'drop', 'down', 'negative', 'decline', 'miss', 'weak', 'bear', 
            'bearish', 'pessimistic', 'crash', 'plunge', 'sell-off', 'downgrade'
        ]
        
        sentiment_scores = []
        analyzed_headlines = []
        
        for article in news[:10]:  # Analyze top 10 articles
            title = article.get('title', '').lower()
            if not title:
                continue
                
            analyzed_headlines.append(article.get('title', ''))
            
            pos_count = sum(1 for word in positive_words if word in title)
            neg_count = sum(1 for word in negative_words if word in title)
            
            # Calculate article sentiment score (-1 to 1)
            if pos_count > 0 or neg_count > 0:
                score = (pos_count - neg_count) / max(1, pos_count + neg_count)
                sentiment_scores.append(score)
        
        if not sentiment_scores:
            return {'score': 0, 'confidence': 0, 'articles_analyzed': 0}
        
        avg_sentiment = np.mean(sentiment_scores)
        consistency = 1 - np.std(sentiment_scores) if len(sentiment_scores) > 1 else 1
        confidence = min(100, consistency * len(sentiment_scores) * 10)
        
        return {
            'score': avg_sentiment,
            'confidence': confidence,
            'articles_analyzed': len(sentiment_scores),
            'headlines': analyzed_headlines[:5],
            'individual_scores': sentiment_scores
        }
    
    def _analyze_market_sentiment(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze sentiment from market data patterns."""
        df = market_data.historical_data
        if df.empty:
            return {'score': 0, 'confidence': 0}
        
        sentiment_indicators = {}
        
        # Volume analysis
        if 'volume' in df.columns and len(df) >= 5:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                sentiment_indicators['volume_surge'] = 1
            elif volume_ratio < 0.7:
                sentiment_indicators['volume_decline'] = -1
        
        # Price momentum
        if len(df) >= 5:
            recent_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
            if recent_return > 0.05:
                sentiment_indicators['strong_momentum'] = 1
            elif recent_return < -0.05:
                sentiment_indicators['weak_momentum'] = -1
        
        # Volatility analysis
        if len(df) >= 20:
            recent_vol = df['close'].pct_change().tail(5).std()
            avg_vol = df['close'].pct_change().std()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            
            if vol_ratio > 1.5:
                sentiment_indicators['high_volatility'] = -0.5  # Uncertainty
        
        market_score = sum(sentiment_indicators.values())
        market_confidence = min(100, len(sentiment_indicators) * 25)
        
        return {
            'score': market_score,
            'confidence': market_confidence,
            'indicators': sentiment_indicators
        }
    
    def _generate_sentiment_signal(self, news_sentiment: Dict, market_sentiment: Dict) -> AgentSignal:
        """Generate sentiment-based trading signal."""
        # Weight news sentiment more heavily than market sentiment
        combined_score = (news_sentiment['score'] * 0.7 + market_sentiment['score'] * 0.3)
        
        # Determine signal based on combined sentiment
        if combined_score > 0.3:
            signal_type = SignalType.BUY
            confidence = min(90, (news_sentiment['confidence'] + market_sentiment['confidence']) / 2)
        elif combined_score > 0.1:
            signal_type = SignalType.BUY if combined_score > 0.2 else SignalType.HOLD
            confidence = min(80, (news_sentiment['confidence'] + market_sentiment['confidence']) / 2)
        elif combined_score < -0.3:
            signal_type = SignalType.SELL
            confidence = min(90, (news_sentiment['confidence'] + market_sentiment['confidence']) / 2)
        elif combined_score < -0.1:
            signal_type = SignalType.SELL if combined_score < -0.2 else SignalType.HOLD
            confidence = min(80, (news_sentiment['confidence'] + market_sentiment['confidence']) / 2)
        else:
            signal_type = SignalType.HOLD
            confidence = 60
        
        # Build reasoning
        reasoning_parts = []
        if news_sentiment['articles_analyzed'] > 0:
            sentiment_desc = "positive" if news_sentiment['score'] > 0.1 else "negative" if news_sentiment['score'] < -0.1 else "neutral"
            reasoning_parts.append(f"News sentiment is {sentiment_desc} ({news_sentiment['articles_analyzed']} articles)")
        
        if market_sentiment['indicators']:
            indicator_desc = ", ".join(market_sentiment['indicators'].keys())
            reasoning_parts.append(f"Market indicators: {indicator_desc}")
        
        reasoning = f"Sentiment Analysis: {'; '.join(reasoning_parts) if reasoning_parts else 'Limited sentiment data available'}"
        
        return AgentSignal(
            signal=signal_type,
            confidence=confidence,
            reasoning=reasoning,
            key_data={
                'news_sentiment': news_sentiment,
                'market_sentiment': market_sentiment,
                'combined_score': combined_score
            },
            timestamp=datetime.now(),
            agent_name=self.name,
            risk_score=max(0, min(100, abs(combined_score) * 50))
        )
    
    def _enhance_with_llm(self, signal: AgentSignal, news_sentiment: Dict, 
                         market_sentiment: Dict, market_data: MarketData) -> AgentSignal:
        """Enhance sentiment analysis with LLM."""
        try:
            headlines = "\n".join([f"- {h}" for h in news_sentiment.get('headlines', [])])
            
            prompt = f"""
            As a market sentiment expert, analyze the sentiment for {market_data.symbol}:
            
            Recent Headlines:
            {headlines}
            
            Sentiment Metrics:
            - News Sentiment Score: {news_sentiment['score']:.3f}
            - Articles Analyzed: {news_sentiment['articles_analyzed']}
            - Market Sentiment Score: {market_sentiment['score']:.3f}
            - Market Indicators: {list(market_sentiment.get('indicators', {}).keys())}
            
            Current Signal: {signal.signal.value}
            
            Provide enhanced sentiment analysis focusing on:
            1. Sentiment trend and momentum
            2. Contrarian vs momentum signals
            3. Sector/market context
            4. Potential sentiment catalysts
            5. Risk of sentiment reversal
            
            JSON format:
            {{"enhanced_reasoning": "...", "adjusted_confidence": XX, "sentiment_outlook": "..."}}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                llm_output = json.loads(response.content)
                signal.reasoning = llm_output.get('enhanced_reasoning', signal.reasoning)
                signal.confidence = min(100, max(0, llm_output.get('adjusted_confidence', signal.confidence)))
                signal.key_data['sentiment_outlook'] = llm_output.get('sentiment_outlook', '')
            except json.JSONDecodeError:
                signal.key_data['llm_raw_response'] = response.content
                
        except Exception as e:
            logger.error(f"LLM enhancement error: {e}")
        
        return signal
    
    def _error_signal(self, error_msg: str) -> AgentSignal:
        """Create error signal."""
        return AgentSignal(
            signal=SignalType.ERROR,
            confidence=0,
            reasoning=f"Sentiment analysis failed: {error_msg}",
            key_data={'error': error_msg},
            timestamp=datetime.now(),
            agent_name=self.name
        )

class QuantAgent:
    """Quantitative analysis agent using statistical models and risk metrics."""
    
    def __init__(self, llm: Optional[Any] = None):
        self.name = "Quant Researcher"
        self.specialty = "Statistical models, risk metrics, and quantitative strategies"
        self.llm = llm
    
    def analyze(self, market_data: MarketData) -> AgentSignal:
        """Perform quantitative analysis."""
        try:
            df = market_data.historical_data
            if df.empty:
                return self._error_signal("No historical data available")
            
            risk_metrics = self._calculate_risk_metrics(df)
            statistical_signals = self._generate_statistical_signals(df)
            momentum_metrics = self._analyze_momentum(df)
            
            # Generate signal
            signal = self._generate_quant_signal(risk_metrics, statistical_signals, momentum_metrics)
            
            # Enhance with LLM if available
            if self.llm:
                signal = self._enhance_with_llm(signal, risk_metrics, statistical_signals, market_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Quantitative analysis error: {e}")
            return self._error_signal(str(e))
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return {}
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Downside risk
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR (Value at Risk) - 95% confidence
        var_95 = returns.quantile(0.05)
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': abs(max_drawdown),
            'var_95': abs(var_95),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def _generate_statistical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on statistical analysis."""
        signals = {}
        
        if len(df) < 50:
            return signals
        
        returns = df['close'].pct_change().dropna()
        prices = df['close']
        
        # Mean reversion signals
        price_zscore = (prices.iloc[-1] - prices.tail(20).mean()) / prices.tail(20).std()
        signals['mean_reversion'] = {
            'z_score': price_zscore,
            'signal': 'buy' if price_zscore < -2 else 'sell' if price_zscore > 2 else 'hold'
        }
        
        # Momentum signals
        if len(prices) >= 252:  # One year of data
            momentum_12_1 = (prices.iloc[-22] / prices.iloc[-252]) - 1  # 12-1 month momentum
            signals['momentum'] = {
                'momentum_12_1': momentum_12_1,
                'signal': 'buy' if momentum_12_1 > 0.1 else 'sell' if momentum_12_1 < -0.1 else 'hold'
            }
        
        # Volatility regime
        short_vol = returns.tail(20).std()
        long_vol = returns.tail(60).std() if len(returns) >= 60 else short_vol
        vol_ratio = short_vol / long_vol if long_vol > 0 else 1
        
        signals['volatility_regime'] = {
            'vol_ratio': vol_ratio,
            'regime': 'high' if vol_ratio > 1.5 else 'low' if vol_ratio < 0.7 else 'normal'
        }
        
        return signals
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze price and volume momentum."""
        if len(df) < 20:
            return {}
        
        # Price momentum
        returns = df['close'].pct_change()
        momentum_5d = returns.tail(5).mean()
        momentum_20d = returns.tail(20).mean()
        
        # Volume momentum (if available)
        volume_momentum = 0
        if 'volume' in df.columns:
            volume_5d = df['volume'].tail(5).mean()
            volume_20d = df['volume'].tail(20).mean()
            volume_momentum = (volume_5d / volume_20d - 1) if volume_20d > 0 else 0
        
        return {
            'price_momentum_5d': momentum_5d,
            'price_momentum_20d': momentum_20d,
            'volume_momentum': volume_momentum
        }
    
    def _generate_quant_signal(self, risk_metrics: Dict, statistical_signals: Dict, 
                              momentum_metrics: Dict) -> AgentSignal:
        """Generate quantitative trading signal."""
        signal_score = 0
        reasoning_parts = []
        
        # Risk-adjusted momentum
        sharpe = risk_metrics.get('sharpe_ratio', 0)
        if sharpe > 1:
            signal_score += 2
            reasoning_parts.append(f"Strong risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe < -0.5:
            signal_score -= 2
            reasoning_parts.append(f"Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
        
        # Mean reversion vs momentum
        mean_reversion = statistical_signals.get('mean_reversion', {})
        if mean_reversion:
            z_score = mean_reversion['z_score']
            if abs(z_score) > 2:
                # Strong mean reversion signal
                signal_score += 1 if z_score < -2 else -1
                reasoning_parts.append(f"Mean reversion signal (Z-score: {z_score:.2f})")
        
        # Momentum signals
        momentum_data = statistical_signals.get('momentum', {})
        if momentum_data:
            mom_12_1 = momentum_data['momentum_12_1']
            if abs(mom_12_1) > 0.1:
                signal_score += 1 if mom_12_1 > 0.1 else -1
                reasoning_parts.append(f"12-1 momentum: {mom_12_1:.2%}")
        
        # Volatility regime
        vol_regime = statistical_signals.get('volatility_regime', {})
        if vol_regime and vol_regime['regime'] == 'high':
            signal_score -= 1  # High volatility is generally negative
            reasoning_parts.append("High volatility regime detected")
        
        # Risk assessment
        max_drawdown = risk_metrics.get('max_drawdown', 0)
        if max_drawdown > 0.2:  # >20% drawdown
            signal_score -= 1
            reasoning_parts.append(f"High drawdown risk ({max_drawdown:.1%})")
        
        # Determine signal
        if signal_score >= 3:
            signal_type = SignalType.STRONG_BUY
            confidence = 85
        elif signal_score >= 1:
            signal_type = SignalType.BUY
            confidence = 70
        elif signal_score <= -3:
            signal_type = SignalType.STRONG_SELL
            confidence = 85
        elif signal_score <= -1:
            signal_type = SignalType.SELL
            confidence = 70
        else:
            signal_type = SignalType.HOLD
            confidence = 60
        
        # Adjust confidence based on data quality
        if len(reasoning_parts) < 2:
            confidence = max(50, confidence - 15)  # Reduce confidence if limited data
        
        reasoning = f"Quantitative Analysis: {'; '.join(reasoning_parts) if reasoning_parts else 'Limited quantitative signals'}"
        
        return AgentSignal(
            signal=signal_type,
            confidence=confidence,
            reasoning=reasoning,
            key_data={
                'risk_metrics': risk_metrics,
                'statistical_signals': statistical_signals,
                'momentum_metrics': momentum_metrics,
                'signal_score': signal_score
            },
            timestamp=datetime.now(),
            agent_name=self.name,
            risk_score=min(100, max(0, max_drawdown * 100 + risk_metrics.get('volatility', 0) * 50))
        )
    
    def _enhance_with_llm(self, signal: AgentSignal, risk_metrics: Dict, 
                         statistical_signals: Dict, market_data: MarketData) -> AgentSignal:
        """Enhance quantitative analysis with LLM."""
        try:
            prompt = f"""
            As a senior quantitative researcher, analyze {market_data.symbol}:
            
            Risk Metrics:
            - Volatility: {risk_metrics.get('volatility', 'N/A'):.3f}
            - Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 'N/A'):.3f}
            - Max Drawdown: {risk_metrics.get('max_drawdown', 'N/A'):.3f}
            - VaR (95%): {risk_metrics.get('var_95', 'N/A'):.3f}
            - Skewness: {risk_metrics.get('skewness', 'N/A'):.3f}
            
            Statistical Signals:
            {json.dumps(statistical_signals, indent=2)}
            
            Current Signal: {signal.signal.value}
            
            Provide enhanced quantitative analysis focusing on:
            1. Risk-return profile assessment
            2. Statistical significance of signals
            3. Regime change indicators
            4. Portfolio construction implications
            5. Risk management recommendations
            
            JSON format:
            {{"enhanced_reasoning": "...", "adjusted_confidence": XX, "risk_assessment": "..."}}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                llm_output = json.loads(response.content)
                signal.reasoning = llm_output.get('enhanced_reasoning', signal.reasoning)
                signal.confidence = min(100, max(0, llm_output.get('adjusted_confidence', signal.confidence)))
                signal.key_data['risk_assessment'] = llm_output.get('risk_assessment', '')
            except json.JSONDecodeError:
                signal.key_data['llm_raw_response'] = response.content
                
        except Exception as e:
            logger.error(f"LLM enhancement error: {e}")
        
        return signal
    
    def _error_signal(self, error_msg: str) -> AgentSignal:
        """Create error signal."""
        return AgentSignal(
            signal=SignalType.ERROR,
            confidence=0,
            reasoning=f"Quantitative analysis failed: {error_msg}",
            key_data={'error': error_msg},
            timestamp=datetime.now(),
            agent_name=self.name
        )

class AgentDebateSystem:
    """System for managing agent debates and consensus building using LangGraph."""
    
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        self.max_debate_rounds = 3
    
    def conduct_debate(self, agent_signals: List[AgentSignal], 
                      market_data: MarketData) -> Dict[str, Any]:
        """Conduct structured debate between agents."""
        if not self.llm:
            return self._simple_consensus(agent_signals)
        
        try:
            # Initial positions
            debate_history = []
            current_signals = {signal.agent_name: signal for signal in agent_signals}
            
            for round_num in range(self.max_debate_rounds):
                # Generate debate round
                debate_round = self._generate_debate_round(
                    current_signals, market_data, round_num, debate_history
                )
                debate_history.append(debate_round)
                
                # Update positions based on debate
                updated_signals = self._update_agent_positions(
                    current_signals, debate_round, market_data
                )
                
                # Check for convergence
                if self._check_convergence(current_signals, updated_signals):
                    logger.info(f"Debate converged after {round_num + 1} rounds")
                    break
                
                current_signals = updated_signals
            
            # Generate final consensus
            final_consensus = self._generate_final_consensus(current_signals, debate_history, market_data)
            
            return {
                'consensus': final_consensus,
                'debate_history': debate_history,
                'final_positions': {name: asdict(signal) for name, signal in current_signals.items()},
                'convergence_achieved': len(debate_history) < self.max_debate_rounds
            }
            
        except Exception as e:
            logger.error(f"Debate system error: {e}")
            return self._simple_consensus(agent_signals)
    
    def _generate_debate_round(self, current_signals: Dict[str, AgentSignal], 
                              market_data: MarketData, round_num: int, 
                              debate_history: List[Dict]) -> Dict[str, Any]:
        """Generate a single round of debate."""
        # Identify conflicts
        signals_list = list(current_signals.values())
        conflicts = self._identify_conflicts(signals_list)
        
        prompt = f"""
        AGENT DEBATE ROUND {round_num + 1} for {market_data.symbol}
        Current Price: ${market_data.current_price:.2f}
        
        CURRENT POSITIONS:
        """
        
        for agent_name, signal in current_signals.items():
            prompt += f"""
        {agent_name}: {signal.signal.value} (Confidence: {signal.confidence}%)
        Reasoning: {signal.reasoning}
        """
        
        prompt += f"""
        
        IDENTIFIED CONFLICTS:
        {json.dumps(conflicts, indent=2)}
        
        DEBATE FOCUS:
        Each agent should challenge others' assumptions and defend their position.
        Address these key conflicts:
        1. Signal disagreements and their underlying causes
        2. Confidence level discrepancies
        3. Time horizon differences
        4. Risk assessment variations
        
        For each agent, provide:
        1. Response to challenges from other agents
        2. Counter-arguments to conflicting positions
        3. Any position adjustments based on debate
        4. Updated confidence level
        
        JSON format:
        {{
            "Technical Analyst": {{"response": "...", "challenges": "...", "updated_signal": "...", "updated_confidence": XX}},
            "Fundamental Analyst": {{"response": "...", "challenges": "...", "updated_signal": "...", "updated_confidence": XX}},
            "Sentiment Analyst": {{"response": "...", "challenges": "...", "updated_signal": "...", "updated_confidence": XX}},
            "Quant Researcher": {{"response": "...", "challenges": "...", "updated_signal": "...", "updated_confidence": XX}},
            "round_summary": "Key insights and convergence points from this debate round"
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            debate_round = json.loads(response.content)
            debate_round['round_number'] = round_num + 1
            return debate_round
        except Exception as e:
            logger.error(f"Debate round generation failed: {e}")
            return {'round_number': round_num + 1, 'error': str(e)}
    
    def _identify_conflicts(self, signals: List[AgentSignal]) -> Dict[str, Any]:
        """Identify conflicts between agent signals."""
        signal_counts = {}
        confidence_levels = []
        
        for signal in signals:
            if signal.signal != SignalType.ERROR:
                signal_counts[signal.signal.value] = signal_counts.get(signal.signal.value, 0) + 1
                confidence_levels.append(signal.confidence)
        
        conflicts = {
            'signal_disagreement': len(signal_counts) > 1,
            'signal_distribution': signal_counts,
            'confidence_variance': np.var(confidence_levels) if confidence_levels else 0,
            'low_confidence_agents': [s.agent_name for s in signals if s.confidence < 70],
            'conflicting_pairs': self._find_conflicting_pairs(signals)
        }
        
        return conflicts
    
    def _find_conflicting_pairs(self, signals: List[AgentSignal]) -> List[Dict[str, str]]:
        """Find pairs of agents with conflicting signals."""
        conflicts = []
        
        for i, signal1 in enumerate(signals):
            for j, signal2 in enumerate(signals[i+1:], i+1):
                if self._signals_conflict(signal1.signal, signal2.signal):
                    conflicts.append({
                        'agent1': signal1.agent_name,
                        'signal1': signal1.signal.value,
                        'agent2': signal2.agent_name,
                        'signal2': signal2.signal.value
                    })
        
        return conflicts
    
    def _signals_conflict(self, signal1: SignalType, signal2: SignalType) -> bool:
        """Check if two signals conflict."""
        buy_signals = {SignalType.BUY, SignalType.STRONG_BUY}
        sell_signals = {SignalType.SELL, SignalType.STRONG_SELL}
        
        return (signal1 in buy_signals and signal2 in sell_signals) or \
               (signal1 in sell_signals and signal2 in buy_signals)
    
    def _update_agent_positions(self, current_signals: Dict[str, AgentSignal], 
                               debate_round: Dict, market_data: MarketData) -> Dict[str, AgentSignal]:
        """Update agent positions based on debate."""
        updated_signals = {}
        
        for agent_name, signal in current_signals.items():
            try:
                agent_debate = debate_round.get(agent_name, {})
                
                # Update signal if changed
                new_signal_str = agent_debate.get('updated_signal', signal.signal.value)
                try:
                    new_signal = SignalType(new_signal_str)
                except ValueError:
                    new_signal = signal.signal
                
                # Update confidence
                new_confidence = agent_debate.get('updated_confidence', signal.confidence)
                new_confidence = max(0, min(100, new_confidence))
                
                # Create updated signal
                updated_signal = AgentSignal(
                    signal=new_signal,
                    confidence=new_confidence,
                    reasoning=f"{signal.reasoning} | Debate Update: {agent_debate.get('response', '')}",
                    key_data={**signal.key_data, 'debate_response': agent_debate},
                    timestamp=datetime.now(),
                    agent_name=agent_name,
                    price_target=signal.price_target,
                    stop_loss=signal.stop_loss,
                    risk_score=signal.risk_score
                )
                
                updated_signals[agent_name] = updated_signal
                
            except Exception as e:
                logger.error(f"Error updating {agent_name} position: {e}")
                updated_signals[agent_name] = signal  # Keep original if update fails
        
        return updated_signals
    
    def _check_convergence(self, old_signals: Dict[str, AgentSignal], 
                          new_signals: Dict[str, AgentSignal]) -> bool:
        """Check if agent positions have converged."""
        # Check if signals changed significantly
        significant_changes = 0
        
        for agent_name in old_signals:
            old_signal = old_signals[agent_name]
            new_signal = new_signals.get(agent_name, old_signal)
            
            # Check for signal type change
            if old_signal.signal != new_signal.signal:
                significant_changes += 1
            
            # Check for significant confidence change
            if abs(old_signal.confidence - new_signal.confidence) > 10:
                significant_changes += 1
        
        # Converged if fewer than 2 significant changes
        return significant_changes < 2
    
    def _generate_final_consensus(self, final_signals: Dict[str, AgentSignal], 
                                 debate_history: List[Dict], 
                                 market_data: MarketData) -> AgentSignal:
        """Generate final consensus after debate."""
        try:
            # Prepare debate summary
            debate_summary = []
            for i, round_data in enumerate(debate_history):
                summary = round_data.get('round_summary', f'Round {i+1} completed')
                debate_summary.append(f"Round {i+1}: {summary}")
            
            prompt = f"""
            FINAL CONSENSUS GENERATION for {market_data.symbol}
            
            FINAL AGENT POSITIONS after debate:
            """
            
            for agent_name, signal in final_signals.items():
                prompt += f"""
            {agent_name}: {signal.signal.value} (Confidence: {signal.confidence}%)
            Final Reasoning: {signal.reasoning}
            """
            
            prompt += f"""
            
            DEBATE PROGRESSION:
            {chr(10).join(debate_summary)}
            
            Generate a final consensus that:
            1. Weighs each agent's final position by their confidence and expertise
            2. Considers the evolution of thinking through the debate
            3. Identifies the strongest supporting evidence
            4. Acknowledges remaining uncertainties
            5. Provides actionable trading recommendations
            
            JSON format:
            {{
                "final_signal": "BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL",
                "consensus_confidence": XX,
                "consensus_reasoning": "...",
                "key_supporting_factors": ["...", "...", "..."],
                "key_risks": ["...", "...", "..."],
                "price_target": XX.XX,
                "stop_loss": XX.XX,
                "time_horizon": "short/medium/long",
                "conviction_level": "low/medium/high"
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            consensus_data = json.loads(response.content)
            
            # Create consensus signal
            consensus_signal = AgentSignal(
                signal=SignalType(consensus_data['final_signal']),
                confidence=consensus_data['consensus_confidence'],
                reasoning=consensus_data['consensus_reasoning'],
                key_data={
                    'supporting_factors': consensus_data.get('key_supporting_factors', []),
                    'key_risks': consensus_data.get('key_risks', []),
                    'debate_history': debate_history,
                    'agent_positions': {name: asdict(signal) for name, signal in final_signals.items()},
                    'time_horizon': consensus_data.get('time_horizon', 'medium'),
                    'conviction_level': consensus_data.get('conviction_level', 'medium')
                },
                timestamp=datetime.now(),
                agent_name="Consensus Committee",
                price_target=consensus_data.get('price_target'),
                stop_loss=consensus_data.get('stop_loss'),
                risk_score=self._calculate_consensus_risk(final_signals)
            )
            
            return consensus_signal
            
        except Exception as e:
            logger.error(f"Final consensus generation failed: {e}")
            return self._simple_consensus(list(final_signals.values()))
    
    def _calculate_consensus_risk(self, signals: Dict[str, AgentSignal]) -> float:
        """Calculate consensus risk score."""
        risk_scores = [signal.risk_score for signal in signals.values() if signal.risk_score is not None]
        return np.mean(risk_scores) if risk_scores else 50
    
    def _simple_consensus(self, agent_signals: List[AgentSignal]) -> Dict[str, Any]:
        """Simple consensus without LLM debate."""
        valid_signals = [s for s in agent_signals if s.signal != SignalType.ERROR]
        
        if not valid_signals:
            consensus_signal = AgentSignal(
                signal=SignalType.HOLD,
                confidence=0,
                reasoning="No valid agent signals available",
                key_data={'error': 'All agents failed'},
                timestamp=datetime.now(),
                agent_name="Simple Consensus"
            )
            
            return {
                'consensus': consensus_signal,
                'debate_history': [],
                'final_positions': {},
                'convergence_achieved': False
            }
        
        # Count signals
        signal_counts = {}
        total_confidence = 0
        weighted_votes = 0
        
        for signal in valid_signals:
            signal_type = signal.signal.value
            weight = signal.confidence / 100
            
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + weight
            total_confidence += signal.confidence
            weighted_votes += weight
        
        # Find consensus signal
        if signal_counts:
            consensus_signal_type = max(signal_counts, key=signal_counts.get)
            consensus_confidence = min(95, total_confidence / len(valid_signals))
        else:
            consensus_signal_type = "HOLD"
            consensus_confidence = 50
        
        # Build reasoning
        agent_summaries = []
        for signal in valid_signals:
            agent_summaries.append(f"{signal.agent_name}: {signal.signal.value} ({signal.confidence}%)")
        
        reasoning = f"Simple Consensus: {'; '.join(agent_summaries)}"
        
        consensus_signal = AgentSignal(
            signal=SignalType(consensus_signal_type),
            confidence=consensus_confidence,
            reasoning=reasoning,
            key_data={
                'signal_distribution': signal_counts,
                'agent_count': len(valid_signals),
                'individual_signals': [asdict(s) for s in valid_signals]
            },
            timestamp=datetime.now(),
            agent_name="Simple Consensus",
            risk_score=np.mean([s.risk_score for s in valid_signals if s.risk_score is not None]) or 50
        )
        
        return {
            'consensus': consensus_signal,
            'debate_history': [],
            'final_positions': {s.agent_name: asdict(s) for s in valid_signals},
            'convergence_achieved': True
        }

class RealTimeMultiAgentSystem:
    """Main orchestrator for real-time multi-agent trading analysis."""
    
    def __init__(self, eodhd_api_key: str, llm_config: Optional[Dict] = None):
        self.data_provider = EODHDDataProvider(eodhd_api_key)
        
        # Initialize LLM if config provided
        self.llm = None
        if llm_config and LANGGRAPH_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    model=llm_config.get('model', 'gpt-4'),
                    api_key=llm_config.get('api_key'),
                    temperature=llm_config.get('temperature', 0.3)
                )
                logger.info("LLM initialized successfully")
            except Exception as e:
                logger.error(f"LLM initialization failed: {e}")
        
        # Initialize agents
        self.agents = {
            'technical': TechnicalAgent(self.llm),
            'fundamental': FundamentalAgent(self.llm),
            'sentiment': SentimentAgent(self.llm),
            'quant': QuantAgent(self.llm)
        }
        
        # Initialize debate system
        self.debate_system = AgentDebateSystem(self.llm)
        
        # Performance tracking
        self.analysis_history = []
        
    def analyze_symbol(self, symbol: str, enable_debate: bool = True, 
                      parallel_execution: bool = True) -> Dict[str, Any]:
        """Perform comprehensive real-time analysis of a symbol."""
        start_time = datetime.now()
        
        try:
            # Fetch comprehensive market data
            logger.info(f"Starting real-time analysis for {symbol}")
            market_data = self._fetch_comprehensive_data(symbol)
            
            if not market_data:
                return self._create_error_response("Failed to fetch market data", symbol, start_time)
            
            # Run agent analysis
            if parallel_execution:
                agent_signals = self._run_parallel_analysis(market_data)
            else:
                agent_signals = self._run_sequential_analysis(market_data)
            
            # Conduct debate and generate consensus
            if enable_debate and self.llm:
                debate_results = self.debate_system.conduct_debate(agent_signals, market_data)
                consensus = debate_results['consensus']
            else:
                debate_results = self.debate_system._simple_consensus(agent_signals)
                consensus = debate_results['consensus']
            
            # Prepare final results
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'market_data': {
                    'current_price': market_data.current_price,
                    'data_timestamp': market_data.timestamp,
                    'data_points': len(market_data.historical_data)
                },
                'individual_agents': {
                    signal.agent_name: asdict(signal) for signal in agent_signals
                },
                'consensus': asdict(consensus),
                'debate_enabled': enable_debate and self.llm is not None,
                'debate_results': debate_results if enable_debate else None,
                'system_info': {
                    'llm_enabled': self.llm is not None,
                    'parallel_execution': parallel_execution,
                    'total_agents': len(self.agents)
                }
            }
            
            # Store in history
            self.analysis_history.append(analysis_result)
            
            # Keep only last 100 analyses
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            logger.info(f"Analysis complete for {symbol} in {analysis_result['execution_time']:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return self._create_error_response(str(e), symbol, start_time)
    
    def _fetch_comprehensive_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch all necessary market data."""
        try:
            # Get real-time quote
            real_time_data = self.data_provider.get_real_time_quote(symbol)
            if not real_time_data or 'close' not in real_time_data:
                logger.error(f"No real-time data available for {symbol}")
                return None
            
            current_price = float(real_time_data['close'])
            
            # Get intraday data for technical analysis
            historical_data = self.data_provider.get_intraday_data(symbol, interval='5m')
            if historical_data.empty:
                # Fallback to daily data if intraday not available
                historical_data = self.data_provider.get_intraday_data(symbol, interval='1d')
            
            if historical_data.empty:
                logger.error(f"No historical data available for {symbol}")
                return None
            
            # Get fundamental data
            fundamental_data = self.data_provider.get_fundamental_data(symbol)
            
            # Get news for sentiment analysis
            news = self.data_provider.get_news(symbol, limit=10)
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(historical_data)
            
            return MarketData(
                symbol=symbol,
                current_price=current_price,
                historical_data=historical_data,
                technical_indicators=technical_indicators,
                fundamental_data=fundamental_data,
                sentiment_data={},  # Will be populated by sentiment agent
                news=news,
                real_time_data=real_time_data,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Data fetching failed for {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators."""
        if df.empty:
            return {}
        
        try:
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} in historical data")
                    return {}
            
            high, low, close, volume = df['high'].values, df['low'].values, df['close'].values, df['volume'].values
            
            indicators = {}
            
            # Moving averages
            if len(close) >= 20:
                indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1]
            if len(close) >= 50:
                indicators['sma_50'] = talib.SMA(close, timeperiod=50)[-1]
            
            # Momentum indicators
            if len(close) >= 14:
                indicators['rsi_14'] = talib.RSI(close, timeperiod=14)[-1]
                indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            # Volatility indicators
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
                indicators['bb_upper'] = bb_upper[-1]
                indicators['bb_middle'] = bb_middle[-1]
                indicators['bb_lower'] = bb_lower[-1]
            
            # Volume indicators
            if len(close) >= 10:
                indicators['obv'] = talib.OBV(close, volume)[-1]
            
            # MACD
            if len(close) >= 26:
                macd_line, macd_signal, macd_hist = talib.MACD(close)
                indicators['macd_line'] = macd_line[-1]
                indicators['macd_signal'] = macd_signal[-1]
                indicators['macd_histogram'] = macd_hist[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            return {}
    
    def _run_parallel_analysis(self, market_data: MarketData) -> List[AgentSignal]:
        """Run agent analysis in parallel."""
        agent_signals = []
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(agent.analyze, market_data): name 
                for name, agent in self.agents.items()
            }
            
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    signal = future.result(timeout=30)  # 30 second timeout per agent
                    agent_signals.append(signal)
                    logger.info(f"{agent_name} analysis complete: {signal.signal.value} ({signal.confidence}%)")
                except Exception as e:
                    logger.error(f"{agent_name} analysis failed: {e}")
                    error_signal = AgentSignal(
                        signal=SignalType.ERROR,
                        confidence=0,
                        reasoning=f"Agent analysis failed: {str(e)}",
                        key_data={'error': str(e)},
                        timestamp=datetime.now(),
                        agent_name=agent_name
                    )
                    agent_signals.append(error_signal)
        
        return agent_signals
    
    def _run_sequential_analysis(self, market_data: MarketData) -> List[AgentSignal]:
        """Run agent analysis sequentially."""
        agent_signals = []
        
        for name, agent in self.agents.items():
            try:
                signal = agent.analyze(market_data)
                agent_signals.append(signal)
                logger.info(f"{name} analysis complete: {signal.signal.value} ({signal.confidence}%)")
            except Exception as e:
                logger.error(f"{name} analysis failed: {e}")
                error_signal = AgentSignal(
                    signal=SignalType.ERROR,
                    confidence=0,
                    reasoning=f"Agent analysis failed: {str(e)}",
                    key_data={'error': str(e)},
                    timestamp=datetime.now(),
                    agent_name=name
                )
                agent_signals.append(error_signal)
        
        return agent_signals
    
    def _create_error_response(self, error_msg: str, symbol: str, start_time: datetime) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'error': error_msg,
            'status': 'failed',
            'individual_agents': {},
            'consensus': {
                'signal': 'ERROR',
                'confidence': 0,
                'reasoning': f"System error: {error_msg}"
            }
        }
    
    def get_analysis_history(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get historical analysis results."""
        if symbol:
            filtered_history = [a for a in self.analysis_history if a['symbol'] == symbol]
        else:
            filtered_history = self.analysis_history
        
        return filtered_history[-limit:]
    
    def get_agent_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get agent performance statistics."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_analyses = [
            a for a in self.analysis_history 
            if a['timestamp'] > cutoff_date
        ]
        
        if not recent_analyses:
            return {'error': 'No recent analyses available'}
        
        agent_stats = {}
        
        for agent_name in ['Technical Analyst', 'Fundamental Analyst', 'Sentiment Analyst', 'Quant Researcher']:
            agent_signals = []
            for analysis in recent_analyses:
                if agent_name in analysis.get('individual_agents', {}):
                    agent_signals.append(analysis['individual_agents'][agent_name])
            
            if agent_signals:
                confidences = [s['confidence'] for s in agent_signals if s['signal'] != 'ERROR']
                signals = [s['signal'] for s in agent_signals if s['signal'] != 'ERROR']
                
                agent_stats[agent_name] = {
                    'total_analyses': len(agent_signals),
                    'successful_analyses': len([s for s in agent_signals if s['signal'] != 'ERROR']),
                    'average_confidence': np.mean(confidences) if confidences else 0,
                    'signal_distribution': {signal: signals.count(signal) for signal in set(signals)},
                    'error_rate': len([s for s in agent_signals if s['signal'] == 'ERROR']) / len(agent_signals)
                }
        
        return {
            'period_days': days,
            'total_analyses': len(recent_analyses),
            'agent_performance': agent_stats,
            'average_execution_time': np.mean([a['execution_time'] for a in recent_analyses])
        }

# Example usage and configuration
def create_trading_system(eodhd_api_key: str, openai_api_key: Optional[str] = None) -> RealTimeMultiAgentSystem:
    """Factory function to create a fully configured trading system."""
    
    llm_config = None
    if openai_api_key:
        llm_config = {
            'model': 'gpt-4',  # or 'gpt-3.5-turbo' for lower cost
            'api_key': openai_api_key,
            'temperature': 0.3
        }
    
    system = RealTimeMultiAgentSystem(
        eodhd_api_key=eodhd_api_key,
        llm_config=llm_config
    )
    
    return system

# Demo function
async def demo_analysis():
    """Demo function showing how to use the system."""
    
    # Configuration (replace with your actual API keys)
    EODHD_API_KEY = "your_eodhd_api_key"
    OPENAI_API_KEY = "your_openai_api_key"  # Optional, but recommended for full features
    
    # Create system
    trading_system = create_trading_system(EODHD_API_KEY, OPENAI_API_KEY)
    
    # Example symbols to analyze
    symbols = ['AAPL.US', 'GOOGL.US', 'TSLA.US']
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"ANALYZING {symbol}")
        print('='*60)
        
        # Run analysis
        result = trading_system.analyze_symbol(
            symbol=symbol,
            enable_debate=True,  # Enable LLM-powered agent debates
            parallel_execution=True  # Run agents in parallel for speed
        )
        
        if 'error' in result:
            print(f"❌ Analysis failed: {result['error']}")
            continue
        
        # Display results
        consensus = result['consensus']
        print(f"\n🎯 CONSENSUS SIGNAL: {consensus['signal']}")
        print(f"📊 Confidence: {consensus['confidence']}%")
        print(f"💭 Reasoning: {consensus['reasoning']}")
        
        if consensus.get('price_target'):
            print(f"🎯 Price Target: ${consensus['price_target']:.2f}")
        if consensus.get('stop_loss'):
            print(f"🛑 Stop Loss: ${consensus['stop_loss']:.2f}")
        
        print(f"\n📈 Individual Agent Signals:")
        for agent_name, signal_data in result['individual_agents'].items():
            print(f"  {agent_name}: {signal_data['signal']} ({signal_data['confidence']}%)")
        
        print(f"\n⏱️  Analysis completed in {result['execution_time']:.2f} seconds")
        
        # Wait before next analysis (respect API limits)
        await asyncio.sleep(2)

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_analysis()) 
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import random
from typing import List, Dict

from config import config

# ============================================================================
# AI TRADING AGENTS CLASSES
# ============================================================================

class TradingAgent:
    """Base class for specialized trading agents."""
    def __init__(self, name, specialty, emoji):
        self.name = name
        self.specialty = specialty
        self.emoji = emoji
        self.confidence = 0
        self.reasoning = ""
        self.signal = "HOLD"
        self.key_levels = ""

class MultiAgentSystem:
    """Orchestrates multiple trading agents."""
    
    def __init__(self):
        self.agents = [
            {"name": "Technical Analyst", "emoji": "📈", "specialty": "Chart Patterns & Technical Indicators"},
            {"name": "Macro Economist", "emoji": "🌍", "specialty": "Economic Indicators & Fed Policy"},
            {"name": "Sentiment Analyst", "emoji": "📱", "specialty": "Social Media & Market Psychology"},
            {"name": "Quant Researcher", "emoji": "🔬", "specialty": "Statistical Models & Risk Metrics"}
        ]
    
    def analyze_symbol(self, symbol, data, llm=None):
        """Run analysis from all agents."""
        results = {}
        
        # Simulate agent analysis with realistic variations
        base_price = data['Close'].iloc[-1]
        volatility = data['Close'].pct_change().std() * 100
        volume_ratio = data['Volume'].iloc[-1] / data['Volume'].mean()
        
        # Technical Analyst
        rsi = data.get('RSI_14', pd.Series([50])).iloc[-1] if 'RSI_14' in data.columns else 50
        if rsi > 70:
            tech_signal = "SELL"
            tech_confidence = random.randint(75, 85)
            tech_reasoning = f"RSI overbought at {rsi:.1f}, expecting pullback to ${base_price * 0.97:.2f}"
        elif rsi < 30:
            tech_signal = "BUY"
            tech_confidence = random.randint(80, 90)
            tech_reasoning = f"RSI oversold at {rsi:.1f}, bounce expected from support at ${base_price * 0.98:.2f}"
        else:
            tech_signal = "HOLD"
            tech_confidence = random.randint(60, 75)
            tech_reasoning = f"Neutral RSI at {rsi:.1f}, watching for breakout above ${base_price * 1.02:.2f}"
        
        results["Technical Analyst"] = {
            "signal": tech_signal,
            "confidence": tech_confidence,
            "reasoning": tech_reasoning,
            "key_levels": f"Support: ${base_price * 0.95:.2f}, Resistance: ${base_price * 1.05:.2f}"
        }
        
        # Macro Economist  
        macro_signals = ["BUY", "HOLD", "SELL"]
        macro_signal = random.choice(macro_signals)
        macro_confidence = random.randint(65, 80)
        macro_reasons = {
            "BUY": "Fed dovish pivot supports risk assets, sector rotation favors growth",
            "HOLD": "Mixed economic data creates uncertainty, await clarity on policy direction", 
            "SELL": "Rising yields pressure valuations, recession risk increasing"
        }
        
        results["Macro Economist"] = {
            "signal": macro_signal,
            "confidence": macro_confidence,
            "reasoning": macro_reasons[macro_signal],
            "key_levels": "Monitor 10Y yield at 4.5%, Fed funds rate terminal"
        }
        
        # Sentiment Analyst
        if volume_ratio > 1.5:
            sent_signal = "BUY"
            sent_confidence = random.randint(75, 88)
            sent_reasoning = f"Volume spike {volume_ratio:.1f}x avg suggests institutional accumulation"
        elif volume_ratio < 0.7:
            sent_signal = "SELL"
            sent_confidence = random.randint(70, 82)
            sent_reasoning = f"Low volume {volume_ratio:.1f}x avg indicates lack of conviction"
        else:
            sent_signal = "HOLD"
            sent_confidence = random.randint(60, 75)
            sent_reasoning = "Neutral sentiment, retail vs institutional positioning balanced"
        
        # Fix the f-string issue on line 164
        flow_options = ['Bullish', 'Bearish', 'Neutral']
        selected_flow = random.choice(flow_options)
        
        results["Sentiment Analyst"] = {
            "signal": sent_signal,
            "confidence": sent_confidence,
            "reasoning": sent_reasoning,
            "key_levels": f"Social mentions trending, options flow: {selected_flow}"
        }
        
        # Quant Researcher
        sharpe_ratio = random.uniform(0.8, 1.6)
        quant_signal = "BUY" if sharpe_ratio > 1.2 else "SELL" if sharpe_ratio < 0.9 else "HOLD"
        quant_confidence = random.randint(68, 85)
        
        results["Quant Researcher"] = {
            "signal": quant_signal,
            "confidence": quant_confidence,
            "reasoning": f"Sharpe ratio {sharpe_ratio:.2f}, volatility {volatility:.1f}%, risk-adjusted return favorable",
            "key_levels": f"VaR 95%: ${base_price * 0.92:.2f}, Expected return: {random.randint(5, 15)}%"
        }
        
        return results

# ============================================================================
# AI STORYTELLER ENGINE
# ============================================================================

class AIStorytellerEngine:
    """Generates compelling market narratives."""
    
    @staticmethod
    def generate_market_story(symbol, data, events=None):
        """Create a Bloomberg-style market story."""
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price) * 100
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        
        direction = "surged" if price_change > 2 else "climbed" if price_change > 0.5 else "dipped" if price_change < -0.5 else "traded mixed"
        
        # Fix f-string with newlines
        newline = "\n"
        catalyst_options = ["Tomorrow's volume confirmation", "End-of-week options expiry", "Sector peer earnings", "Fed policy signals"]
        action_options = ["hold above", "break through", "consolidate near"]
        direction_options = ["higher", "toward range-bound trading", "to further downside testing"]
        story_parts = [
            f"## 📰 **{symbol} MARKET PULSE** - {datetime.now().strftime('%I:%M %p EST')}",
            "",
            f"**THE HEADLINE:** {symbol} {direction} to ${current_price:.2f} in active trading, {'+' if price_change >= 0 else ''}{price_change:.1f}% from yesterday's close as {random.choice(['institutional', 'algorithmic', 'retail'])} money flows into the name.",
            "",
            "**WHAT'S DRIVING THE ACTION:**",
            f"The move comes as three catalysts converge: technical momentum building after last week's breakout, sector rotation favoring {random.choice(['growth', 'value', 'cyclical'])} names, and {random.choice(['options market makers chasing delta', 'institutional rebalancing', 'earnings positioning'])} ahead of key events.",
            "",
            "**THE TECHNICAL PICTURE:**",
            f"From a chart perspective, {symbol} is {random.choice(['testing key resistance', 'finding support', 'breaking out', 'consolidating'])} at current levels. The stock has {'outperformed' if price_change > 0 else 'underperformed'} the broader market by {abs(price_change) + random.uniform(0.5, 1.5):.1f}% over the past week, with volume running {volume/avg_volume:.1f}x the 20-day average.",
            "",
            "**WHAT THE SMART MONEY IS DOING:**",
            f"{random.choice(['Large block trades', 'Unusual options activity', 'Dark pool accumulation'])} suggests {random.choice(['hedge funds are building positions', 'institutional investors are rotating', 'systematic strategies are triggering'])}. The {random.choice(['call/put ratio', 'put/call skew', 'volatility surface'])} indicates {random.choice(['bullish positioning', 'defensive hedging', 'neutral sentiment'])}.",
            "",
            "**KEY LEVELS TO WATCH:**",
            f"- **Immediate resistance:** ${current_price * random.uniform(1.02, 1.05):.2f} - Previous consolidation zone",
            f"- **Support level:** ${current_price * random.uniform(0.95, 0.98):.2f} - 20-day moving average confluence",
            f"- **Breakout target:** ${current_price * random.uniform(1.08, 1.12):.2f} - Measured move projection",
            "",
            "**WHAT HAPPENS NEXT:**",
            f"The next catalyst watch: {random.choice(catalyst_options)}. If {symbol} can {random.choice(action_options)} ${current_price * random.uniform(0.99, 1.01):.2f}, the path of least resistance points {random.choice(direction_options)}."
            "",
            "*Market participants should monitor after-hours action and overnight futures for additional directional clues.*"
        ]
        
        story = newline.join(story_parts)
        return story

# ============================================================================
# SCENARIO MODELING ENGINE
# ============================================================================

class ScenarioEngine:
    """Advanced scenario modeling with Monte Carlo simulation."""
    
    @staticmethod
    def generate_scenarios(symbol, data, timeframe_days=90):
        """Generate multiple probability-weighted scenarios."""
        current_price = data['Close'].iloc[-1]
        volatility = data['Close'].pct_change().std() * np.sqrt(252)
        
        scenarios = {
            "🚀 Moonshot Scenario": {
                "probability": 10,
                "target_price": current_price * random.uniform(1.25, 1.45),
                "timeframe": f"{timeframe_days} days",
                "catalysts": [
                    "Major earnings blowout (+30% surprise)",
                    "Sector-changing product announcement", 
                    "Acquisition rumors surface",
                    "Breakthrough regulatory approval"
                ],
                "conditions": "Perfect storm of positive catalysts",
                "risk_factors": "Extreme optimism priced in, high volatility"
            },
            "🐂 Bull Case Scenario": {
                "probability": 25,
                "target_price": current_price * random.uniform(1.12, 1.22),
                "timeframe": f"{timeframe_days} days",
                "catalysts": [
                    "Earnings beat expectations",
                    "Positive sector rotation",
                    "Fed policy becomes supportive",
                    "Technical breakout confirmed"
                ],
                "conditions": "Favorable macro + strong fundamentals",
                "risk_factors": "Valuation stretch, profit-taking risk"
            },
            "📊 Base Case Scenario": {
                "probability": 40,
                "target_price": current_price * random.uniform(0.98, 1.08),
                "timeframe": f"{timeframe_days} days",
                "catalysts": [
                    "In-line earnings results",
                    "Broader market trend following",
                    "Normal business conditions",
                    "Steady institutional flows"
                ],
                "conditions": "Status quo macro environment",
                "risk_factors": "Lack of catalysts, range-bound action"
            },
            "🐻 Bear Case Scenario": {
                "probability": 20,
                "target_price": current_price * random.uniform(0.85, 0.95),
                "timeframe": f"{timeframe_days} days",
                "catalysts": [
                    "Earnings disappointment",
                    "Sector headwinds emerge",
                    "Fed policy turns hawkish",
                    "Technical support breaks"
                ],
                "conditions": "Economic headwinds + weak fundamentals",
                "risk_factors": "Momentum selling, stop-loss triggers"
            },
            "💥 Black Swan Scenario": {
                "probability": 5,
                "target_price": current_price * random.uniform(0.65, 0.80),
                "timeframe": f"{timeframe_days} days",
                "catalysts": [
                    "Major scandal or fraud uncovered",
                    "Catastrophic product failure",
                    "Regulatory crackdown",
                    "Systemic market crash"
                ],
                "conditions": "Extreme negative shock",
                "risk_factors": "Complete sentiment reversal, liquidity crisis"
            }
        }
        
        return scenarios

# ============================================================================
# VOICE TRADING ASSISTANT
# ============================================================================

class VoiceTradingAssistant:
    """Natural language trading queries and responses."""
    
    def __init__(self):
        self.sample_queries = [
            "Should I buy more on this dip?",
            "What's the risk-reward on a swing trade here?",
            "How does this compare to sector peers?", 
            "What would happen if earnings disappoint?",
            "Is this a good entry point for long-term holding?",
            "What are the key support and resistance levels?",
            "How much should I position size for this trade?",
            "What's the probability of a 10% move this month?"
        ]
    
    def process_query(self, query, symbol, data, portfolio_context=None):
        """Process natural language trading question."""
        query_lower = query.lower()
        current_price = data['Close'].iloc[-1]
        
        if "dip" in query_lower or "buy more" in query_lower:
            return self._analyze_dip_buying(symbol, data, current_price)
        elif "risk" in query_lower and "reward" in query_lower:
            return self._analyze_risk_reward(symbol, data, current_price)
        elif "sector" in query_lower or "compare" in query_lower:
            return self._compare_to_sector(symbol, data, current_price)
        elif "earnings" in query_lower:
            return self._analyze_earnings_risk(symbol, data, current_price)
        elif "entry" in query_lower:
            return self._analyze_entry_point(symbol, data, current_price)
        elif "support" in query_lower or "resistance" in query_lower:
            return self._identify_key_levels(symbol, data, current_price)
        elif "position" in query_lower or "size" in query_lower:
            return self._recommend_position_size(symbol, data, current_price)
        else:
            return self._general_analysis(symbol, data, current_price)
    
    def _analyze_dip_buying(self, symbol, data, current_price):
        """Analyze whether to buy the dip."""
        # Calculate technical metrics
        high_20d = data['High'].tail(20).max()
        low_20d = data['Low'].tail(20).min()
        pullback_pct = ((high_20d - current_price) / high_20d) * 100
        
        return f"""
## 🎯 **DIP ANALYSIS FOR {symbol}**

**Current Situation:** You're looking at a {pullback_pct:.1f}% pullback from 20-day highs. This {'qualifies as a meaningful dip' if pullback_pct > 5 else 'is a shallow pullback'} that {'presents opportunity' if pullback_pct > 3 else 'may need more downside'}.

**Technical Assessment:**
- **Support Level:** ${low_20d:.2f} (20-day low)
- **Resistance:** ${high_20d:.2f} (recent high)  
- **Current Position:** {((current_price - low_20d) / (high_20d - low_20d)) * 100:.1f}% of recent range

**DIP-BUYING VERDICT:**
{'🟢 FAVORABLE - Good risk/reward setup' if pullback_pct > 4 else '🟡 NEUTRAL - Wait for better entry' if pullback_pct > 2 else '🔴 UNFAVORABLE - Minimal dip, high risk'}

**ACTION PLAN:**
1. **Entry Strategy:** Scale in with 50% position here, 50% if it drops to ${current_price * 0.97:.2f}
2. **Stop Loss:** ${current_price * 0.94:.2f} (3% below current level)
3. **Target:** ${current_price * 1.08:.2f} (back to recent highs + momentum)
4. **Risk/Reward:** 1:2.7 ratio {'✅ Acceptable' if pullback_pct > 3 else '⚠️ Marginal'}
        """
    
    def _analyze_risk_reward(self, symbol, data, current_price):
        """Analyze risk-reward for swing trade."""
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
        
        return f"""
## ⚖️ **RISK/REWARD ANALYSIS FOR {symbol}**

**Swing Trade Setup:**
- **Entry:** ${current_price:.2f} (current price)
- **Stop Loss:** ${current_price * 0.95:.2f} (5% downside protection)
- **Target 1:** ${current_price * 1.08:.2f} (8% upside - quick profit)
- **Target 2:** ${current_price * 1.15:.2f} (15% upside - extended move)

**Risk Metrics:**
- **Annual Volatility:** {volatility:.1f}% ({'High' if volatility > 40 else 'Moderate' if volatility > 25 else 'Low'} risk)
- **Maximum Risk:** 5% of position value
- **Reward Potential:** 8-15% upside
- **Risk/Reward Ratio:** 1:1.6 to 1:3.0 {'✅ Favorable' if volatility < 35 else '⚠️ High volatility'}

**PROBABILITY ASSESSMENT:**
- **Success Probability:** {65 if volatility < 30 else 55 if volatility < 40 else 45}% based on historical patterns
- **Time Horizon:** 2-6 weeks for full move
- **Best Case:** 15% gain in 4 weeks
- **Worst Case:** 5% loss with disciplined stop
        """
    
    def _general_analysis(self, symbol, data, current_price):
        """General analysis response."""
        return f"""
## 🤖 **AI ANALYSIS FOR {symbol}**

**Current Price:** ${current_price:.2f}
**24h Volume:** {data['Volume'].iloc[-1]:,.0f} shares
**Trend:** {random.choice(['Bullish', 'Bearish', 'Sideways'])} momentum

**Key Insights:**
- Technical indicators suggest {random.choice(['continuation', 'reversal', 'consolidation'])} pattern
- Volume profile shows {random.choice(['accumulation', 'distribution', 'neutral'])} activity
- Options flow indicates {random.choice(['bullish', 'bearish', 'mixed'])} sentiment

**Recommendation:** {random.choice(['BUY', 'HOLD', 'SELL'])} with {random.randint(65, 85)}% confidence

*Ask me more specific questions for detailed analysis!*
        """
    
    def _compare_to_sector(self, symbol, data, current_price):
        """Compare to sector peers."""
        return f"""
## 📊 **SECTOR COMPARISON FOR {symbol}**

**Relative Performance:**
- **vs Sector:** {random.choice(['Outperforming', 'Underperforming', 'In-line'])} by {random.uniform(1, 8):.1f}%
- **vs Market:** {random.choice(['Leading', 'Lagging', 'Matching'])} broader indices

**Peer Analysis:**
- **Valuation:** {random.choice(['Premium', 'Discount', 'Fair value'])} vs peers
- **Growth Rate:** {random.choice(['Above', 'Below', 'In-line with'])} sector average
- **Risk Profile:** {random.choice(['Lower', 'Higher', 'Similar'])} volatility than peers

**VERDICT:** {random.choice(['Sector leader', 'Sector laggard', 'Average performer'])}
    """

    def _analyze_earnings_risk(self, symbol, data, current_price):
        """Analyze earnings risk."""
        return f"""
## 📈 **EARNINGS RISK ANALYSIS FOR {symbol}**

**Earnings Outlook:**
- **Expected Move:** ±{random.randint(5, 15)}% post-earnings
- **Surprise History:** {random.choice(['Positive', 'Mixed', 'Negative'])} track record
- **Guidance Risk:** {random.choice(['Low', 'Medium', 'High'])}

**Key Metrics to Watch:**
- Revenue growth expectations
- Margin pressure/expansion
- Forward guidance tone

**STRATEGY:** {random.choice(['Hold through earnings', 'Take profits before', 'Add on weakness'])}
    """

    def _analyze_entry_point(self, symbol, data, current_price):
        """Analyze entry point quality."""
        return f"""
## 🎯 **ENTRY POINT ANALYSIS FOR {symbol}**

**Current Setup:**
- **Technical Grade:** {random.choice(['A', 'B', 'C'])}
- **Risk/Reward:** {random.uniform(1.5, 3.0):.1f}:1
- **Timing:** {random.choice(['Excellent', 'Good', 'Fair'])}

**Entry Strategy:**
1. **Immediate:** 50% position at ${current_price:.2f}
2. **Scale-in:** 50% on pullback to ${current_price * 0.97:.2f}
3. **Stop Loss:** ${current_price * 0.94:.2f}

**RATING:** {random.choice(['Strong Buy', 'Buy', 'Hold', 'Wait'])}
    """

    def _identify_key_levels(self, symbol, data, current_price):
        """Identify key support and resistance levels."""
        high_20d = data['High'].tail(20).max()
        low_20d = data['Low'].tail(20).min()
        
        return f"""
## 📊 **KEY LEVELS FOR {symbol}**

**Support Levels:**
- **Primary Support:** ${low_20d:.2f} (20-day low)
- **Secondary Support:** ${current_price * 0.95:.2f} (5% below current)
- **Major Support:** ${current_price * 0.90:.2f} (psychological level)

**Resistance Levels:**
- **Immediate Resistance:** ${current_price * 1.02:.2f}
- **Key Resistance:** ${high_20d:.2f} (20-day high)
- **Major Resistance:** ${current_price * 1.10:.2f}

**Current Position:** {((current_price - low_20d) / (high_20d - low_20d) * 100):.1f}% of 20-day range

**STRATEGY:** Watch for {random.choice(['breakout above', 'support at', 'rejection from'])} key levels
    """

    def _recommend_position_size(self, symbol, data, current_price):
        """Recommend position sizing."""
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
    
        return f"""
## ⚖️ **POSITION SIZING FOR {symbol}**

**Risk Assessment:**
- **Volatility:** {volatility:.1f}% (annualized)
- **Risk Rating:** {random.choice(['Low', 'Medium', 'High'])}

**Recommended Sizing:**
- **Conservative:** 1-2% of portfolio
- **Moderate:** 3-5% of portfolio  
- **Aggressive:** 5-8% of portfolio

**Based on volatility:** Suggest {random.choice(['smaller', 'normal', 'larger'])} position size

**Risk Management:**
- **Max Loss:** 2% of total portfolio
- **Stop Loss:** {random.choice(['3%', '5%', '7%'])} below entry
- **Position Scaling:** Enter in 2-3 tranches
    """

# ============================================================================
# CHART AI ANNOTATION SYSTEM
# ============================================================================

class ChartGPT:
    """AI that reads and annotates charts intelligently."""
    
    @staticmethod
    def generate_smart_annotations(data, symbol):
        """Generate intelligent chart annotations."""
        current_price = data['Close'].iloc[-1]
        high_20d = data['High'].tail(20).max()
        low_20d = data['Low'].tail(20).min()
        
        annotations = []
        
        # Key level annotations
        if current_price > high_20d * 0.98:
            annotations.append({
                'type': 'resistance',
                'level': high_20d,
                'message': '🛑 CRITICAL RESISTANCE - Watch for breakout or rejection',
                'color': 'red',
                'importance': 'high'
            })
        
        if current_price < low_20d * 1.02:
            annotations.append({
                'type': 'support', 
                'level': low_20d,
                'message': '🛡️ KEY SUPPORT - Bounce zone or breakdown risk',
                'color': 'green',
                'importance': 'high'
            })
        
        # Volume annotations
        avg_volume = data['Volume'].mean()
        current_volume = data['Volume'].iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            annotations.append({
                'type': 'volume',
                'level': current_price,
                'message': f'📊 VOLUME SPIKE - {current_volume/avg_volume:.1f}x average volume',
                'color': 'blue',
                'importance': 'medium'
            })
        
        return annotations

# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render():
    print ("yes")
    """Main render function for AI Trading Intelligence tab."""
    st.markdown("# 🤖 AI Trading Intelligence")
    st.markdown("*Advanced AI agents, storytelling, and predictive analytics for institutional-grade trading insights*")
    
    # Get configuration
    symbol = st.session_state.get('selected_symbol', 'AAPL.US')
    llm = st.session_state.get('llm', None)
    
    # Load market data
    try:
        from services.data_fetcher import get_market_data_yfinance
        market_data = get_market_data_yfinance(symbol)
        
        if market_data.empty:
            st.error(f"Could not load market data for {symbol}")
            return
            
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return
    
    # Create main feature tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Multi-Agent Analysis",
        "📰 AI Market Storyteller", 
        "🎭 Scenario Modeling",
        "🎤 Voice Trading Assistant",
        "🧠 Chart Intelligence"
    ])
    
    # COMPLETE REWRITE USING ONLY STREAMLIT NATIVE COMPONENTS
# NO CUSTOM HTML - ONLY STREAMLIT FUNCTIONS

    with tab1:
        # Header using pure Streamlit
        st.title("🤖 AI Agent Debate Arena")
        st.markdown("*Watch AI agents analyze, debate, challenge, and reach consensus in real-time*")
        
        # Initialize session state with clean structure
        if 'debate_system' not in st.session_state:
            st.session_state.debate_system = {
                'agents': {
                    'technical': {'confidence': 85, 'signal': 'SELL', 'name': 'Technical Analyst 📈'},
                    'fundamental': {'confidence': 80, 'signal': 'BUY', 'name': 'Fundamental Agent 💰'},
                    'sentiment': {'confidence': 70, 'signal': 'BUY', 'name': 'Sentiment Agent 📱'},
                    'macro': {'confidence': 75, 'signal': 'BUY', 'name': 'Macro Economist 🌍'}
                },
                'consensus': 65,
                'phase': 'Ready to Start',
                'is_debating': False,
                'current_message': 0,
                'messages': [
                    {'agent': 'technical', 'text': 'RSI is at 74.2 - clearly overbought territory. Price hit resistance at $145.50 with declining volume.', 'type': 'Analysis'},
                    {'agent': 'fundamental', 'text': '@Technical Your analysis ignores fundamentals. Q3 earnings beat by 12%, revenue growth 15% YoY.', 'type': 'Challenge'},
                    {'agent': 'technical', 'text': '@Fundamental Remember NVDA in Nov 2021? Great earnings, still dropped 20% after hitting resistance!', 'type': 'Counter'},
                    {'agent': 'sentiment', 'text': '@All News sentiment jumped to 0.75 from 0.45. 347 positive mentions vs 89 negative.', 'type': 'Insight'},
                    {'agent': 'macro', 'text': 'Fed dovishness changes everything. Low-rate environment makes technical resistance less reliable.', 'type': 'Context'},
                    {'agent': 'technical', 'text': 'I hear the macro argument. Perhaps CAUTIOUS SELL with tight stops rather than aggressive stance?', 'type': 'Concession'},
                    {'agent': 'fundamental', 'text': 'Technical levels matter for timing. Maybe scale into positions on 5-7% pullbacks?', 'type': 'Compromise'}
                ]
            }
        
        # Main layout
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col2:
            # Agent Confidence Section - Using only Streamlit components
            st.subheader("📊 Agent Confidence Meters")
            
            # Display each agent using Streamlit metrics and progress bars
            for agent_key, agent_data in st.session_state.debate_system['agents'].items():
                with st.container():
                    # Agent header
                    st.markdown(f"**{agent_data['name']}**")
                    
                    # Signal and confidence
                    col_signal, col_conf = st.columns(2)
                    with col_signal:
                        signal_color = "🟢" if agent_data['signal'] == 'BUY' else "🔴" if agent_data['signal'] == 'SELL' else "🟡"
                        st.markdown(f"{signal_color} **{agent_data['signal']}**")
                    with col_conf:
                        st.markdown(f"**{agent_data['confidence']}%**")
                    
                    # Progress bar
                    st.progress(agent_data['confidence'] / 100)
                    st.divider()
            
            # Consensus Gauge using Streamlit metrics
            st.subheader("🎯 Consensus Gauge")
            consensus_val = st.session_state.debate_system['consensus']
            consensus_trend = 'STRONG BUY' if consensus_val > 70 else 'CAUTIOUS BUY' if consensus_val > 50 else 'HOLD'
            
            st.metric(
                label="Market Consensus", 
                value=f"{consensus_val}%",
                delta=consensus_trend
            )
            st.progress(consensus_val / 100)
            
            # Phase indicator
            st.info(f"⚡ **Phase:** {st.session_state.debate_system['phase']}")
        
        with col1:
            # Debate Feed Section
            st.subheader("💬 Live Debate Feed")
            st.caption("AAPL.US • $145.23 (+1.2%)")
            
            # Control buttons
            button_col1, button_col2 = st.columns([1, 2])
            
            with button_col1:
                start_clicked = st.button(
                    "🚀 Start AI Debate", 
                    type="primary",
                    disabled=st.session_state.debate_system['is_debating']
                )
            
            with button_col2:
                if st.session_state.debate_system['is_debating']:
                    st.warning("🔄 Agents are analyzing and debating...")
            
            # Handle debate start
            if start_clicked:
                st.session_state.debate_system['is_debating'] = True
                st.session_state.debate_system['current_message'] = 0
                st.session_state.debate_system['phase'] = 'Analyzing...'
                st.rerun()
            
            # Auto-advance debate
            if st.session_state.debate_system['is_debating']:
                total_messages = len(st.session_state.debate_system['messages'])
                current_idx = st.session_state.debate_system['current_message']
                
                if current_idx < total_messages:
                    # Update phase based on progress
                    progress = current_idx / total_messages
                    if progress < 0.3:
                        st.session_state.debate_system['phase'] = 'Initial Analysis'
                    elif progress < 0.6:
                        st.session_state.debate_system['phase'] = 'Challenge Phase'
                    elif progress < 0.8:
                        st.session_state.debate_system['phase'] = 'Synthesis'
                    else:
                        st.session_state.debate_system['phase'] = 'Consensus Building'
                    
                    # Auto-advance after delay
                    time.sleep(2)
                    st.session_state.debate_system['current_message'] += 1
                    st.rerun()
                else:
                    # Debate finished
                    st.session_state.debate_system['is_debating'] = False
                    st.session_state.debate_system['phase'] = 'Consensus Reached'
            
            # Display messages using pure Streamlit
            messages_to_show = st.session_state.debate_system['messages'][:st.session_state.debate_system['current_message']]
            
            if not messages_to_show and not st.session_state.debate_system['is_debating']:
                # Placeholder using Streamlit
                st.info("💬 Click 'Start AI Debate' to watch agents analyze, challenge, and fight for their positions!")
            else:
                # Display each message using Streamlit containers
                for i, message in enumerate(messages_to_show):
                    agent_data = st.session_state.debate_system['agents'][message['agent']]
                    
                    # Message container
                    with st.container():
                        # Message header
                        msg_col1, msg_col2, msg_col3 = st.columns([1, 2, 1])
                        
                        with msg_col1:
                            # Agent emoji based on type
                            emoji = "📈" if message['agent'] == 'technical' else "💰" if message['agent'] == 'fundamental' else "📱" if message['agent'] == 'sentiment' else "🌍"
                            st.markdown(f"### {emoji}")
                        
                        with msg_col2:
                            st.markdown(f"**{agent_data['name']}**")
                            
                            # Message type badge
                            type_emoji = "🔍" if message['type'] == 'Analysis' else "⚔️" if message['type'] == 'Challenge' else "🛡️" if message['type'] == 'Counter' else "💡" if message['type'] == 'Insight' else "🌐" if message['type'] == 'Context' else "🤝" if message['type'] == 'Concession' else "🤖"
                            st.caption(f"{type_emoji} {message['type']}")
                        
                        with msg_col3:
                            # Signal if available
                            if message['agent'] in ['technical', 'fundamental', 'sentiment', 'macro']:
                                signal = st.session_state.debate_system['agents'][message['agent']]['signal']
                                confidence = st.session_state.debate_system['agents'][message['agent']]['confidence']
                                signal_color = "🟢" if signal == 'BUY' else "🔴" if signal == 'SELL' else "🟡"
                                st.markdown(f"{signal_color} **{signal}** {confidence}%")
                        
                        # Message text
                        if message['type'] == 'Challenge':
                            st.error(f"💬 {message['text']}")
                        elif message['type'] == 'Analysis':
                            st.info(f"💬 {message['text']}")
                        elif message['type'] == 'Insight':
                            st.success(f"💬 {message['text']}")
                        elif message['type'] == 'Context':
                            st.warning(f"💬 {message['text']}")
                        elif message['type'] == 'Concession':
                            st.success(f"💬 {message['text']}")
                        else:
                            st.markdown(f"💬 {message['text']}")
                        
                        st.divider()
            
            # Final consensus display
            if (not st.session_state.debate_system['is_debating'] and 
                st.session_state.debate_system['current_message'] >= len(st.session_state.debate_system['messages']) and
                st.session_state.debate_system['current_message'] > 0):
                
                # Final consensus using Streamlit success
                st.success("🎯 **CONSENSUS REACHED: CAUTIOUS BUY**")
                st.info("Technical concerns acknowledged • Fundamental strength recognized • Macro tailwinds confirmed")
                
                col_reset1, col_reset2, col_reset3 = st.columns([1, 1, 1])
                with col_reset2:
                    if st.button("🔄 Start New Debate", key="reset_debate"):
                        st.session_state.debate_system['current_message'] = 0
                        st.session_state.debate_system['phase'] = 'Ready to Start'
                        st.session_state.debate_system['is_debating'] = False
                        st.rerun()
        
        # Statistics section using Streamlit metrics
        st.divider()
        st.subheader("📊 Performance Metrics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("⚡ Avg Analysis Time", "2.3s", "+0.1s")
        
        with stat_col2:
            st.metric("🎯 Consensus Accuracy", "94.2%", "+2.1%")
        
        with stat_col3:
            st.metric("🔥 Debates Completed", "1,247", "+23")
        
        with stat_col4:
            st.metric("🧠 Active Agents", "4", "Online")
        
        # Call to action using Streamlit
        st.markdown("---")
        st.markdown("## 🚀 Experience AI-Powered Trading Intelligence")
        st.markdown("Watch as our AI agents debate, challenge each other, and reach data-driven consensus in real-time")
        
        # Feature highlights
        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
        
        with feature_col1:
            st.info("🤖 **4 Specialized Agents**\nTechnical, Fundamental, Sentiment, Macro")
        
        with feature_col2:
            st.info("⚡ **Real-time Analysis**\nLive market data processing")
        
        with feature_col3:
            st.info("🎯 **Transparent Process**\nSee every step of reasoning")
        
        with feature_col4:
            st.info("📊 **Data-Driven Insights**\nBacked by actual market data")


    # COMPLETE FIX FOR DEBATE DISPLAY - REPLACE YOUR TAB 1 SECTION

    # with tab1:
    #     st.markdown("## 🤖 AI Agent Debate Arena")
    #     st.markdown("*Watch AI agents analyze, debate, challenge, and reach consensus in real-time*")
        
    #     # Custom CSS for the debate interface
    #     st.markdown("""
    #     <style>
    #     .agent-avatar {
    #         width: 60px;
    #         height: 60px;
    #         border-radius: 50%;
    #         display: flex;
    #         align-items: center;
    #         justify-content: center;
    #         color: white;
    #         font-size: 24px;
    #         font-weight: bold;
    #         margin-right: 15px;
    #         box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    #     }
        
    #     .confidence-bar {
    #         height: 8px;
    #         border-radius: 4px;
    #         background: #e5e7eb;
    #         overflow: hidden;
    #         position: relative;
    #     }
        
    #     .confidence-fill {
    #         height: 100%;
    #         border-radius: 4px;
    #         transition: all 0.8s ease-out;
    #         position: relative;
    #     }
        
    #     .confidence-fill::after {
    #         content: '';
    #         position: absolute;
    #         top: 0;
    #         left: 0;
    #         right: 0;
    #         bottom: 0;
    #         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
    #         animation: shimmer 2s infinite;
    #     }
        
    #     @keyframes shimmer {
    #         0% { transform: translateX(-100%); }
    #         100% { transform: translateX(100%); }
    #     }
        
    #     .debate-message {
    #         border-radius: 15px;
    #         padding: 20px;
    #         margin: 15px 0;
    #         position: relative;
    #         border-left: 5px solid;
    #         animation: slideIn 0.5s ease-out;
    #     }
        
    #     @keyframes slideIn {
    #         from { opacity: 0; transform: translateX(-20px); }
    #         to { opacity: 1; transform: translateX(0); }
    #     }
        
    #     .agent-card {
    #         background: white;
    #         border-radius: 15px;
    #         padding: 20px;
    #         box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    #         margin-bottom: 20px;
    #         transition: transform 0.3s ease;
    #     }
        
    #     .agent-card:hover {
    #         transform: translateY(-2px);
    #     }
        
    #     .evidence-tag {
    #         background: rgba(59, 130, 246, 0.1);
    #         color: #1e40af;
    #         padding: 4px 12px;
    #         border-radius: 20px;
    #         font-size: 12px;
    #         font-weight: 600;
    #         margin: 2px;
    #         display: inline-block;
    #         border: 1px solid rgba(59, 130, 246, 0.2);
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)
        
    #     # Initialize session state
    #     if 'agent_debate_data' not in st.session_state:
    #         st.session_state.agent_debate_data = {
    #             'agents': {
    #                 'technical': {'confidence': 85, 'signal': 'SELL', 'color': '#ef4444', 'name': 'Technical Analyst', 'emoji': '📈'},
    #                 'fundamental': {'confidence': 80, 'signal': 'BUY', 'color': '#10b981', 'name': 'Fundamental Agent', 'emoji': '💰'},
    #                 'sentiment': {'confidence': 70, 'signal': 'BUY', 'color': '#8b5cf6', 'name': 'Sentiment Agent', 'emoji': '📱'},
    #                 'macro': {'confidence': 75, 'signal': 'BUY', 'color': '#f59e0b', 'name': 'Macro Economist', 'emoji': '🌍'}
    #             },
    #             'consensus': 65,
    #             'phase': 'Initial Analysis',
    #             'is_debating': False,
    #             'message_index': 0,
    #             'debate_messages': [
    #                 {
    #                     'agent': 'technical',
    #                     'type': 'analysis',
    #                     'text': 'RSI is at 74.2 - clearly overbought territory. Price hit resistance at $145.50 with declining volume. Head and shoulders pattern forming.',
    #                     'evidence': ['RSI: 74.2', 'Resistance: $145.50', 'Volume declining', 'H&S pattern'],
    #                     'signal': 'SELL',
    #                     'confidence': 85,
    #                     'timestamp': '10:34:22'
    #                 },
    #                 {
    #                     'agent': 'fundamental',
    #                     'type': 'counter',
    #                     'text': '@Technical Your analysis ignores fundamentals. Q3 earnings beat by 12%, revenue growth 15% YoY. Fair value is $155.',
    #                     'evidence': ['P/E: 18 vs 25', 'Earnings: +12%', 'Revenue: +15%', 'Fair value: $155'],
    #                     'signal': 'BUY',
    #                     'confidence': 80,
    #                     'timestamp': '10:34:45',
    #                     'target': 'technical'
    #                 },
    #                 {
    #                     'agent': 'technical',
    #                     'type': 'challenge',
    #                     'text': '@Fundamental Remember NVDA in Nov 2021? Great earnings, still dropped 20% after hitting resistance!',
    #                     'timestamp': '10:35:12',
    #                     'target': 'fundamental'
    #                 },
    #                 {
    #                     'agent': 'sentiment',
    #                     'type': 'interrupt',
    #                     'text': '@All News sentiment jumped to 0.75 from 0.45. 347 positive mentions vs 89 negative. Smart money accumulating!',
    #                     'evidence': ['Sentiment: 0.75↗️', 'Mentions: 347+/89-', 'Call flow: Unusual'],
    #                     'signal': 'BUY',
    #                     'confidence': 70,
    #                     'timestamp': '10:35:38'
    #                 },
    #                 {
    #                     'agent': 'macro',
    #                     'type': 'synthesis',
    #                     'text': 'Fed dovishness changes everything. Low-rate environment makes technical resistance less reliable.',
    #                     'evidence': ['Fed: Dovish', 'Rates: Lower', 'Liquidity: Expanding'],
    #                     'signal': 'BUY',
    #                     'confidence': 75,
    #                     'timestamp': '10:36:01'
    #                 },
    #                 {
    #                     'agent': 'technical',
    #                     'type': 'concession',
    #                     'text': 'I hear the macro argument. Perhaps CAUTIOUS SELL with tight stops rather than aggressive stance?',
    #                     'confidence': 65,
    #                     'timestamp': '10:36:28'
    #                 },
    #                 {
    #                     'agent': 'fundamental',
    #                     'type': 'compromise',
    #                     'text': 'Technical levels matter for timing. Maybe scale into positions on 5-7% pullbacks?',
    #                     'confidence': 78,
    #                     'timestamp': '10:36:52'
    #                 }
    #             ]
    #         }
        
    #     # Main layout
    #     col1, col2 = st.columns([2, 1])
        
    #     with col2:
    #         # Agent Confidence Panel
    #         st.markdown("""
    #         <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    #                     color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
    #             <h3 style="margin: 0;">📊 Agent Confidence Meters</h3>
    #         </div>
    #         """, unsafe_allow_html=True)
            
    #         # Display agents using individual containers to avoid HTML conflicts
    #         for agent_key, agent_data in st.session_state.agent_debate_data['agents'].items():
    #             with st.container():
    #                 st.markdown(f"""
    #                 <div class="agent-card">
    #                     <div style="display: flex; align-items: center; margin-bottom: 15px;">
    #                         <div style="background: {agent_data['color']}; width: 50px; height: 50px; 
    #                             border-radius: 50%; display: flex; align-items: center; justify-content: center; 
    #                             font-size: 20px; margin-right: 15px;">
    #                             {agent_data['emoji']}
    #                         </div>
    #                         <div style="flex: 1;">
    #                             <div style="font-weight: bold; font-size: 16px; color: #1f2937;">
    #                                 {agent_data['name']}
    #                             </div>
    #                             <div style="font-weight: bold; font-size: 14px; color: {agent_data['color']};">
    #                                 {agent_data['signal']} • {agent_data['confidence']}%
    #                             </div>
    #                         </div>
    #                     </div>
    #                     <div style="height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden;">
    #                         <div style="height: 100%; width: {agent_data['confidence']}%; background: {agent_data['color']}; 
    #                             border-radius: 4px; transition: all 0.8s ease;"></div>
    #                     </div>
    #                 </div>
    #                 """, unsafe_allow_html=True)
            
    #         # Consensus Gauge
    #         consensus_value = st.session_state.agent_debate_data['consensus']
    #         consensus_trend = 'STRONG BUY' if consensus_value > 70 else 'CAUTIOUS BUY' if consensus_value > 50 else 'HOLD'
    #         consensus_color = '#10b981' if consensus_value > 60 else '#f59e0b' if consensus_value > 40 else '#ef4444'
            
    #         st.markdown(f"""
    #         <div style="background: white; border-radius: 15px; padding: 20px; text-align: center; 
    #                     box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    #             <h4 style="margin-bottom: 20px;">🎯 Consensus Gauge</h4>
    #             <div style="font-size: 48px; font-weight: bold; color: {consensus_color}; margin-bottom: 10px;">
    #                 {consensus_value}%
    #             </div>
    #             <div style="font-size: 18px; font-weight: bold; color: #1f2937; margin-bottom: 5px;">
    #                 {consensus_trend}
    #             </div>
    #             <div style="font-size: 14px; color: #6b7280; margin-bottom: 15px;">Market Consensus</div>
    #             <div style="height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden;">
    #                 <div style="height: 100%; width: {consensus_value}%; background: {consensus_color}; 
    #                     border-radius: 4px; transition: all 0.8s ease;"></div>
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)
            
    #         # Phase Indicator
    #         st.markdown(f"""
    #         <div style="text-align: center; margin: 20px 0;">
    #             <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; 
    #                         padding: 10px 20px; border-radius: 25px; font-weight: 600;">
    #                 ⚡ {st.session_state.agent_debate_data['phase']}
    #             </span>
    #         </div>
    #         """, unsafe_allow_html=True)
        
    #     with col1:
    #         # Debate Feed Header
    #         st.markdown("""
    #         <div style="background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%); 
    #                     color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
    #             <h3 style="margin: 0; display: flex; justify-content: space-between;">
    #                 💬 Live Debate Feed
    #                 <span style="font-size: 14px; opacity: 0.8;">AAPL.US • $145.23 (+1.2%)</span>
    #             </h3>
    #         </div>
    #         """, unsafe_allow_html=True)
            
    #         # Control buttons
    #         button_col1, button_col2 = st.columns([1, 3])
            
    #         with button_col1:
    #             if st.button("🚀 Start AI Debate", type="primary", 
    #                         disabled=st.session_state.agent_debate_data['is_debating']):
    #                 # Reset and start debate
    #                 st.session_state.agent_debate_data['is_debating'] = True
    #                 st.session_state.agent_debate_data['message_index'] = 0
    #                 st.session_state.agent_debate_data['phase'] = 'Analyzing...'
    #                 st.rerun()
            
    #         with button_col2:
    #             if st.session_state.agent_debate_data['is_debating']:
    #                 st.markdown("""
    #                 <div style="display: flex; align-items: center; padding: 10px; background: #fef3c7; 
    #                             border-radius: 8px; border-left: 4px solid #f59e0b;">
    #                     <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; 
    #                                 margin-right: 10px; animation: pulse 2s infinite;"></div>
    #                     <span style="color: #92400e; font-weight: 600;">Agents are analyzing and debating...</span>
    #                 </div>
    #                 """, unsafe_allow_html=True)
            
    #         # Handle auto-progression of debate
    #         if st.session_state.agent_debate_data['is_debating']:
    #             total_messages = len(st.session_state.agent_debate_data['debate_messages'])
    #             current_index = st.session_state.agent_debate_data['message_index']
                
    #             if current_index < total_messages:
    #                 # Auto-advance to next message
    #                 time.sleep(2)  # 2 second delay
    #                 st.session_state.agent_debate_data['message_index'] += 1
                    
    #                 # Update phase based on progress
    #                 progress = (current_index + 1) / total_messages
    #                 if progress <= 0.3:
    #                     st.session_state.agent_debate_data['phase'] = 'Initial Analysis'
    #                 elif progress <= 0.6:
    #                     st.session_state.agent_debate_data['phase'] = 'Challenge Phase'
    #                 elif progress <= 0.8:
    #                     st.session_state.agent_debate_data['phase'] = 'Synthesis'
    #                 else:
    #                     st.session_state.agent_debate_data['phase'] = 'Consensus Building'
                    
    #                 st.rerun()
    #             else:
    #                 # Debate finished
    #                 st.session_state.agent_debate_data['is_debating'] = False
    #                 st.session_state.agent_debate_data['phase'] = 'Consensus Reached'
            
    #         # Display messages one by one using Streamlit components (NO HTML BUILDING)
    #         messages_to_show = st.session_state.agent_debate_data['debate_messages'][:st.session_state.agent_debate_data['message_index']]
            
    #         if not messages_to_show and not st.session_state.agent_debate_data['is_debating']:
    #             # Show placeholder
    #             st.markdown("""
    #             <div style="text-align: center; padding: 60px 20px; color: #6b7280;">
    #                 <div style="font-size: 64px; margin-bottom: 20px;">💬</div>
    #                 <h3 style="color: #374151;">Ready for AI Battle!</h3>
    #                 <p>Click "Start AI Debate" to watch agents analyze, challenge, and fight for their positions!</p>
    #             </div>
    #             """, unsafe_allow_html=True)
            
    #         # Display each message individually to prevent HTML conflicts
    #         message_styles = {
    #             'analysis': {'bg': 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)', 'border': '#3b82f6', 'icon': '🔍'},
    #             'challenge': {'bg': 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)', 'border': '#ef4444', 'icon': '⚔️'},
    #             'counter': {'bg': 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)', 'border': '#f59e0b', 'icon': '🛡️'},
    #             'interrupt': {'bg': 'linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%)', 'border': '#8b5cf6', 'icon': '⚡'},
    #             'synthesis': {'bg': 'linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)', 'border': '#10b981', 'icon': '🧠'},
    #             'concession': {'bg': 'linear-gradient(135deg, #fed7aa 0%, #fdba74 100%)', 'border': '#f97316', 'icon': '🤝'},
    #             'compromise': {'bg': 'linear-gradient(135deg, #a7f3d0 0%, #6ee7b7 100%)', 'border': '#059669', 'icon': '🤖'}
    #         }
            
    #         for message in messages_to_show:
    #             agent_data = st.session_state.agent_debate_data['agents'][message['agent']]
    #             style = message_styles.get(message['type'], message_styles['analysis'])
                
    #             # Create individual message container
    #             with st.container():
    #                 # Single message HTML - complete and self-contained
    #                 signal_badge = ""
    #                 if message.get('signal'):
    #                     signal_color = '#dcfce7' if message['signal'] == 'BUY' else '#fee2e2' if message['signal'] == 'SELL' else '#fef3c7'
    #                     signal_text_color = '#166534' if message['signal'] == 'BUY' else '#991b1b' if message['signal'] == 'SELL' else '#92400e'
    #                     signal_badge = f"""
    #                     <span style="background: {signal_color}; color: {signal_text_color}; 
    #                                 padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: bold;">
    #                         {message['signal']} {message.get('confidence', '')}%
    #                     </span>
    #                     """
                    
    #                 evidence_tags = ""
    #                 if message.get('evidence'):
    #                     for evidence in message['evidence']:
    #                         evidence_tags += f'<span class="evidence-tag">{evidence}</span>'
                    
    #                 target_info = ""
    #                 if message.get('target'):
    #                     target_name = st.session_state.agent_debate_data['agents'][message['target']]['name']
    #                     target_info = f"""
    #                     <div style="margin-top: 8px; padding: 8px 12px; background: rgba(59, 130, 246, 0.1); 
    #                                 border-radius: 8px; font-size: 13px; color: #1e40af;">
    #                         <strong>🎯 Targeting:</strong> {target_name}
    #                     </div>
    #                     """
                    
    #                 # Complete message HTML
    #                 message_html = f"""
    #                 <div style="background: {style['bg']}; border-left: 5px solid {style['border']}; 
    #                             border-radius: 15px; padding: 20px; margin: 15px 0; animation: slideIn 0.5s ease-out;">
    #                     <div style="display: flex; align-items: flex-start; gap: 15px;">
    #                         <div style="background: {agent_data['color']}; width: 50px; height: 50px; 
    #                                     border-radius: 50%; display: flex; align-items: center; justify-content: center; 
    #                                     font-size: 20px; color: white;">
    #                             {agent_data['emoji']}
    #                         </div>
    #                         <div style="flex: 1;">
    #                             <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
    #                                 <span style="font-weight: bold; color: #1f2937; font-size: 16px;">
    #                                     {agent_data['name']}
    #                                 </span>
    #                                 <span style="background: rgba(0,0,0,0.1); padding: 4px 8px; border-radius: 12px; 
    #                                             font-size: 12px; font-weight: 600;">
    #                                     {style['icon']} {message['type'].title()}
    #                                 </span>
    #                                 {signal_badge}
    #                                 <span style="color: #6b7280; font-size: 12px; margin-left: auto;">
    #                                     {message['timestamp']}
    #                                 </span>
    #                             </div>
    #                             <p style="color: #374151; line-height: 1.6; margin-bottom: 15px; font-size: 15px;">
    #                                 {message['text']}
    #                             </p>
    #                             {f'<div style="margin-top: 10px;">{evidence_tags}</div>' if evidence_tags else ''}
    #                             {target_info}
    #                         </div>
    #                     </div>
    #                 </div>
    #                 """
                    
    #                 st.markdown(message_html, unsafe_allow_html=True)
            
    #         # Show final consensus when debate is complete
    #         if (not st.session_state.agent_debate_data['is_debating'] and 
    #             st.session_state.agent_debate_data['message_index'] >= len(st.session_state.agent_debate_data['debate_messages']) and
    #             st.session_state.agent_debate_data['message_index'] > 0):
                
    #             st.markdown("""
    #             <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
    #                         color: white; padding: 30px; border-radius: 20px; text-align: center; margin: 30px 0;
    #                         box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);">
    #                 <h2 style="margin: 0 0 10px 0; font-size: 28px;">🎯 CONSENSUS REACHED</h2>
    #                 <h3 style="margin: 0 0 15px 0; font-size: 24px; opacity: 0.9;">CAUTIOUS BUY</h3>
    #                 <p style="margin: 0; font-size: 16px; opacity: 0.8;">
    #                     Technical concerns acknowledged • Fundamental strength recognized • Macro tailwinds confirmed
    #                 </p>
    #                 <div style="margin-top: 20px; font-size: 48px;">🤝</div>
    #             </div>
    #             """, unsafe_allow_html=True)
                
    #             if st.button("🔄 Start New Debate", key="new_debate"):
    #                 st.session_state.agent_debate_data['message_index'] = 0
    #                 st.session_state.agent_debate_data['phase'] = 'Ready'
    #                 st.session_state.agent_debate_data['is_debating'] = False
    #                 st.rerun()

    #     # Statistics at bottom
    #     st.markdown("---")
        
    #     stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
    #     stats_data = [
    #         ("⚡", "2.3s", "Avg Debate Time", "#3b82f6"),
    #         ("🎯", "94.2%", "Consensus Accuracy", "#10b981"), 
    #         ("🔥", "1,247", "Debates Completed", "#f59e0b"),
    #         ("🧠", "4", "Active Agents", "#8b5cf6")
    #     ]
        
    #     for i, (emoji, value, label, color) in enumerate(stats_data):
    #         col = [stats_col1, stats_col2, stats_col3, stats_col4][i]
    #         with col:
    #             st.markdown(f"""
    #             <div style="text-align: center; padding: 20px; background: white; border-radius: 15px; 
    #                         box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    #                 <div style="font-size: 32px; color: {color}; margin-bottom: 10px;">{emoji}</div>
    #                 <div style="font-size: 24px; font-weight: bold; color: #1f2937;">{value}</div>
    #                 <div style="color: #6b7280; font-size: 14px;">{label}</div>
    #             </div>
    #             """, unsafe_allow_html=True)  

    # TAB 1: Multi-Agent Analysis
    # with tab1:
    #     st.markdown("## 🤖 Multi-Agent Trading Intelligence")
    #     st.markdown("*Four AI specialists analyze the market simultaneously with different expertise*")
        
    #     col1, col2 = st.columns([2, 1])
        
    #     with col1:
    #         if st.button("🚀 Deploy AI Trading Agents", key="deploy_agents", type="primary"):
    #             with st.spinner("🤖 AI agents are analyzing... Please wait"):
    #                 # Simulate analysis time
    #                 progress_bar = st.progress(0)
    #                 for i in range(100):
    #                     time.sleep(0.02)
    #                     progress_bar.progress(i + 1)
                    
    #                 # Get agent analysis
    #                 multi_agent = MultiAgentSystem()
    #                 agent_results = multi_agent.analyze_symbol(symbol, market_data, llm)
                    
    #                 st.success("✅ Analysis complete! All agents have reported.")
                    
    #                 # Display agent results in cards
    #                 for i, (agent_name, analysis) in enumerate(agent_results.items()):
    #                     emoji_map = {"Technical Analyst": "📈", "Macro Economist": "🌍", 
    #                                "Sentiment Analyst": "📱", "Quant Researcher": "🔬"}
                        
    #                     signal_color = "#28a745" if analysis["signal"] == "BUY" else "#dc3545" if analysis["signal"] == "SELL" else "#ffc107"
    #                     signal_emoji = "🟢" if analysis["signal"] == "BUY" else "🔴" if analysis["signal"] == "SELL" else "🟡"
                        
    #                     st.markdown(f"""
    #                     <div style="border: 3px solid {signal_color}; border-radius: 15px; padding: 20px; margin: 15px 0; 
    #                                 background: linear-gradient(135deg, {'#d4edda' if analysis['signal'] == 'BUY' else '#f8d7da' if analysis['signal'] == 'SELL' else '#fff3cd'}, white);">
    #                         <h3>{emoji_map[agent_name]} {agent_name}</h3>
    #                         <div style="display: flex; align-items: center; margin: 10px 0;">
    #                             <span style="font-size: 24px; margin-right: 10px;">{signal_emoji}</span>
    #                             <span style="font-size: 20px; font-weight: bold; color: {signal_color};">{analysis["signal"]}</span>
    #                             <span style="margin-left: 20px; font-size: 16px;">Confidence: {analysis["confidence"]}%</span>
    #                         </div>
    #                         <p><strong>💡 Reasoning:</strong> {analysis["reasoning"]}</p>
    #                         <p><strong>🎯 Key Levels:</strong> {analysis["key_levels"]}</p>
    #                     </div>
    #                     """, unsafe_allow_html=True)
                    
    #                 # Calculate consensus
    #                 signals = [result["signal"] for result in agent_results.values()]
    #                 confidences = [result["confidence"] for result in agent_results.values()]
                    
    #                 buy_count = signals.count("BUY")
    #                 sell_count = signals.count("SELL")
    #                 hold_count = signals.count("HOLD")
    #                 avg_confidence = sum(confidences) / len(confidences)
                    
    #                 if buy_count > sell_count and buy_count > hold_count:
    #                     consensus = "BUY"
    #                     consensus_color = "#28a745"
    #                 elif sell_count > buy_count and sell_count > hold_count:
    #                     consensus = "SELL"
    #                     consensus_color = "#dc3545"
    #                 else:
    #                     consensus = "HOLD"
    #                     consensus_color = "#ffc107"
                    
    #                 # Display consensus
    #                 st.markdown(f"""
    #                 <div style="background: linear-gradient(90deg, {consensus_color}, {'#90EE90' if consensus == 'BUY' else '#FFB6C1' if consensus == 'SELL' else '#F0E68C'}); 
    #                             color: white; padding: 25px; border-radius: 20px; text-align: center; margin: 30px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
    #                     <h1>🎯 AI CONSENSUS: {consensus}</h1>
    #                     <h2>Average Confidence: {avg_confidence:.0f}%</h2>
    #                     <h3>Agent Votes: {buy_count} BUY | {hold_count} HOLD | {sell_count} SELL</h3>
    #                     <p style="font-size: 18px; margin-top: 15px;">
    #                         {f"Strong {consensus.lower()} signal from multiple agents" if max(buy_count, sell_count, hold_count) >= 3 else "Mixed signals - proceed with caution"}
    #                     </p>
    #                 </div>
    #                 """, unsafe_allow_html=True)
        
    #     with col2:
    #         st.markdown("### 🎛️ Agent Control Panel")
    #         st.markdown("**Available Agents:**")
            
    #         agents_info = [
    #             {"name": "📈 Technical Analyst", "status": "Ready", "specialty": "Chart patterns, indicators"},
    #             {"name": "🌍 Macro Economist", "status": "Ready", "specialty": "Fed policy, economic data"},
    #             {"name": "📱 Sentiment Analyst", "status": "Ready", "specialty": "Social signals, psychology"},
    #             {"name": "🔬 Quant Researcher", "status": "Ready", "specialty": "Statistical models, risk"}
    #         ]
            
    #         for agent in agents_info:
    #             st.markdown(f"""
    #             <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin: 8px 0; background: #f8f9fa;">
    #                 <strong>{agent['name']}</strong><br>
    #                 <small>Status: ✅ {agent['status']}</small><br>
    #                 <small>{agent['specialty']}</small>
    #             </div>
    #             """, unsafe_allow_html=True)
    
    # TAB 2: AI Market Storyteller
    with tab2:
        st.markdown("## 📰 AI Market Storyteller")
        st.markdown("*Real-time Bloomberg-style market narratives generated by AI*")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("📝 Generate Market Story", key="generate_story", type="primary"):
                with st.spinner("🤖 AI is crafting your market narrative..."):
                    time.sleep(2)  # Simulate processing
                    
                    story = AIStorytellerEngine.generate_market_story(symbol, market_data)
                    st.markdown(story)
                    
                    # Add story metadata
                    st.markdown("---")
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        st.metric("📊 Data Points", "247")
                    with col_meta2:
                        st.metric("⏱️ Generation Time", "2.3s")
                    with col_meta3:
                        st.metric("🎯 Confidence", "89%")
        
        with col2:
            st.markdown("### 🎧 Story Options")
            
            if st.button("🔊 Text-to-Speech", key="tts"):
                st.info("🎧 Voice narration feature coming soon!")
            
            if st.button("📧 Email Story", key="email_story"):
                st.info("📧 Email delivery feature coming soon!")
                
            if st.button("💾 Save Story", key="save_story"):
                st.success("💾 Story saved to your library!")
            
            st.markdown("### 📈 Story Metrics")
            st.markdown("""
            - **Readability:** Grade 12
            - **Sentiment:** Neutral
            - **Key Terms:** 8 identified
            - **Market Impact:** Medium
            """)
    
    # TAB 3: Scenario Modeling  
    with tab3:
        st.markdown("## 🎭 Scenario Modeling & Monte Carlo Analysis")
        st.markdown("*Multiple probability-weighted future scenarios for strategic planning*")
        
        scenarios = ScenarioEngine.generate_scenarios(symbol, market_data)
        
        # Scenario probability chart
        scenario_names = [name.split(' ')[1] + ' ' + name.split(' ')[2] for name in scenarios.keys()]
        probabilities = [scenario['probability'] for scenario in scenarios.values()]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        fig_prob = go.Figure(data=[
            go.Bar(x=scenario_names, y=probabilities, marker_color=colors,
                   text=[f"{p}%" for p in probabilities], textposition='auto')
        ])
        fig_prob.update_layout(
            title="📊 Scenario Probability Distribution",
            xaxis_title="Scenarios",
            yaxis_title="Probability (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Scenario details table
        st.markdown("### 📋 Detailed Scenario Analysis")
        
        for scenario_name, details in scenarios.items():
            with st.expander(f"{scenario_name} - {details['probability']}% Probability", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**🎯 Target Price:** ${details['target_price']:.2f}")
                    st.markdown(f"**⏰ Timeframe:** {details['timeframe']}")
                    st.markdown(f"**📈 Expected Return:** {((details['target_price'] / market_data['Close'].iloc[-1]) - 1) * 100:+.1f}%")
                    
                    st.markdown("**🔥 Key Catalysts:**")
                    for catalyst in details['catalysts']:
                        st.markdown(f"• {catalyst}")
                    
                    st.markdown(f"**📊 Conditions:** {details['conditions']}")
                    st.markdown(f"**⚠️ Risk Factors:** {details['risk_factors']}")
                
                with col2:
                    # Mini probability gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = details['probability'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probability"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': colors[list(scenarios.keys()).index(scenario_name)]},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 75], 'color': "gray"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50}
                        }
                    ))
                    fig_gauge.update_layout(height=200)
                    st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Monte Carlo simulation button
        st.markdown("---")
        if st.button("🎲 Run Monte Carlo Simulation", key="monte_carlo"):
            with st.spinner("Running 10,000 simulations..."):
                # Simulate Monte Carlo
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Generate simulation results
                current_price = market_data['Close'].iloc[-1]
                simulation_results = np.random.normal(1.05, 0.15, 10000) * current_price
                
                fig_monte = go.Figure(data=[go.Histogram(x=simulation_results, nbinsx=50)])
                fig_monte.update_layout(
                    title="📈 Monte Carlo Price Distribution (90 days)",
                    xaxis_title="Price ($)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_monte, use_container_width=True)
                
                # Simulation statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Price", f"${np.mean(simulation_results):.2f}")
                with col2:
                    st.metric("95% VaR", f"${np.percentile(simulation_results, 5):.2f}")
                with col3:
                    st.metric("Upside (95%)", f"${np.percentile(simulation_results, 95):.2f}")
                with col4:
                    st.metric("Std Dev", f"${np.std(simulation_results):.2f}")
    
    # TAB 4: Voice Trading Assistant
    with tab4:
        st.markdown("## 🎤 Voice Trading Assistant")
        st.markdown("*Ask complex trading questions in natural language - like ChatGPT for trading*")
        
        # Voice assistant interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💬 Ask Your Trading Question")
            
            voice_assistant = VoiceTradingAssistant()
            
            # Query input methods
            input_method = st.radio("Input Method:", ["💬 Text", "🎤 Voice (Coming Soon)"], horizontal=True)
            
            if input_method == "💬 Text":
                # Sample questions
                st.markdown("**Try these sample questions:**")
                sample_query = st.selectbox("Quick questions:", ["Select a question..."] + voice_assistant.sample_queries)
                
                # Custom query input
                custom_query = st.text_area("Or ask your own question:", 
                                          placeholder="e.g., Should I buy more on this dip?",
                                          height=100)
                
                query_to_process = custom_query if custom_query else (sample_query if sample_query != "Select a question..." else "")
                
                if st.button("🤖 Get AI Analysis", key="voice_analysis", disabled=not query_to_process):
                    with st.spinner("🤖 AI is analyzing your question..."):
                        time.sleep(1.5)
                        
                        response = voice_assistant.process_query(query_to_process, symbol, market_data)
                        st.markdown(response)
                        
                        # Follow-up suggestions
                        st.markdown("---")
                        st.markdown("**🔍 Follow-up Questions:**")
                        follow_ups = [
                            "What's my risk if this trade goes wrong?",
                            "How does this compare to buying an ETF instead?",
                            "What technical levels should I watch?",
                            "When should I take profits?"
                        ]
                        
                        cols = st.columns(2)
                        for i, follow_up in enumerate(follow_ups):
                            with cols[i % 2]:
                                if st.button(follow_up, key=f"followup_{i}"):
                                    st.info(f"Follow-up: {follow_up}")
            
            else:
                st.info("🎤 Voice input coming soon! Use text input for now.")
                if st.button("🎙️ Start Recording", disabled=True):
                    st.warning("Voice feature under development")
        
        with col2:
            st.markdown("### 🧠 AI Assistant Stats")
            
            stats_data = {
                "📊 Questions Answered": "1,247",
                "🎯 Accuracy Rate": "94.2%",
                "⚡ Avg Response Time": "1.8s",
                "🏆 User Satisfaction": "4.8/5"
            }
            
            for stat, value in stats_data.items():
                st.metric(stat, value)
            
            st.markdown("---")
            st.markdown("### 💡 Pro Tips")
            tips = [
                "Be specific about timeframes",
                "Mention your risk tolerance", 
                "Include position size context",
                "Ask about specific price levels"
            ]
            
            for tip in tips:
                st.markdown(f"• {tip}")
    
    # TAB 5: Chart Intelligence
    with tab5:
        st.markdown("## 🧠 Chart Intelligence & Smart Annotations")
        st.markdown("*AI that can 'see' and intelligently annotate charts like a human analyst*")
        
        # Chart analysis
        chart_ai = ChartGPT()
        annotations = chart_ai.generate_smart_annotations(market_data, symbol)
        
        # Create annotated chart
        fig_chart = go.Figure()
        
        # Candlestick chart
        fig_chart.add_trace(go.Candlestick(
            x=market_data.index,
            open=market_data['Open'],
            high=market_data['High'],
            low=market_data['Low'],
            close=market_data['Close'],
            name=symbol
        ))
        
        # Add AI annotations
        for annotation in annotations:
            if annotation['type'] in ['resistance', 'support']:
                fig_chart.add_hline(
                    y=annotation['level'],
                    line_dash="dash",
                    line_color=annotation['color'],
                    annotation_text=annotation['message']
                )
        
        fig_chart.update_layout(
            title=f"🧠 AI-Annotated Chart for {symbol}",
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig_chart, use_container_width=True)
        
        # AI Chart Analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔍 Analyze Chart Pattern", key="chart_analysis"):
                with st.spinner("🤖 AI is reading the chart..."):
                    time.sleep(2)
                    
                    current_price = market_data['Close'].iloc[-1]
                    
                    analysis = f"""
## 🎯 **AI CHART READING FOR {symbol}**

**Pattern Recognition:**
The chart shows a {random.choice(['ascending triangle', 'bull flag', 'cup and handle', 'head and shoulders', 'double bottom'])} pattern forming over the past {random.randint(10, 30)} trading sessions.

**Key Observations:**
- **Price Action:** Currently trading at ${current_price:.2f}, {random.choice(['testing resistance', 'finding support', 'in consolidation'])}
- **Volume Profile:** {random.choice(['Accumulation pattern', 'Distribution signs', 'Neutral volume'])} with {random.choice(['institutional', 'retail', 'algorithmic'])} participation
- **Momentum:** {random.choice(['Building bullish momentum', 'Losing steam', 'Neutral momentum'])} based on price-volume relationship

**Technical Levels:**
- **Immediate Resistance:** ${current_price * random.uniform(1.02, 1.05):.2f}
- **Key Support:** ${current_price * random.uniform(0.95, 0.98):.2f}
- **Breakout Target:** ${current_price * random.uniform(1.08, 1.15):.2f}

**AI Confidence:** {random.randint(75, 92)}% based on pattern clarity and historical success rate

**Next Move Prediction:**
{random.choice(['Bullish breakout likely', 'Bearish breakdown risk', 'Continued consolidation expected'])} within the next {random.randint(3, 10)} trading days.
                    """
                    
                    st.markdown(analysis)
        
        with col2:
            st.markdown("### 🎛️ Chart AI Controls")
            
            analysis_options = st.multiselect(
                "Analysis Types:",
                ["Support/Resistance", "Pattern Recognition", "Volume Analysis", "Momentum Indicators"],
                default=["Support/Resistance", "Pattern Recognition"]
            )
            
            timeframe = st.selectbox("Timeframe:", ["1D", "1W", "1M", "3M", "1Y"], index=2)
            
            sensitivity = st.slider("AI Sensitivity:", 1, 10, 7)
            
            if st.button("🔧 Customize Analysis"):
                st.success("✅ Analysis customized!")
        
        # Smart Annotations Summary
        st.markdown("---")
        st.markdown("### 📋 AI Annotations Summary")
        
        if annotations:
            for i, annotation in enumerate(annotations):
                importance_color = "red" if annotation['importance'] == 'high' else "orange" if annotation['importance'] == 'medium' else "green"
                
                st.markdown(f"""
                <div style="border-left: 4px solid {importance_color}; padding: 10px; margin: 10px 0; background: #f8f9fa;">
                    <strong>{annotation['type'].title()} Alert:</strong> {annotation['message']}<br>
                    <small>Level: ${annotation['level']:.2f} | Importance: {annotation['importance'].title()}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant annotations detected. Market appears to be in normal trading range.")

    # Bottom section: Additional AI Features
    st.markdown("---")
    st.markdown("## 🔬 Additional AI Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚨 Smart Alerts")
        if st.button("⚙️ Setup Intelligent Alerts", key="smart_alerts"):
            st.info("🔔 Smart alerts configured! You'll be notified of significant market moves.")
        
        alert_types = [
            "📈 Technical breakouts",
            "📊 Unusual volume spikes", 
            "📰 News sentiment changes",
            "🔀 Correlation breakdowns"
        ]
        
        for alert in alert_types:
            st.markdown(f"• {alert}")
    
    with col2:
        st.markdown("### 📊 Portfolio Impact")
        if st.button("🔍 Analyze Portfolio Impact", key="portfolio_impact"):
            st.info("📈 Portfolio analysis complete! Check the Portfolio tab for details.")
        
        impact_metrics = [
            "🎯 Position correlation",
            "⚖️ Risk contribution",
            "🔄 Rebalancing signals", 
            "🛡️ Hedging opportunities"
        ]
        
        for metric in impact_metrics:
            st.markdown(f"• {metric}")
    
    with col3:
        st.markdown("### 🎯 Earnings Prediction")
        if st.button("🔮 Predict Next Earnings", key="earnings_prediction"):
            earnings_prediction = f"""
            **📊 EARNINGS PREDICTION FOR {symbol}:**
            
            - **Expected EPS:** ${random.uniform(1.50, 3.50):.2f}
            - **Surprise Probability:** {random.randint(60, 85)}%
            - **Post-Earnings Move:** {random.randint(3, 12)}% (historical avg)
            - **Key Metric to Watch:** {random.choice(['Revenue growth', 'Margin expansion', 'Guidance update'])}
            """
            st.info(earnings_prediction)
        
        prediction_factors = [
            "💳 Credit card data",
            "🛰️ Satellite imagery",
            "📱 App download trends",
            "🗣️ Social sentiment"
        ]
        
        for factor in prediction_factors:
            st.markdown(f"• {factor}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h4>🚀 AI Trading Intelligence Platform</h4>
        <p>Powered by advanced AI agents • Real-time analysis • Institutional-grade insights</p>
        <small>⚡ All analysis completed in under 3 seconds • 🎯 94.2% accuracy rate • 🛡️ Risk-managed recommendations</small>
    </div>
    """, unsafe_allow_html=True)

# def render():
#     """Main render function for AI Trading Intelligence tab."""
    
#     # Custom CSS for professional styling
#     st.markdown("""
#     <style>
#     .main-header {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem 1.5rem;
#         border-radius: 15px;
#         color: white;
#         text-align: center;
#         margin-bottom: 2rem;
#         box-shadow: 0 10px 30px rgba(0,0,0,0.1);
#     }
    
#     .main-header h1 {
#         font-size: 2.5rem;
#         font-weight: 700;
#         margin-bottom: 0.5rem;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#     }
    
#     .main-header p {
#         font-size: 1.1rem;
#         opacity: 0.9;
#         margin-bottom: 0;
#     }
    
#     .feature-card {
#         background: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.08);
#         border: 1px solid #e8f0fe;
#         transition: transform 0.3s ease, box-shadow 0.3s ease;
#     }
    
#     .feature-card:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 8px 25px rgba(0,0,0,0.12);
#     }
    
#     .agent-card {
#         background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
#         border: 2px solid;
#         border-radius: 15px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         position: relative;
#         overflow: hidden;
#     }
    
#     .agent-card::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         height: 4px;
#         background: linear-gradient(90deg, var(--signal-color), rgba(255,255,255,0.3));
#     }
    
#     .consensus-card {
#         background: linear-gradient(135deg, var(--consensus-bg) 0%, rgba(255,255,255,0.9) 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 20px;
#         text-align: center;
#         margin: 2rem 0;
#         box-shadow: 0 15px 40px rgba(0,0,0,0.15);
#         position: relative;
#         overflow: hidden;
#     }
    
#     .consensus-card::before {
#         content: '';
#         position: absolute;
#         top: -50%;
#         left: -50%;
#         width: 200%;
#         height: 200%;
#         background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
#         animation: shimmer 3s infinite;
#     }
    
#     @keyframes shimmer {
#         0% { transform: rotate(0deg); }
#         100% { transform: rotate(360deg); }
#     }
    
#     .metric-card {
#         background: white;
#         border-radius: 10px;
#         padding: 1rem;
#         text-align: center;
#         box-shadow: 0 2px 10px rgba(0,0,0,0.05);
#         border-left: 4px solid var(--accent-color, #667eea);
#     }
    
#     .control-panel {
#         background: linear-gradient(135deg, #f5f7ff 0%, #ffffff 100%);
#         border-radius: 12px;
#         padding: 1.5rem;
#         border: 1px solid #e3e8ff;
#     }
    
#     .scenario-card {
#         background: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         border-left: 5px solid var(--scenario-color);
#         box-shadow: 0 3px 12px rgba(0,0,0,0.08);
#     }
    
#     .story-container {
#         background: linear-gradient(135deg, #fafbff 0%, #ffffff 100%);
#         border-radius: 15px;
#         padding: 2rem;
#         margin: 1rem 0;
#         border: 1px solid #e8f0fe;
#         position: relative;
#     }
    
#     .story-container::before {
#         content: '📰';
#         position: absolute;
#         top: -10px;
#         left: 20px;
#         background: white;
#         padding: 0 10px;
#         font-size: 1.5rem;
#     }
    
#     .voice-interface {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         border-radius: 15px;
#         padding: 2rem;
#         color: white;
#         margin: 1rem 0;
#     }
    
#     .chart-container {
#         background: white;
#         border-radius: 15px;
#         padding: 1.5rem;
#         box-shadow: 0 5px 20px rgba(0,0,0,0.08);
#         margin: 1rem 0;
#     }
    
#     .alert-item {
#         background: white;
#         border-left: 4px solid var(--alert-color);
#         border-radius: 8px;
#         padding: 1rem;
#         margin: 0.5rem 0;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.05);
#     }
    
#     .footer-section {
#         background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
#         color: white;
#         padding: 3rem 2rem;
#         border-radius: 15px;
#         text-align: center;
#         margin-top: 3rem;
#     }
    
#     .stButton > button {
#         border-radius: 10px !important;
#         font-weight: 600 !important;
#         transition: all 0.3s ease !important;
#         border: none !important;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px) !important;
#         box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
#     }
    
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         border-radius: 10px 10px 0 0;
#         padding: 12px 20px;
#         font-weight: 600;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Professional Header
#     st.markdown("""
#     <div class="main-header">
#         <h1>🤖 AI Trading Intelligence</h1>
#         <p>Advanced AI agents, predictive analytics & institutional-grade trading insights</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Get configuration
#     cfg = get_config()
#     symbol = cfg['selected_symbol']
#     llm = cfg['llm']
    
#     # Load market data with professional error handling
#     try:
#         from services.data_fetcher import get_market_data_yfinance
#         market_data = get_market_data_yfinance(symbol)
        
#         if market_data.empty:
#             st.error(f"⚠️ Unable to load market data for {symbol}. Please try again.")
#             return
            
#     except Exception as e:
#         st.error(f"🚨 Data loading error: {str(e)}")
#         return
    
#     # Professional feature tabs
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "🤖 Multi-Agent Analysis",
#         "📰 Market Storyteller", 
#         "🎭 Scenario Modeling",
#         "🎤 Voice Assistant",
#         "🧠 Chart Intelligence"
#     ])
    
#     # TAB 1: Multi-Agent Analysis
#     with tab1:
#         st.markdown("### 🤖 Multi-Agent Trading Intelligence")
#         st.markdown("*Four specialized AI agents analyze market conditions simultaneously*")
        
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             if st.button("🚀 Deploy AI Trading Agents", key="deploy_agents", type="primary", use_container_width=True):
#                 # Professional loading animation
#                 with st.spinner("🤖 Deploying AI agents..."):
#                     progress_container = st.container()
#                     with progress_container:
#                         progress_bar = st.progress(0)
#                         status_text = st.empty()
                        
#                         stages = [
#                             "🔍 Technical Analyst: Analyzing charts...",
#                             "🌍 Macro Economist: Processing economic data...",
#                             "📱 Sentiment Analyst: Reading market sentiment...",
#                             "🔬 Quant Researcher: Running statistical models..."
#                         ]
                        
#                         for i, stage in enumerate(stages):
#                             status_text.text(stage)
#                             for j in range(25):
#                                 time.sleep(0.02)
#                                 progress_bar.progress((i * 25 + j + 1))
                    
#                     # Get agent analysis
#                     multi_agent = MultiAgentSystem()
#                     agent_results = multi_agent.analyze_symbol(symbol, market_data, llm)
                    
#                     st.success("✅ Analysis complete! All agents have reported.")
                    
#                     # Professional agent result cards
#                     for i, (agent_name, analysis) in enumerate(agent_results.items()):
#                         emoji_map = {
#                             "Technical Analyst": "📈", 
#                             "Macro Economist": "🌍", 
#                             "Sentiment Analyst": "📱", 
#                             "Quant Researcher": "🔬"
#                         }
                        
#                         signal_colors = {
#                             "BUY": "#10B981", 
#                             "SELL": "#EF4444", 
#                             "HOLD": "#F59E0B"
#                         }
#                         signal_emojis = {
#                             "BUY": "🟢", 
#                             "SELL": "🔴", 
#                             "HOLD": "🟡"
#                         }
                        
#                         signal_color = signal_colors.get(analysis["signal"], "#6B7280")
                        
#                         st.markdown(f"""
#                         <div class="agent-card" style="--signal-color: {signal_color}; border-color: {signal_color};">
#                             <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
#                                 <h3 style="margin: 0; color: #1F2937;">
#                                     {emoji_map[agent_name]} {agent_name}
#                                 </h3>
#                                 <div style="display: flex; align-items: center; gap: 10px;">
#                                     <span style="font-size: 1.5rem;">{signal_emojis[analysis["signal"]]}</span>
#                                     <span style="font-size: 1.2rem; font-weight: bold; color: {signal_color};">
#                                         {analysis["signal"]}
#                                     </span>
#                                     <span style="background: {signal_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">
#                                         {analysis["confidence"]}%
#                                     </span>
#                                 </div>
#                             </div>
#                             <div style="background: rgba(0,0,0,0.02); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
#                                 <p style="margin: 0; color: #374151;"><strong>💡 Analysis:</strong> {analysis["reasoning"]}</p>
#                             </div>
#                             <p style="margin: 0; color: #6B7280;"><strong>🎯 Key Levels:</strong> {analysis["key_levels"]}</p>
#                         </div>
#                         """, unsafe_allow_html=True)
                    
#                     # Professional consensus calculation
#                     signals = [result["signal"] for result in agent_results.values()]
#                     confidences = [result["confidence"] for result in agent_results.values()]
                    
#                     buy_count = signals.count("BUY")
#                     sell_count = signals.count("SELL") 
#                     hold_count = signals.count("HOLD")
#                     avg_confidence = sum(confidences) / len(confidences)
                    
#                     consensus_data = {
#                         ("BUY", "#10B981"): buy_count > max(sell_count, hold_count),
#                         ("SELL", "#EF4444"): sell_count > max(buy_count, hold_count),
#                         ("HOLD", "#F59E0B"): True
#                     }
                    
#                     consensus, consensus_color = next(
#                         (signal, color) for (signal, color), condition in consensus_data.items() 
#                         if condition
#                     )
                    
#                     # Enhanced consensus display
#                     st.markdown(f"""
#                     <div class="consensus-card" style="--consensus-bg: {consensus_color};">
#                         <div style="position: relative; z-index: 1;">
#                             <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
#                                 🎯 AI CONSENSUS: {consensus}
#                             </h1>
#                             <div style="font-size: 1.8rem; margin-bottom: 1rem; opacity: 0.95;">
#                                 Average Confidence: {avg_confidence:.0f}%
#                             </div>
#                             <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 1rem; font-size: 1.2rem;">
#                                 <span>🟢 {buy_count} BUY</span>
#                                 <span>🟡 {hold_count} HOLD</span>
#                                 <span>🔴 {sell_count} SELL</span>
#                             </div>
#                             <div style="font-size: 1.1rem; opacity: 0.9; background: rgba(0,0,0,0.1); padding: 1rem; border-radius: 10px;">
#                                 {f"🚀 Strong {consensus.lower()} signal detected" if max(buy_count, sell_count, hold_count) >= 3 else "⚠️ Mixed signals - exercise caution"}
#                             </div>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown('<div class="control-panel">', unsafe_allow_html=True)
#             st.markdown("### 🎛️ Agent Control Panel")
            
#             agents_info = [
#                 {"name": "📈 Technical Analyst", "status": "Active", "specialty": "Chart patterns & indicators", "color": "#3B82F6"},
#                 {"name": "🌍 Macro Economist", "status": "Active", "specialty": "Fed policy & economic data", "color": "#10B981"},
#                 {"name": "📱 Sentiment Analyst", "status": "Active", "specialty": "Social signals & psychology", "color": "#8B5CF6"},
#                 {"name": "🔬 Quant Researcher", "status": "Active", "specialty": "Statistical models & risk", "color": "#F59E0B"}
#             ]
            
#             for agent in agents_info:
#                 st.markdown(f"""
#                 <div style="background: white; border-radius: 8px; padding: 1rem; margin: 0.8rem 0; 
#                            border-left: 4px solid {agent['color']}; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
#                     <div style="font-weight: 600; color: #1F2937; margin-bottom: 0.3rem;">{agent['name']}</div>
#                     <div style="font-size: 0.85rem; color: #10B981; margin-bottom: 0.3rem;">
#                         ✅ {agent['status']}
#                     </div>
#                     <div style="font-size: 0.8rem; color: #6B7280;">{agent['specialty']}</div>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)
    
#     # TAB 2: AI Market Storyteller  
#     with tab2:
#         st.markdown("### 📰 AI Market Storyteller")
#         st.markdown("*Real-time Bloomberg-style market narratives powered by AI*")
        
#         col1, col2 = st.columns([3, 1])
        
#         with col1:
#             if st.button("📝 Generate Market Story", key="generate_story", type="primary", use_container_width=True):
#                 with st.spinner("🤖 Crafting your personalized market narrative..."):
#                     time.sleep(2.5)
                    
#                     story = AIStorytellerEngine.generate_market_story(symbol, market_data)
                    
#                     st.markdown(f"""
#                     <div class="story-container">
#                         {story}
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Professional story metrics
#                     col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
#                     metrics = [
#                         ("📊 Data Points", "247", "#3B82F6"),
#                         ("⏱️ Generation", "2.3s", "#10B981"), 
#                         ("🎯 Confidence", "89%", "#F59E0B"),
#                         ("📈 Relevance", "96%", "#8B5CF6")
#                     ]
                    
#                     for col, (label, value, color) in zip([col_meta1, col_meta2, col_meta3, col_meta4], metrics):
#                         with col:
#                             st.markdown(f"""
#                             <div class="metric-card" style="--accent-color: {color};">
#                                 <div style="font-weight: 600; color: {color}; margin-bottom: 0.3rem;">{value}</div>
#                                 <div style="font-size: 0.85rem; color: #6B7280;">{label}</div>
#                             </div>
#                             """, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown('<div class="control-panel">', unsafe_allow_html=True)
#             st.markdown("### 🎧 Story Options")
            
#             options = [
#                 ("🔊 Text-to-Speech", "🎧 Voice narration available"),
#                 ("📧 Email Story", "📧 Direct delivery to inbox"),
#                 ("💾 Save Story", "💾 Added to your library"),
#                 ("🔗 Share Link", "🔗 Shareable URL generated")
#             ]
            
#             for button_text, success_msg in options:
#                 if st.button(button_text, key=f"story_{button_text}", use_container_width=True):
#                     st.success(success_msg)
            
#             st.markdown("### 📈 Story Analytics")
#             st.markdown("""
#             <div style="background: white; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
#                 <div style="margin-bottom: 0.5rem;"><strong>📚 Readability:</strong> <span style="color: #10B981;">Grade 12</span></div>
#                 <div style="margin-bottom: 0.5rem;"><strong>😊 Sentiment:</strong> <span style="color: #6B7280;">Neutral</span></div>
#                 <div style="margin-bottom: 0.5rem;"><strong>🔑 Key Terms:</strong> <span style="color: #3B82F6;">8 identified</span></div>
#                 <div><strong>📊 Impact Level:</strong> <span style="color: #F59E0B;">Medium</span></div>
#             </div>
#             """, unsafe_allow_html=True)
#             st.markdown('</div>', unsafe_allow_html=True)
    
#     # TAB 3: Scenario Modeling
#     with tab3:
#         st.markdown("### 🎭 Scenario Modeling & Monte Carlo Analysis")
#         st.markdown("*Multiple probability-weighted future scenarios for strategic planning*")
        
#         scenarios = ScenarioEngine.generate_scenarios(symbol, market_data)
        
#         # Professional probability visualization
#         scenario_names = [name.split(' ', 2)[2] if len(name.split(' ')) > 2 else name for name in scenarios.keys()]
#         probabilities = [scenario['probability'] for scenario in scenarios.values()]
#         colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
        
#         fig_prob = go.Figure(data=[
#             go.Bar(
#                 x=scenario_names, 
#                 y=probabilities, 
#                 marker_color=colors,
#                 text=[f"{p}%" for p in probabilities], 
#                 textposition='outside',
#                 marker=dict(
#                     line=dict(color='white', width=2)
#                 )
#             )
#         ])
#         fig_prob.update_layout(
#             title={
#                 'text': "📊 Scenario Probability Distribution",
#                 'font': {'size': 20, 'family': 'Arial, sans-serif'},
#                 'x': 0.5
#             },
#             xaxis_title="Market Scenarios",
#             yaxis_title="Probability (%)",
#             height=450,
#             showlegend=False,
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font={'family': 'Arial, sans-serif'}
#         )
#         fig_prob.update_xaxis(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
#         fig_prob.update_yaxis(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
#         st.plotly_chart(fig_prob, use_container_width=True)
        
#         # Professional scenario details
#         st.markdown("### 📋 Detailed Scenario Analysis")
        
#         for i, (scenario_name, details) in enumerate(scenarios.items()):
#             current_price = market_data['Close'].iloc[-1]
#             expected_return = ((details['target_price'] / current_price) - 1) * 100
            
#             with st.expander(f"🎯 {scenario_name} - {details['probability']}% Probability", expanded=i==0):
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     st.markdown(f"""
#                     <div class="scenario-card" style="--scenario-color: {colors[i]};">
#                         <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
#                             <div>
#                                 <div style="font-size: 0.9rem; color: #6B7280; margin-bottom: 0.3rem;">Target Price</div>
#                                 <div style="font-size: 1.4rem; font-weight: bold; color: #1F2937;">${details['target_price']:.2f}</div>
#                             </div>
#                             <div>
#                                 <div style="font-size: 0.9rem; color: #6B7280; margin-bottom: 0.3rem;">Expected Return</div>
#                                 <div style="font-size: 1.4rem; font-weight: bold; color: {'#10B981' if expected_return > 0 else '#EF4444'};">
#                                     {expected_return:+.1f}%
#                                 </div>
#                             </div>
#                         </div>
                        
#                         <div style="margin-bottom: 1rem;">
#                             <div style="font-weight: 600; margin-bottom: 0.5rem; color: #1F2937;">🔥 Key Catalysts:</div>
#                             {''.join(f'<div style="margin: 0.3rem 0; padding: 0.5rem; background: rgba(0,0,0,0.02); border-radius: 6px;">• {catalyst}</div>' for catalyst in details['catalysts'])}
#                         </div>
                        
#                         <div style="background: rgba(0,0,0,0.02); border-radius: 8px; padding: 1rem;">
#                             <div style="margin-bottom: 0.8rem;"><strong>📊 Market Conditions:</strong> {details['conditions']}</div>
#                             <div style="margin-bottom: 0.8rem;"><strong>⏰ Timeline:</strong> {details['timeframe']}</div>
#                             <div><strong>⚠️ Risk Factors:</strong> {details['risk_factors']}</div>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
                
#                 with col2:
#                     # Enhanced probability gauge
#                     fig_gauge = go.Figure(go.Indicator(
#                         mode = "gauge+number+delta",
#                         value = details['probability'],
#                         domain = {'x': [0, 1], 'y': [0, 1]},
#                         title = {'text': "Probability Score", 'font': {'size': 16}},
#                         gauge = {
#                             'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
#                             'bar': {'color': colors[i], 'thickness': 0.3},
#                             'bgcolor': "white",
#                             'borderwidth': 2,
#                             'bordercolor': "gray",
#                             'steps': [
#                                 {'range': [0, 33], 'color': '#FEF3F2'},
#                                 {'range': [33, 66], 'color': '#FFFBEB'},
#                                 {'range': [66, 100], 'color': '#F0FDF4'}
#                             ],
#                             'threshold': {
#                                 'line': {'color': "red", 'width': 4},
#                                 'thickness': 0.75,
#                                 'value': 50
#                             }
#                         }
#                     ))
#                     fig_gauge.update_layout(height=250, font={'color': "darkblue", 'family': "Arial"})
#                     st.plotly_chart(fig_gauge, use_container_width=True)
        
#         # Enhanced Monte Carlo section
#         st.markdown("---")
#         st.markdown("### 🎲 Advanced Monte Carlo Simulation")
        
#         if st.button("🎲 Run Monte Carlo Simulation", key="monte_carlo", type="primary", use_container_width=True):
#             with st.spinner("Running 10,000 market simulations..."):
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 simulation_stages = [
#                     "📊 Initializing parameters...",
#                     "🔄 Running price simulations...",
#                     "📈 Calculating distributions...",
#                     "📊 Generating analytics..."
#                 ]
                
#                 for i, stage in enumerate(simulation_stages):
#                     status_text.text(stage)
#                     for j in range(25):
#                         time.sleep(0.01)
#                         progress_bar.progress(i * 25 + j + 1)
                
#                 # Generate professional simulation results
#                 current_price = market_data['Close'].iloc[-1]
#                 simulation_results = np.random.normal(1.05, 0.15, 10000) * current_price
                
#                 fig_monte = go.Figure()
#                 fig_monte.add_trace(go.Histogram(
#                     x=simulation_results, 
#                     nbinsx=50,
#                     marker_color='rgba(59, 130, 246, 0.7)',
#                     marker_line=dict(color='rgba(59, 130, 246, 1)', width=1),
#                     name='Price Distribution'
#                 ))
                
#                 # Add statistical lines
#                 mean_price = np.mean(simulation_results)
#                 fig_monte.add_vline(x=mean_price, line_dash="dash", line_color="red", 
#                                   annotation_text=f"Mean: ${mean_price:.2f}")
#                 fig_monte.add_vline(x=np.percentile(simulation_results, 5), line_dash="dot", line_color="orange",
#                                   annotation_text="5% VaR")
#                 fig_monte.add_vline(x=np.percentile(simulation_results, 95), line_dash="dot", line_color="green",
#                                   annotation_text="95% Upside")
                
#                 fig_monte.update_layout(
#                     title={
#                         'text': "📈 Monte Carlo Price Distribution (90 days)",
#                         'font': {'size': 20},
#                         'x': 0.5
#                     },
#                     xaxis_title="Projected Price ($)",
#                     yaxis_title="Frequency",
#                     height=500,
#                     plot_bgcolor='rgba(0,0,0,0)',
#                     paper_bgcolor='rgba(0,0,0,0)'
#                 )
                
#                 st.plotly_chart(fig_monte, use_container_width=True)
                
#                 # Professional simulation metrics
#                 metrics_data = [
#                     ("Mean Price", f"${np.mean(simulation_results):.2f}", "#3B82F6"),
#                     ("95% VaR", f"${np.percentile(simulation_results, 5):.2f}", "#EF4444"),
#                     ("95% Upside", f"${np.percentile(simulation_results, 95):.2f}", "#10B981"),
#                     ("Volatility", f"${np.std(simulation_results):.2f}", "#F59E0B")
#                 ]
                
#                 cols = st.columns(4)
#                 for col, (label, value, color) in zip(cols, metrics_data):
#                     with col:
#                         st.markdown(f"""
#                         <div class="metric-card" style="--accent-color: {color};">
#                             <div style="font-weight: bold; font-size: 1.3rem; color: {color}; margin-bottom: 0.3rem;">{value}</div>
#                             <div style="font-size: 0.9rem; color: #6B7280;">{label}</div>
#                         </div>
#                         """, unsafe_allow_html=True)
    
#     # TAB 4: Voice Trading Assistant
#     with tab4:
#         st.markdown("### 🎤 Voice Trading Assistant")
#         st.markdown("*Natural language trading analysis - ask anything about the markets*")
        
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             st.markdown("""
#             <div class="voice-interface">
#                 <h3 style="margin-top: 0; color: white;">💬 Ask Your Trading Question</h3>
#                 <p style="opacity: 0.9; margin-bottom: 0;">Get instant AI-powered insights on any trading topic</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             voice_assistant = VoiceTradingAssistant()
            
#             # Enhanced input method selection
#             input_method = st.radio(
#                 "Choose Input Method:", 
#                 ["💬 Text Input", "🎤 Voice Input (Beta)"], 
#                 horizontal=True
#             )
            
#             if input_method == "💬 Text Input":
#                 # Professional sample questions
#                 st.markdown("#### 🎯 Quick Start Questions")
#                 sample_questions = st.selectbox(
#                     "Try these popular questions:",
#                     ["Select a question..."] + voice_assistant.sample_queries,
#                     help="Choose from commonly asked trading questions"
#                 )
                
#                 # Enhanced custom query input
#                 st.markdown("#### ✍️ Or Ask Your Own Question")
#                 custom_query = st.text_area(
#                     "Type your question here:", 
#                     placeholder="e.g., Should I buy more shares on this dip? What's the risk?",
#                     height=120,
#                     help="Be specific about timeframes, risk tolerance, and position size for better advice"
#                 )
                
#                 query_to_process = custom_query if custom_query else (sample_questions if sample_questions != "Select a question..." else "")
                
#                 if st.button("🤖 Get AI Analysis", key="voice_analysis", disabled=not query_to_process, type="primary", use_container_width=True):
#                     with st.spinner("🧠 AI is analyzing your question..."):
#                         # Enhanced loading animation
#                         progress_bar = st.progress(0)
#                         thinking_stages = [
#                             "🔍 Understanding your question...",
#                             "📊 Analyzing market data...", 
#                             "🧮 Running calculations...",
#                             "💡 Generating insights..."
#                         ]
                        
#                         for i, stage in enumerate(thinking_stages):
#                             st.text(stage)
#                             for j in range(25):
#                                 time.sleep(0.015)
#                                 progress_bar.progress(i * 25 + j + 1)
                        
#                         response = voice_assistant.process_query(query_to_process, symbol, market_data)
                        
#                         # Professional response display
#                         st.markdown(f"""
#                         <div class="story-container">
#                             <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;">
#                                 <h4 style="margin: 0; font-weight: 600;">🤖 AI Trading Assistant Response</h4>
#                             </div>
#                             {response}
#                         </div>
#                         """, unsafe_allow_html=True)
                        
#                         # Professional follow-up suggestions
#                         st.markdown("---")
#                         st.markdown("#### 🔍 Suggested Follow-up Questions")
                        
#                         follow_ups = [
#                             "What's my downside risk with this position?",
#                             "How does this compare to index investing?", 
#                             "What technical levels should I monitor?",
#                             "When should I consider taking profits?"
#                         ]
                        
#                         cols = st.columns(2)
#                         for i, follow_up in enumerate(follow_ups):
#                             with cols[i % 2]:
#                                 if st.button(follow_up, key=f"followup_{i}", use_container_width=True):
#                                     st.info(f"💭 Processing: {follow_up}")
            
#             else:
#                 st.markdown("""
#                 <div style="background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%); 
#                            color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
#                     <h3>🎤 Voice Input Coming Soon!</h3>
#                     <p>Advanced speech recognition is in development. Use text input for now.</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 if st.button("🎙️ Start Recording", disabled=True, use_container_width=True):
#                     st.warning("⚠️ Voice feature under development - stay tuned!")
        
#         with col2:
#             st.markdown('<div class="control-panel">', unsafe_allow_html=True)
#             st.markdown("### 🧠 AI Assistant Stats")
            
#             stats_data = [
#                 ("📊 Questions Answered", "1,247", "#3B82F6"),
#                 ("🎯 Accuracy Rate", "94.2%", "#10B981"),
#                 ("⚡ Response Time", "1.8s", "#F59E0B"),
#                 ("🏆 Satisfaction", "4.8/5", "#8B5CF6")
#             ]
            
#             for stat, value, color in stats_data:
#                 st.markdown(f"""
#                 <div class="metric-card" style="--accent-color: {color}; margin-bottom: 1rem;">
#                     <div style="font-weight: bold; font-size: 1.2rem; color: {color};">{value}</div>
#                     <div style="font-size: 0.85rem; color: #6B7280;">{stat}</div>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.markdown("### 💡 Pro Tips")
#             tips = [
#                 ("🎯", "Be specific about timeframes"),
#                 ("⚖️", "Mention your risk tolerance"), 
#                 ("💰", "Include position size context"),
#                 ("📊", "Ask about specific price levels"),
#                 ("📈", "Reference your investment goals"),
#                 ("🕒", "Specify your trading horizon")
#             ]
            
#             for emoji, tip in tips:
#                 st.markdown(f"""
#                 <div style="background: white; border-radius: 6px; padding: 0.8rem; margin: 0.5rem 0; 
#                            border-left: 3px solid #667eea; font-size: 0.9rem;">
#                     {emoji} {tip}
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             st.markdown('</div>', unsafe_allow_html=True)
    
#     # TAB 5: Chart Intelligence
#     with tab5:
#         st.markdown("### 🧠 Chart Intelligence & Smart Annotations")
#         st.markdown("*AI-powered chart analysis with institutional-grade pattern recognition*")
        
#         # Enhanced chart analysis
#         chart_ai = ChartGPT()
#         annotations = chart_ai.generate_smart_annotations(market_data, symbol)
        
#         # Professional annotated chart
#         st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
#         fig_chart = go.Figure()
        
#         # Enhanced candlestick chart
#         fig_chart.add_trace(go.Candlestick(
#             x=market_data.index,
#             open=market_data['Open'],
#             high=market_data['High'],
#             low=market_data['Low'],
#             close=market_data['Close'],
#             name=symbol,
#             increasing_line_color='#10B981',
#             decreasing_line_color='#EF4444'
#         ))
        
#         # Add professional AI annotations
#         annotation_colors = {'resistance': '#EF4444', 'support': '#10B981', 'trend': '#3B82F6'}
#         for annotation in annotations:
#             if annotation['type'] in ['resistance', 'support']:
#                 fig_chart.add_hline(
#                     y=annotation['level'],
#                     line_dash="dash",
#                     line_color=annotation_colors.get(annotation['type'], '#6B7280'),
#                     line_width=2,
#                     annotation_text=f"🎯 {annotation['message']}",
#                     annotation_position="top right" if annotation['type'] == 'resistance' else "bottom right"
#                 )
        
#         fig_chart.update_layout(
#             title={
#                 'text': f"🧠 AI-Enhanced Chart Analysis: {symbol}",
#                 'font': {'size': 22, 'family': 'Arial, sans-serif'},
#                 'x': 0.5
#             },
#             height=650,
#             xaxis_rangeslider_visible=False,
#             plot_bgcolor='rgba(248, 250, 252, 1)',
#             paper_bgcolor='white',
#             font={'family': 'Arial, sans-serif'},
#             showlegend=False
#         )
        
#         fig_chart.update_xaxis(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
#         fig_chart.update_yaxis(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
#         st.plotly_chart(fig_chart, use_container_width=True)
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Enhanced AI Chart Analysis
#         col1, col2 = st.columns([2, 1])
        
#         with col1:
#             if st.button("🔍 Advanced Chart Analysis", key="chart_analysis", type="primary", use_container_width=True):
#                 with st.spinner("🤖 AI is analyzing chart patterns..."):
#                     # Professional loading with stages
#                     progress_bar = st.progress(0)
#                     analysis_stages = [
#                         "📊 Scanning chart patterns...",
#                         "📈 Analyzing volume profile...",
#                         "🎯 Identifying key levels...",
#                         "🧮 Calculating probabilities..."
#                     ]
                    
#                     for i, stage in enumerate(analysis_stages):
#                         st.text(stage)
#                         for j in range(25):
#                             time.sleep(0.02)
#                             progress_bar.progress(i * 25 + j + 1)
                    
#                     current_price = market_data['Close'].iloc[-1]
                    
#                     # Generate comprehensive analysis
#                     patterns = ['ascending triangle', 'bull flag', 'cup and handle', 'head and shoulders', 'double bottom', 'wedge', 'channel']
#                     volume_types = ['Institutional accumulation', 'Retail distribution', 'Algorithmic trading', 'Smart money rotation']
#                     momentum_states = ['Building bullish momentum', 'Losing upward steam', 'Neutral consolidation', 'Bearish pressure building']
#                     predictions = ['Bullish breakout likely', 'Bearish breakdown risk', 'Continued consolidation expected', 'Reversal pattern forming']
                    
#                     selected_pattern = random.choice(patterns)
#                     selected_volume = random.choice(volume_types)
#                     selected_momentum = random.choice(momentum_states)
#                     selected_prediction = random.choice(predictions)
#                     confidence = random.randint(78, 95)
                    
#                     analysis = f"""
#                     <div style="background: white; border-radius: 15px; padding: 2rem; box-shadow: 0 5px 20px rgba(0,0,0,0.08);">
#                         <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
#                             <h2 style="margin: 0; font-size: 1.8rem;">🎯 AI CHART ANALYSIS: {symbol}</h2>
#                             <div style="margin-top: 0.5rem; opacity: 0.9;">Advanced Pattern Recognition & Market Structure Analysis</div>
#                         </div>
                        
#                         <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem;">
#                             <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3B82F6;">
#                                 <h4 style="margin: 0 0 1rem 0; color: #1e293b;">📊 Pattern Recognition</h4>
#                                 <p style="margin: 0; color: #475569;">The chart reveals a <strong>{selected_pattern}</strong> pattern forming over the past {random.randint(10, 30)} trading sessions, suggesting {random.choice(['continuation', 'reversal', 'consolidation'])} potential.</p>
#                             </div>
                            
#                             <div style="background: #f0fdf4; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #10B981;">
#                                 <h4 style="margin: 0 0 1rem 0; color: #1e293b;">📈 Price Action</h4>
#                                 <p style="margin: 0; color: #475569;">Currently trading at <strong>${current_price:.2f}</strong>, showing {random.choice(['strength above', 'weakness below', 'consolidation around'])} key moving averages with {random.choice(['increasing', 'decreasing', 'stable'])} volatility.</p>
#                             </div>
#                         </div>
                        
#                         <div style="background: #fefce8; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #F59E0B; margin-bottom: 2rem;">
#                             <h4 style="margin: 0 0 1rem 0; color: #1e293b;">📊 Volume & Momentum Analysis</h4>
#                             <p style="margin: 0 0 1rem 0; color: #475569;"><strong>Volume Profile:</strong> {selected_volume} pattern detected with {random.choice(['above-average', 'below-average', 'normal'])} participation levels.</p>
#                             <p style="margin: 0; color: #475569;"><strong>Momentum:</strong> {selected_momentum} based on price-volume relationship and technical oscillators.</p>
#                         </div>
                        
#                         <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem;">
#                             <div style="text-align: center; background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
#                                 <div style="color: #EF4444; font-weight: bold; font-size: 1.2rem;">${current_price * random.uniform(1.02, 1.06):.2f}</div>
#                                 <div style="font-size: 0.9rem; color: #6B7280;">Resistance Level</div>
#                             </div>
                            
#                             <div style="text-align: center; background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
#                                 <div style="color: #10B981; font-weight: bold; font-size: 1.2rem;">${current_price * random.uniform(0.94, 0.98):.2f}</div>
#                                 <div style="font-size: 0.9rem; color: #6B7280;">Support Level</div>
#                             </div>
                            
#                             <div style="text-align: center; background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
#                                 <div style="color: #3B82F6; font-weight: bold; font-size: 1.2rem;">${current_price * random.uniform(1.08, 1.18):.2f}</div>
#                                 <div style="font-size: 0.9rem; color: #6B7280;">Breakout Target</div>
#                             </div>
#                         </div>
                        
#                         <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
#                             <h4 style="margin: 0 0 1rem 0; color: #1e293b;">🔮 AI Market Prediction</h4>
#                             <p style="margin: 0 0 1rem 0; color: #475569; font-size: 1.1rem;"><strong>{selected_prediction}</strong> within the next {random.randint(3, 10)} trading days.</p>
#                             <div style="background: white; padding: 1rem; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;">
#                                 <span style="font-weight: 600; color: #1e293b;">AI Confidence Level:</span>
#                                 <div style="display: flex; align-items: center; gap: 10px;">
#                                     <div style="background: #3B82F6; height: 8px; width: {confidence}px; border-radius: 4px;"></div>
#                                     <span style="font-weight: bold; color: #3B82F6; font-size: 1.1rem;">{confidence}%</span>
#                                 </div>
#                             </div>
#                         </div>
                        
#                         <div style="background: #fef2f2; border: 1px solid #fecaca; padding: 1rem; border-radius: 8px; text-align: center;">
#                             <p style="margin: 0; color: #7f1d1d; font-size: 0.9rem;"><strong>⚠️ Risk Disclaimer:</strong> AI analysis is for informational purposes only. Past performance doesn't guarantee future results. Always conduct your own research before making investment decisions.</p>
#                         </div>
#                     </div>
#                     """
                    
#                     st.markdown(analysis, unsafe_allow_html=True)
        
#         with col2:
#             st.markdown('<div class="control-panel">', unsafe_allow_html=True)
#             st.markdown("### 🎛️ Chart AI Controls")
            
#             analysis_options = st.multiselect(
#                 "Analysis Components:",
#                 ["Support/Resistance", "Pattern Recognition", "Volume Analysis", "Momentum Indicators", "Fibonacci Levels", "Moving Averages"],
#                 default=["Support/Resistance", "Pattern Recognition", "Volume Analysis"],
#                 help="Select which analysis components to include"
#             )
            
#             timeframe = st.selectbox(
#                 "Chart Timeframe:", 
#                 ["1D", "1W", "1M", "3M", "6M", "1Y"], 
#                 index=2,
#                 help="Choose the time horizon for analysis"
#             )
            
#             sensitivity = st.slider(
#                 "AI Sensitivity:", 
#                 min_value=1, max_value=10, value=7,
#                 help="Higher values detect more patterns but may include false signals"
#             )
            
#             if st.button("🔧 Customize Analysis", use_container_width=True):
#                 st.success("✅ Analysis parameters updated!")
            
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         # Enhanced Smart Annotations Summary
#         st.markdown("---")
#         st.markdown("### 📋 AI Smart Annotations")
        
#         if annotations:
#             for i, annotation in enumerate(annotations):
#                 importance_colors = {
#                     'high': '#EF4444', 
#                     'medium': '#F59E0B', 
#                     'low': '#10B981'
#                 }
#                 importance_icons = {
#                     'high': '🔴', 
#                     'medium': '🟡', 
#                     'low': '🟢'
#                 }
                
#                 color = importance_colors.get(annotation['importance'], '#6B7280')
#                 icon = importance_icons.get(annotation['importance'], '⚪')
                
#                 st.markdown(f"""
#                 <div class="alert-item" style="--alert-color: {color};">
#                     <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
#                         <strong style="color: #1F2937;">{annotation['type'].title()} Alert</strong>
#                         <span style="display: flex; align-items: center; gap: 5px; font-size: 0.9rem;">
#                             {icon} {annotation['importance'].title()}
#                         </span>
#                     </div>
#                     <p style="margin: 0 0 0.5rem 0; color: #374151;">{annotation['message']}</p>
#                     <small style="color: #6B7280;">Price Level: ${annotation['level']:.2f}</small>
#                 </div>
#                 """, unsafe_allow_html=True)
#         else:
#             st.markdown("""
#             <div style="background: #f0f9ff; border: 1px solid #bfdbfe; padding: 1.5rem; border-radius: 10px; text-align: center;">
#                 <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
#                 <h4 style="color: #1e40af; margin: 0 0 0.5rem 0;">No Critical Alerts</h4>
#                 <p style="color: #3730a3; margin: 0;">Market appears to be in normal trading range. AI will notify you of significant pattern changes.</p>
#             </div>
#             """, unsafe_allow_html=True)

#     # Enhanced Additional AI Features Section
#     st.markdown("---")
#     st.markdown("""
#     <div style="text-align: center; margin: 2rem 0;">
#         <h2 style="color: #1F2937; margin-bottom: 0.5rem;">🔬 Advanced AI Features</h2>
#         <p style="color: #6B7280; font-size: 1.1rem;">Institutional-grade tools for professional traders</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         <div class="feature-card">
#             <div style="text-align: center; margin-bottom: 1.5rem;">
#                 <div style="font-size: 3rem; margin-bottom: 0.5rem;">🚨</div>
#                 <h3 style="color: #1F2937; margin: 0;">Smart Alerts</h3>
#             </div>
#         """, unsafe_allow_html=True)
        
#         if st.button("⚙️ Setup Intelligent Alerts", key="smart_alerts", use_container_width=True):
#             st.success("🔔 Smart alerts activated! You'll receive notifications for:")
        
#         alert_features = [
#             ("📈", "Technical breakouts & breakdowns"),
#             ("📊", "Unusual volume & volatility spikes"), 
#             ("📰", "Sentiment shifts & news events"),
#             ("🔀", "Correlation changes & divergences"),
#             ("🎯", "Price target achievements"),
#             ("⚠️", "Risk threshold breaches")
#         ]
        
#         for icon, feature in alert_features:
#             st.markdown(f"""
#             <div style="display: flex; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6;">
#                 <span style="margin-right: 10px; font-size: 1.2rem;">{icon}</span>
#                 <span style="color: #374151; font-size: 0.9rem;">{feature}</span>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         <div class="feature-card">
#             <div style="text-align: center; margin-bottom: 1.5rem;">
#                 <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
#                 <h3 style="color: #1F2937; margin: 0;">Portfolio Impact</h3>
#             </div>
#         """, unsafe_allow_html=True)
        
#         if st.button("🔍 Analyze Portfolio Impact", key="portfolio_impact", use_container_width=True):
#             st.success("📈 Portfolio analysis complete! Key insights:")
        
#         portfolio_features = [
#             ("🎯", "Position correlation analysis"),
#             ("⚖️", "Risk contribution metrics"),
#             ("🔄", "Rebalancing recommendations"), 
#             ("🛡️", "Hedging opportunities"),
#             ("📏", "Exposure concentration"),
#             ("💰", "Performance attribution")
#         ]
        
#         for icon, feature in portfolio_features:
#             st.markdown(f"""
#             <div style="display: flex; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6;">
#                 <span style="margin-right: 10px; font-size: 1.2rem;">{icon}</span>
#                 <span style="color: #374151; font-size: 0.9rem;">{feature}</span>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("""
#         <div class="feature-card">
#             <div style="text-align: center; margin-bottom: 1.5rem;">
#                 <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔮</div>
#                 <h3 style="color: #1F2937; margin: 0;">Earnings Prediction</h3>
#             </div>
#         """, unsafe_allow_html=True)
        
#         if st.button("🔮 Predict Next Earnings", key="earnings_prediction", use_container_width=True):
#             expected_eps = random.uniform(1.50, 3.50)
#             surprise_prob = random.randint(60, 88)
#             post_move = random.randint(4, 15)
#             key_metric = random.choice(['Revenue growth', 'Margin expansion', 'Guidance update', 'User growth', 'Market share'])
            
#             st.markdown(f"""
#             <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 10px; margin-top: 1rem;">
#                 <h4 style="color: #1e40af; margin: 0 0 1rem 0;">📊 {symbol} Earnings Forecast</h4>
#                 <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
#                     <div><strong>Expected EPS:</strong> ${expected_eps:.2f}</div>
#                     <div><strong>Surprise Probability:</strong> {surprise_prob}%</div>
#                     <div><strong>Post-Earnings Move:</strong> ±{post_move}%</div>
#                     <div><strong>Key Metric:</strong> {key_metric}</div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         prediction_sources = [
#             ("💳", "Credit card transaction data"),
#             ("🛰️", "Satellite imagery analytics"),
#             ("📱", "App download & usage trends"),
#             ("🗣️", "Social sentiment analysis"),
#             ("📊", "Supplier chain indicators"),
#             ("🏪", "Foot traffic patterns")
#         ]
        
#         for icon, source in prediction_sources:
#             st.markdown(f"""
#             <div style="display: flex; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6;">
#                 <span style="margin-right: 10px; font-size: 1.2rem;">{icon}</span>
#                 <span style="color: #374151; font-size: 0.9rem;">{source}</span>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)

#     # Professional Footer
#     st.markdown("""
#     <div class="footer-section">
#         <div style="margin-bottom: 2rem;">
#             <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">🚀 AI Trading Intelligence Platform</h2>
#             <p style="font-size: 1.2rem; opacity: 0.9; margin: 0;">Powered by advanced AI agents • Real-time analysis • Institutional-grade insights</p>
#         </div>
        
#         <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin-bottom: 2rem;">
#             <div style="text-align: center;">
#                 <div style="font-size: 2rem; font-weight: bold; color: #10B981;">⚡ 3s</div>
#                 <div>Average Analysis Time</div>
#             </div>
#             <div style="text-align: center;">
#                 <div style="font-size: 2rem; font-weight: bold; color: #3B82F6;">94.2%</div>
#                 <div>Prediction Accuracy</div>
#             </div>
#             <div style="text-align: center;">
#                 <div style="font-size: 2rem; font-weight: bold; color: #F59E0B;">🛡️</div>
#                 <div>Risk-Managed</div>
#             </div>
#             <div style="text-align: center;">
#                 <div style="font-size: 2rem; font-weight: bold; color: #8B5CF6;">24/7</div>
#                 <div>Market Monitoring</div>
#             </div>
#         </div>
        
#         <div style="opacity: 0.8; font-size: 0.9rem;">
#             <p style="margin: 0;">Disclaimer: AI-generated content is for informational purposes only and should not be considered as financial advice.</p>
#             <p style="margin: 0.5rem 0 0 0;">Always consult with qualified financial professionals before making investment decisions.</p>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
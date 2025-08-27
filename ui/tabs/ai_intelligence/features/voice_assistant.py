import streamlit as st
import pandas as pd
import numpy as np
import ta

class VoiceTradingAssistant:
    """
    Processes natural language trading queries using real market data
    to provide data-driven analysis and responses.
    """
    
    def __init__(self):
        """Initializes the assistant with a categorized list of sample queries."""
        self.sample_queries = [
            "Should I buy more on this dip?",
            "What's the risk-reward on a swing trade here?",
            "What are the key support and resistance levels?",
            "How much should I position size for this trade?",
        ]
        self.categorized_queries = {
            "📈 Technical Analysis": [
                "What are the key support and resistance levels?",
                "Is the stock overbought or oversold right now?",
                "What is the current trend based on moving averages?",
                "Analyze the recent volume patterns.",
                "What does the MACD indicator suggest?",
                "How are the Bollinger Bands looking?",
            ],
            "🎯 Entry & Exit Strategy": [
                "Should I buy more on this dip?",
                "Is this a good entry point for a long-term hold?",
                "Where should I place a stop-loss for a swing trade?",
                "What's a reasonable profit target for the current setup?",
                "Analyze the entry point for a short position.",
            ],
            "⚖️ Risk Management": [
                "What's the risk-reward on a swing trade here?",
                "How much should I position size for this trade?",
                "Analyze the stock's recent volatility.",
                "How does the current volatility compare to its history?",
            ],
            "📖 Fundamental Analysis (Placeholder)": [
                "How does this compare to sector peers?",
                "What would happen if earnings disappoint?",
                "What are the key financial ratios for this company?",
                "Summarize the latest news for this stock.",
            ],
            "💡 Trade Ideas & Scenarios": [
                "What's the probability of a 10% move this month?",
                "Simulate a 'what-if' scenario if the market drops 5%.",
                "Are there any bullish chart patterns forming?",
                "Are there any bearish chart patterns forming?",
            ]
        }
    
    def process_query(self, query: str, symbol: str, data: pd.DataFrame, portfolio_context=None) -> str:
        """
        Processes a natural language trading question by routing it to the
        appropriate analysis method based on keywords.
        """
        if data is None or data.empty or len(data) < 20:
            return f"⚠️ Insufficient data for {symbol} to perform analysis. Please select a different stock."

        query_lower = query.lower()
        current_price = data['Close'].iloc[-1]
        
        # Routing logic based on query keywords
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
    
    def _analyze_dip_buying(self, symbol: str, data: pd.DataFrame, current_price: float) -> str:
        """Analyzes whether the current dip is a buying opportunity."""
        high_20d = data['High'].tail(20).max()
        low_20d = data['Low'].tail(20).min()
        pullback_pct = ((high_20d - current_price) / high_20d) * 100
        rsi = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
        
        verdict = "🔴 UNFAVORABLE"
        if pullback_pct > 5 and rsi < 40:
            verdict = "🟢 FAVORABLE"
        elif pullback_pct > 3 and rsi < 50:
            verdict = "🟡 NEUTRAL"

        return f"""
## 🎯 **Dip Analysis for {symbol}**

**Current Situation:** The stock is trading at **${current_price:,.2f}**, which is a **{pullback_pct:.1f}% pullback** from its 20-day high of ${high_20d:,.2f}. The current RSI is **{rsi:.1f}**.

**Technical Assessment:**
- **Support Level:** The 20-day low is at **${low_20d:,.2f}**.
- **RSI Reading:** An RSI of {rsi:.1f} suggests the stock is {'approaching oversold conditions' if rsi < 40 else 'in a neutral zone' if rsi < 60 else 'still has room to fall'}.

**Dip-Buying Verdict: {verdict}**
- {'A meaningful pullback combined with a low RSI presents a good risk/reward setup.' if verdict == "🟢 FAVORABLE" else 'The pullback is moderate. It may be prudent to wait for a clearer signal or a deeper dip.' if verdict == "🟡 NEUTRAL" else 'This is a minor pullback and the RSI is not yet in a buy zone. Buying now could be risky.'}

**Action Plan:**
- **Entry Strategy:** Consider scaling in with a partial position. A potential next entry could be near the 20-day low of **${low_20d:,.2f}**.
- **Stop Loss:** A logical stop loss would be just below the recent support, around **${low_20d * 0.98:,.2f}**.
- **Target:** A primary target would be a retest of the recent high at **${high_20d:,.2f}**.
"""

    def _analyze_risk_reward(self, symbol: str, data: pd.DataFrame, current_price: float) -> str:
        """Analyzes the risk-reward profile for a potential swing trade."""
        volatility = data['Close'].pct_change().tail(60).std() * np.sqrt(252) * 100
        atr = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14).iloc[-1]
        
        stop_loss_price = current_price - (2 * atr)
        target_price = current_price + (3 * atr)
        risk = current_price - stop_loss_price
        reward = target_price - current_price
        risk_reward_ratio = reward / risk if risk > 0 else float('inf')

        return f"""
## ⚖️ **Risk/Reward Analysis for {symbol}**

**Swing Trade Setup (ATR-based):**
- **Entry:** ${current_price:,.2f} (current price)
- **Stop Loss (2x ATR):** **${stop_loss_price:,.2f}** (Risking ${risk:,.2f} per share)
- **Profit Target (3x ATR):** **${target_price:,.2f}** (Potential reward of ${reward:,.2f} per share)

**Risk Metrics:**
- **Annualized Volatility:** {volatility:.1f}% ({'High' if volatility > 40 else 'Moderate' if volatility > 25 else 'Low'} risk profile)
- **Average True Range (ATR):** ${atr:,.2f}, indicating the average daily price movement.
- **Risk/Reward Ratio:** **1:{risk_reward_ratio:.1f}** ({'✅ Favorable' if risk_reward_ratio >= 1.5 else '⚠️ Marginal'})

**Verdict:**
- Based on this ATR-based strategy, the risk/reward ratio is **{risk_reward_ratio:.1f} to 1**. A ratio above 1.5 is generally considered favorable for a swing trade. The stock's volatility is currently {'high' if volatility > 40 else 'moderate' if volatility > 25 else 'low'}.
"""

    def _general_analysis(self, symbol: str, data: pd.DataFrame, current_price: float) -> str:
        """Provides a general, data-driven analysis of the stock."""
        rsi = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
        macd_line = ta.trend.macd(data['Close']).iloc[-1]
        macd_signal = ta.trend.macd_signal(data['Close']).iloc[-1]
        
        trend = "Bullish" if macd_line > macd_signal and current_price > data['Close'].rolling(50).mean().iloc[-1] else "Bearish" if macd_line < macd_signal and current_price < data['Close'].rolling(50).mean().iloc[-1] else "Sideways"
        
        return f"""
## 🤖 **AI Analysis for {symbol}**

**Current Price:** ${current_price:,.2f}
**24h Volume:** {data['Volume'].iloc[-1]:,.0f} shares
**50-Day Avg Volume:** {data['Volume'].rolling(50).mean().iloc[-1]:,.0f}
**Overall Trend:** **{trend}** momentum

**Key Insights:**
- **RSI (14):** **{rsi:.1f}**. This indicates the stock is currently {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'in a Neutral range'}.
- **MACD:** The MACD line is currently {'above' if macd_line > macd_signal else 'below'} its signal line, suggesting {'bullish' if macd_line > macd_signal else 'bearish'} momentum.
- **Volume:** Today's volume is {'above' if data['Volume'].iloc[-1] > data['Volume'].rolling(50).mean().iloc[-1] else 'below'} the 50-day average, indicating {'stronger' if data['Volume'].iloc[-1] > data['Volume'].rolling(50).mean().iloc[-1] else 'weaker'} conviction in the recent price action.

*Ask me more specific questions for detailed analysis!*
"""

    def _identify_key_levels(self, symbol: str, data: pd.DataFrame, current_price: float) -> str:
        """Identifies key support and resistance levels from historical data."""
        # Pivot Points calculation
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        high_52wk = data['High'].rolling(252).max().iloc[-1]
        low_52wk = data['Low'].rolling(252).min().iloc[-1]

        return f"""
## 📊 **Key Levels for {symbol}**

**Support Levels (areas to watch for buying interest):**
- **Primary Support (S1 Pivot):** **${s1:,.2f}**
- **Secondary Support (S2 Pivot):** **${s2:,.2f}**
- **52-Week Low:** ${low_52wk:,.2f}

**Resistance Levels (areas to watch for selling pressure):**
- **Primary Resistance (R1 Pivot):** **${r1:,.2f}**
- **Secondary Resistance (R2 Pivot):** **${r2:,.2f}**
- **52-Week High:** ${high_52wk:,.2f}

**Current Position:** The stock at **${current_price:,.2f}** is currently trading between the primary support and resistance pivot points. A decisive move above **${r1:,.2f}** or below **${s1:,.2f}** could signal the next short-term direction.
"""

    def _recommend_position_size(self, symbol: str, data: pd.DataFrame, current_price: float) -> str:
        """Recommends a position size based on the stock's volatility."""
        volatility = data['Close'].pct_change().tail(60).std() * np.sqrt(252) * 100
        
        risk_rating = "High" if volatility > 50 else "Medium" if volatility > 25 else "Low"
        
        sizing_advice = "a smaller position (e.g., 1-3% of portfolio)" if risk_rating == "High" else "a standard position (e.g., 3-5% of portfolio)" if risk_rating == "Medium" else "a larger position (e.g., 5-7% of portfolio)"

        return f"""
## ⚖️ **Position Sizing for {symbol}**

**Risk Assessment:**
- **60-Day Annualized Volatility:** **{volatility:.1f}%**
- **Risk Rating:** This volatility gives the stock a **{risk_rating}** risk rating.

**Recommended Sizing:**
- Based on its current risk profile, it is advisable to consider **{sizing_advice}**.
- **High Volatility Stocks:** Require smaller position sizes to manage the potential for large price swings.
- **Low Volatility Stocks:** May allow for larger position sizes as the risk of sudden, large losses is lower.

**Risk Management Principle:**
- Always determine your stop-loss level *before* entering a trade. Your position size should be calculated so that a stop-loss event does not result in a loss greater than your predefined portfolio risk limit (e.g., 1-2% of total portfolio value).
"""
    # --- Placeholder methods for queries that require external data/APIs ---
    
    def _compare_to_sector(self, symbol, data, current_price):
        """Placeholder for sector comparison."""
        return f"Sector comparison for **{symbol}** is an advanced feature requiring sector-wide data. A full implementation would compare P/E, growth, and performance against peers like X, Y, and Z."

    def _analyze_earnings_risk(self, symbol, data, current_price):
        """Placeholder for earnings risk analysis."""
        return f"Analyzing earnings risk for **{symbol}** requires options market data to calculate the implied move and historical earnings data. The market is currently pricing in an estimated post-earnings move based on options volatility."

    def _analyze_entry_point(self, symbol, data, current_price):
        """Analyzes the quality of the current price as an entry point."""
        rsi = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
        bbands = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        in_buy_zone = current_price < bbands.bollinger_lband().iloc[-1] or rsi < 35
        
        rating = "Strong Buy" if in_buy_zone and rsi < 30 else "Good" if in_buy_zone else "Fair"

        return f"""
## 🎯 **Entry Point Analysis for {symbol}**

**Current Setup:**
- **Price vs. Bollinger Bands:** The price is currently {'below the lower band (potential buy signal)' if current_price < bbands.bollinger_lband().iloc[-1] else 'above the upper band (potential sell signal)' if current_price > bbands.bollinger_hband().iloc[-1] else 'within the bands'}.
- **RSI:** The RSI is at **{rsi:.1f}**.

**Entry Strategy Grade: {rating}**
- A **'Good'** or **'Strong Buy'** rating often occurs when the price touches or breaches the lower Bollinger Band and the RSI is low, indicating an oversold condition that may be due for a reversal.
- A **'Fair'** rating suggests waiting for a better opportunity, as the stock is neither overbought nor oversold.
"""

class VoiceAssistantTab:
    """
    The UI component for the Voice Assistant tab. It handles user interaction,
    manages state, and displays the analysis from the VoiceTradingAssistant.
    """
    def __init__(self, symbol: str, market_data: pd.DataFrame, ui_components):
        self.symbol = symbol
        self.market_data = market_data
        self.ui = ui_components
        self.assistant = VoiceTradingAssistant()
        
        # Initialize session state
        if 'voice_query' not in st.session_state:
            st.session_state.voice_query = ""
        if 'voice_response' not in st.session_state:
            st.session_state.voice_response = ""

    def render(self):
        """Renders the Voice Assistant tab UI."""
        st.subheader(f"🎤 Voice Assistant for {self.symbol}")
        
        # --- Input Area ---
        query_input = st.text_input(
            "Ask a trading question...", 
            placeholder="e.g., 'What are the key support and resistance levels?'",
            key="voice_query_input"
        )

        # --- Quick Query Buttons ---
        st.write("**Or try a sample query:**")
        cols = st.columns(4)
        for i, sample in enumerate(self.assistant.sample_queries):
            if cols[i % 4].button(sample, key=f"sample_{i}"):
                query_input = sample

        # --- Sample Query Dropdowns ---
        st.markdown("---")
        
        # Dropdown for categories
        categories = list(self.assistant.categorized_queries.keys())
        selected_category = st.selectbox(
            "Select a category for more questions:",
            categories,
            key="category_select"
        )

        # Dropdown for questions based on selected category
        if selected_category:
            questions_for_category = self.assistant.categorized_queries[selected_category]
            selected_question = st.selectbox(
                "Select a question:",
                questions_for_category,
                key="question_select"
            )

            # Button to submit the selected question
            if st.button("Analyze Selected Question", key="analyze_button"):
                query_input = selected_question
        
        # --- Process Query ---
        if query_input and (query_input != st.session_state.get('last_voice_query')):
            st.session_state.last_voice_query = query_input
            with st.spinner(f"Analyzing '{query_input}' for {self.symbol}..."):
                response = self.assistant.process_query(query_input, self.symbol, self.market_data)
                st.session_state.voice_response = response
        
        # --- Display Response ---
        if st.session_state.get('voice_response'):
            st.markdown("---")
            st.markdown(st.session_state.voice_response, unsafe_allow_html=True)

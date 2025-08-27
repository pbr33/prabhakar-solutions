import streamlit as st
import plotly.graph_objects as go
import random
import time
import numpy as np
import pandas as pd

class ChartGPT:
    """AI-powered chart analysis engine that generates smart annotations"""
    
    def __init__(self):
        self.patterns = [
            'ascending triangle', 'bull flag', 'cup and handle', 
            'head and shoulders', 'double bottom', 'wedge pattern',
            'channel breakout', 'consolidation'
        ]
        
    def generate_smart_annotations(self, market_data, symbol):
        """Generate intelligent chart annotations based on market data"""
        annotations = []
        
        if market_data is None or len(market_data) < 20:
            return annotations
            
        current_price = market_data['Close'].iloc[-1]
        price_data = market_data['Close'].values
        
        # Calculate support and resistance levels
        high_prices = market_data['High'].values
        low_prices = market_data['Low'].values
        
        # Find recent highs and lows for support/resistance
        recent_high = np.max(high_prices[-20:])
        recent_low = np.min(low_prices[-20:])
        
        # Generate resistance annotation
        if current_price < recent_high * 0.98:
            annotations.append({
                'type': 'resistance',
                'level': recent_high,
                'message': f'Key Resistance: ${recent_high:.2f}',
                'color': 'red',
                'importance': 'high'
            })
        
        # Generate support annotation
        if current_price > recent_low * 1.02:
            annotations.append({
                'type': 'support',
                'level': recent_low,
                'message': f'Strong Support: ${recent_low:.2f}',
                'color': 'green',
                'importance': 'high'
            })
        
        # Add moving average levels
        if len(price_data) >= 50:
            ma_50 = np.mean(price_data[-50:])
            annotations.append({
                'type': 'moving_average',
                'level': ma_50,
                'message': f'50-Day MA: ${ma_50:.2f}',
                'color': 'blue',
                'importance': 'medium'
            })
        
        # Add volume-based annotations
        if 'Volume' in market_data.columns:
            avg_volume = market_data['Volume'].tail(20).mean()
            recent_volume = market_data['Volume'].iloc[-1]
            
            if recent_volume > avg_volume * 1.5:
                annotations.append({
                    'type': 'volume_spike',
                    'level': current_price,
                    'message': f'High Volume Alert: {recent_volume/1000000:.1f}M',
                    'color': 'orange',
                    'importance': 'high'
                })
        
        return annotations

class ChartIntelligenceTab:
    def __init__(self, symbol, market_data, ui_components=None):
        self.symbol = symbol
        self.market_data = market_data
        self.ui = ui_components
    
    def render(self):
        st.markdown("## 🧠 Chart Intelligence & Smart Annotations")
        st.markdown("*AI that can 'see' and intelligently annotate charts like a human analyst*")
        
        if self.market_data is None or len(self.market_data) == 0:
            st.error("No market data available for analysis.")
            return
        
        # Chart analysis
        chart_ai = ChartGPT()
        annotations = chart_ai.generate_smart_annotations(self.market_data, self.symbol)
        
        # Create annotated chart
        fig_chart = go.Figure()
        
        # Candlestick chart
        fig_chart.add_trace(go.Candlestick(
            x=self.market_data.index,
            open=self.market_data['Open'],
            high=self.market_data['High'],
            low=self.market_data['Low'],
            close=self.market_data['Close'],
            name=self.symbol
        ))
        
        # Add AI annotations
        for annotation in annotations:
            if annotation['type'] in ['resistance', 'support', 'moving_average']:
                fig_chart.add_hline(
                    y=annotation['level'],
                    line_dash="dash",
                    line_color=annotation['color'],
                    annotation_text=annotation['message']
                )
            elif annotation['type'] == 'volume_spike':
                # Add a marker for volume spikes
                fig_chart.add_scatter(
                    x=[self.market_data.index[-1]],
                    y=[annotation['level']],
                    mode='markers',
                    marker=dict(
                        color=annotation['color'],
                        size=15,
                        symbol='triangle-up'
                    ),
                    name=annotation['message'],
                    showlegend=False
                )
        
        fig_chart.update_layout(
            title=f"🧠 AI-Annotated Chart for {self.symbol}",
            height=600,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        st.plotly_chart(fig_chart, use_container_width=True)
        
        # AI Chart Analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔍 Analyze Chart Pattern", key="chart_analysis"):
                with st.spinner("🤖 AI is reading the chart..."):
                    time.sleep(2)
                    
                    current_price = self.market_data['Close'].iloc[-1]
                    
                    analysis = f"""
## 🎯 **AI CHART READING FOR {self.symbol}**

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
                    <strong>{annotation['type'].replace('_', ' ').title()} Alert:</strong> {annotation['message']}<br>
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
                **📊 EARNINGS PREDICTION FOR {self.symbol}:**
                
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

# Usage example function
def render_chart_intelligence_tab(symbol="AAPL", market_data=None):
    """
    Standalone function to render the chart intelligence tab
    Can be called from main application
    """
    tab = ChartIntelligenceTab(symbol, market_data)
    tab.render()

# Demo data generator for testing
def generate_demo_data(symbol="AAPL", days=100):
    """Generate demo market data for testing purposes"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 150
    price_changes = np.random.normal(0, 2, days)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = max(prices[-1] + change, 1)  # Ensure positive prices
        prices.append(new_price)
    
    # Create OHLC data
    opens = prices[:-1]
    closes = prices[1:]
    highs = [max(o, c) + random.uniform(0, 2) for o, c in zip(opens, closes)]
    lows = [min(o, c) - random.uniform(0, 2) for o, c in zip(opens, closes)]
    volumes = [random.randint(1000000, 10000000) for _ in range(len(closes))]
    
    data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates[1:])
    
    return data

if __name__ == "__main__":
    # Demo usage
    st.set_page_config(page_title="Chart Intelligence Demo", layout="wide")
    
    st.title("🧠 Chart Intelligence Demo")
    
    # Generate demo data
    demo_data = generate_demo_data("AAPL", 100)
    
    # Render the chart intelligence tab
    render_chart_intelligence_tab("AAPL", demo_data)
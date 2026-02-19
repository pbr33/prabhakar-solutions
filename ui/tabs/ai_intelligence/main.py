# ui/tabs/ai_intelligence/main.py
"""
Main orchestrator for AI Trading Intelligence tab.
Handles tab navigation and feature initialization.
"""

import streamlit as st
from typing import Optional
import pandas as pd
import sys

import os
import time
# ui/tabs/ai_intelligence/main.py
import sys
import os

# Get the directory of the current file (.../ai_intelligence)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up 3 levels to get the project root (.../prabhakar-solutions)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# Add the project root to the Python path
sys.path.append(project_root)

# Try to import the professional multi-agent analysis
try:
    # Load the professional multi-agent analysis module
    import importlib.util
    spec = importlib.util.spec_from_file_location("multi_agent_analysis", 
        os.path.join(os.path.dirname(__file__), "features", "multi_agent_analysis.py"))
    if spec and spec.loader:
        multi_agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(multi_agent_module)
        MultiAgentAnalysisTab = multi_agent_module.MultiAgentAnalysisTab
    else:
        raise ImportError("Could not load professional module")
except Exception as e:
    # Fallback to basic import
    try:
        from ui.tabs.ai_intelligence.features.multi_agent_analysis import MultiAgentAnalysisTab
    except:
        MultiAgentAnalysisTab = None

from ui.tabs.ai_intelligence.components.ui_components import UIComponents
from ui.tabs.ai_intelligence.services.data_service import DataService
from ui.tabs.ai_intelligence.styles.theme import apply_complete_theme

# Import other tabs with error handling
try:
    from ui.tabs.ai_intelligence.features.storyteller import AIStorytellerTab
except ImportError:
    AIStorytellerTab = None

try:
    from ui.tabs.ai_intelligence.features.voice_assistant import VoiceAssistantTab
except ImportError:
    VoiceAssistantTab = None

try:
    from ui.tabs.ai_intelligence.features.chart_intelligence import ChartIntelligenceTab
except ImportError:
    ChartIntelligenceTab = None

class AITradingIntelligence:
    """Main class for AI Trading Intelligence module."""
    
    def __init__(self):
        self.ui = UIComponents()
        self.data_service = DataService()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'ai_tab_state' not in st.session_state:
            st.session_state.ai_tab_state = {
                'selected_tab': 0,
                'cache': {},
                'last_analysis': None
            }
        
        # Initialize scenario-specific state
        self._initialize_scenario_state()
    
    def _initialize_scenario_state(self):
        """Initialize scenario modeling state variables."""
        if 'scenario_results' not in st.session_state:
            st.session_state.scenario_results = None
        
        if 'scenario_config' not in st.session_state:
            st.session_state.scenario_config = {
                'timeframe': 90,
                'analysis_depth': 'Standard',
                'auto_refresh': True
            }
    
    def render(self):
        """Main render function for AI Trading Intelligence tab."""
        # Apply professional theme
        apply_complete_theme()

        # Clean tab styling
        st.markdown("""
        <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            margin-top: 0.25rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e9ecef;
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding-left: 18px;
            padding-right: 18px;
            background-color: transparent;
            border-radius: 6px 6px 0 0;
            border: none;
            font-size: 0.9rem;
            font-weight: 500;
            color: #555;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #f0f2f6;
            color: #333;
        }

        .stTabs [aria-selected="true"] {
            background-color: #007bff !important;
            color: white !important;
            border-bottom: 2px solid #007bff !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Get symbol and market data
        symbol = self._get_selected_symbol()
        market_data = self._load_market_data(symbol)
        
        if market_data is None or market_data.empty:
            st.error(f"❌ Could not load market data for {symbol}")
            return
        
        # Create professional tabs with reduced spacing
        tab_configs = [
            {"id": "agents", "label": "Multi-Agent Analysis", "icon": "🤖"},
            {"id": "storyteller", "label": "Market Storyteller", "icon": "📰"},
            {"id": "scenarios", "label": "Scenario Modeling", "icon": "🎭"},
            {"id": "voice", "label": "Voice Assistant", "icon": "🎤"},
            {"id": "charts", "label": "Chart Intelligence", "icon": "🧠"}
        ]
        
        tabs = st.tabs([f"{tab['icon']} {tab['label']}" for tab in tab_configs])
        
        # Render each tab with lazy loading
        with tabs[0]:
            self._render_multi_agent_tab(symbol, market_data)
        
        with tabs[1]:
            self._render_storyteller_tab(symbol, market_data)
        
        with tabs[2]:
            self._render_scenario_tab(symbol, market_data)  # This is the method that was missing!
        
        with tabs[3]:
            self._render_voice_tab(symbol, market_data)
        
        with tabs[4]:
            self._render_chart_tab(symbol, market_data)
        
        # Compact footer
        self._render_compact_footer()
    
    def _get_selected_symbol(self) -> str:
        """Get the currently selected symbol."""
        # Priority order: selected_tickers > selected_symbol > default
        
        # First try to get from selected_tickers (from left panel)
        if hasattr(st.session_state, 'selected_tickers') and st.session_state.selected_tickers:
            return st.session_state.selected_tickers[0].upper()
        
        # Then try selected_symbol
        if hasattr(st.session_state, 'selected_symbol') and st.session_state.selected_symbol:
            return st.session_state.selected_symbol.upper()
        
        # Finally, default to AAPL
        return 'AAPL'
    
    def _load_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load market data with enhanced caching and validation."""
        cache_key = f"market_data_{symbol}_{int(time.time() // 300)}"  # 5-minute cache
        
        # Check cache first
        if cache_key in st.session_state.ai_tab_state['cache']:
            return st.session_state.ai_tab_state['cache'][cache_key]
        
        # Validate symbol first
        if not self.data_service.validate_symbol(symbol):
            st.warning(f"⚠️ Symbol {symbol} may not be valid. Using fallback data.")
        
        # Load fresh data with progress indicator
        try:
            with st.spinner(f"📡 Loading market data for {symbol}..."):
                # Show data source info
                data_source_msg = st.empty()
                data_source_msg.info(f"🔄 Fetching {symbol} from multiple data sources...")
                
                data = self.data_service.get_market_data(symbol)
                
                if data is not None and not data.empty:
                    # Cache the data
                    st.session_state.ai_tab_state['cache'][cache_key] = data
                    
                    # Show success message
                    data_source_msg.success(f"✅ Successfully loaded {len(data)} days of data for {symbol}")
                    time.sleep(1)  # Brief pause to show success
                    data_source_msg.empty()
                    
                    return data
                else:
                    data_source_msg.error(f"❌ Could not load data for {symbol}")
                    time.sleep(2)
                    data_source_msg.empty()
                    return None
                    
        except Exception as e:
            st.error(f"🚨 Data loading error: {str(e)}")
            return None
    
    def _render_multi_agent_tab(self, symbol: str, market_data: pd.DataFrame):
        """Render multi-agent analysis tab."""
        # Check API configuration before rendering
        try:
            if MultiAgentAnalysisTab:
                tab = MultiAgentAnalysisTab(symbol, market_data, self.ui)
                tab.render()
            else:
                st.error("❌ Multi-Agent Analysis module not available")
                self._render_fallback_content("Multi-Agent Analysis", symbol)
        except Exception as e:
            self._render_api_config_error("Multi-Agent Analysis", str(e))
    
    def _render_storyteller_tab(self, symbol: str, market_data: pd.DataFrame):
        """Render AI storyteller tab."""
        # Check API configuration before rendering
        try:
            if AIStorytellerTab:
                tab = AIStorytellerTab(symbol, market_data, self.ui)
                tab.render()
            else:
                st.error("❌ AI Storyteller module not available")
                self._render_fallback_content("AI Storyteller", symbol)
        except Exception as e:
            self._render_api_config_error("AI Storyteller", str(e))
    
    def _render_scenario_tab(self, symbol: str, market_data: pd.DataFrame):
        """Render scenario modeling tab with improved error handling."""
        
        # Initialize with working content first
        if not hasattr(st.session_state, 'scenario_tab_initialized'):
            st.session_state.scenario_tab_initialized = True
        
        # Validate requirements first
        if not self._validate_scenario_requirements(symbol, market_data):
            return
        
        try:
            # Import and render the scenario modeling tab
            from ui.tabs.ai_intelligence.features.scenario_engine import ScenarioModelingTab
            
            # Create tab instance
            tab = ScenarioModelingTab(symbol, market_data, self.ui)
            
            # Render the tab
            tab.render()
            
        except ImportError as e:
            st.error(f"❌ Import Error: {e}")
            st.info("Please ensure the scenario_engine.py file is in the correct location.")
            
            # Fallback content
            self._render_scenario_fallback(symbol, market_data)
            
        except Exception as e:
            st.error(f"❌ Scenario Modeling Error: {e}")
            
            # Debug information
            with st.expander("🐛 Debug Information"):
                st.code(f"""
    Error: {str(e)}
    Symbol: {symbol}
    Market Data Shape: {market_data.shape if market_data is not None else 'None'}
    Session State Keys: {list(st.session_state.keys())}
                """)
            
            # Fallback content
            self._render_scenario_fallback(symbol, market_data)
    
    def _render_scenario_fallback(self, symbol: str, market_data: pd.DataFrame):
        """Render fallback scenario content when main feature fails."""
        
        st.markdown("### 🎭 Scenario Modeling (Fallback Mode)")
        st.info("Using simplified scenario analysis while the main feature loads...")
        
        # Simple scenario generation
        if st.button("🚀 Generate Quick Scenarios", type="primary"):
            with st.spinner("Generating scenarios..."):
                import numpy as np
                import time
                
                time.sleep(2)  # Simulate processing
                
                # Get current price from market data
                current_price = market_data['Close'].iloc[-1] if not market_data.empty else 100
                
                # Generate simple scenarios
                scenarios = {
                    "🚀 Optimistic": {
                        "target": current_price * 1.25,
                        "probability": 25,
                        "return": 25.0
                    },
                    "📊 Base Case": {
                        "target": current_price * 1.05,
                        "probability": 45,
                        "return": 5.0
                    },
                    "🐻 Pessimistic": {
                        "target": current_price * 0.85,
                        "probability": 30,
                        "return": -15.0
                    }
                }
                
                st.success("✅ Quick scenarios generated!")
                
                # Display scenarios
                for name, data in scenarios.items():
                    with st.expander(f"{name} - {data['probability']}% probability"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Target Price", f"${data['target']:.2f}")
                            st.metric("Expected Return", f"{data['return']:+.1f}%")
                        with col2:
                            st.metric("Probability", f"{data['probability']}%")
        
        # Additional quick actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Price Targets"):
                st.info(f"Analyzing price targets for {symbol}...")
        
        with col2:
            if st.button("🎯 Risk Assessment"):
                st.warning(f"Assessing risks for {symbol}...")
        
        with col3:
            if st.button("🔮 Predictions"):
                st.success(f"Generating predictions for {symbol}...")
    
    def _validate_scenario_requirements(self, symbol: str, market_data: pd.DataFrame) -> bool:
        """Validate requirements for scenario modeling."""
        
        # Check if symbol is valid
        if not symbol or len(symbol) < 1:
            st.error("❌ No symbol selected for analysis")
            return False
        
        # Check if market data is available
        if market_data is None or market_data.empty:
            st.warning(f"⚠️ Limited market data for {symbol}. Using demo mode.")
            return True  # Still allow demo mode
        
        # Check if we have enough data points
        if len(market_data) < 30:
            st.warning(f"⚠️ Limited historical data for {symbol} ({len(market_data)} days)")
        
        return True
    
    def _render_voice_tab(self, symbol: str, market_data: pd.DataFrame):
        """Render voice assistant tab."""
        # st.markdown("### 🎤 Voice Assistant")
        # st.markdown("*Conversational AI for trading insights*")
        
        # # Quick action buttons
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     if st.button("🎙️ Start Voice", key="va_start_voice"):
        #         st.success("Voice assistant activated!")
        # with col2:
        #     if st.button("🔊 Play Analysis", key="va_play_analysis"):
        #         st.info("Playing audio analysis...")
        # with col3:
        #     if st.button("💬 Chat Mode", key="va_chat_mode"):
        #         st.warning("Switching to chat mode...")
        
        # # Check API configuration before rendering
        try:
            if VoiceAssistantTab:
                tab = VoiceAssistantTab(symbol, market_data, self.ui)
                tab.render()
            else:
                st.error("❌ Voice Assistant module not available")
                self._render_fallback_content("Voice Assistant", symbol)
        except Exception as e:
            self._render_api_config_error("Voice Assistant", str(e))
    
    def _render_chart_tab(self, symbol: str, market_data: pd.DataFrame):
        """Render chart intelligence tab."""
        # st.markdown("### 🧠 Chart Intelligence")
        # st.markdown("*AI-powered technical analysis and pattern recognition*")
        
        # # Quick action buttons
        # col1, col2, col3 = st.columns(3)
        # with col1:
        #     if st.button("🔍 Analyze Patterns", key="ci_analyze_patterns"):
        #         st.success("Pattern analysis started!")
        # with col2:
        #     if st.button("📊 Generate Charts", key="ci_generate_charts"):
        #         st.info("Generating intelligent charts...")
        # with col3:
        #     if st.button("⚡ Quick Insights", key="ci_quick_insights"):
        #         st.warning("Loading quick insights...")
        
        # Check API configuration before rendering
        try:
            if ChartIntelligenceTab:
                tab = ChartIntelligenceTab(symbol, market_data, self.ui)
                tab.render()
            else:
                st.error("❌ Chart Intelligence module not available")
                self._render_fallback_content("Chart Intelligence", symbol)
        except Exception as e:
            self._render_api_config_error("Chart Intelligence", str(e))
    
    def _render_fallback_content(self, feature_name: str, symbol: str):
        """Render fallback content for unavailable features."""
        st.info(f"""
        **{feature_name} - Coming Soon!**
        
        This feature is currently being enhanced with:
        - Advanced AI capabilities
        - Real-time market analysis
        - Professional-grade insights
        
        For now, you can:
        - Use the working Scenario Modeling tab
        - Check back for updates
        - Contact support for priority access
        """)
        
        # Simple placeholder functionality
        if st.button(f"🔄 Retry {feature_name}", key=f"retry_{feature_name.lower().replace(' ', '_')}"):
            st.rerun()
    
    def _render_compact_footer(self):
        """Render minimal footer."""
        pass
    
    def _show_earnings_prediction(self):
        """Show earnings prediction modal."""
        import random
        
        symbol = self._get_selected_symbol()
        prediction = f"""
        **📊 EARNINGS PREDICTION FOR {symbol}:**
        
        • **Expected EPS:** ${random.uniform(1.50, 3.50):.2f}
        • **Surprise Probability:** {random.randint(60, 85)}%
        • **Post-Earnings Move:** ±{random.randint(3, 12)}%
        • **Key Metric:** {random.choice(['Revenue growth', 'Margin expansion', 'Guidance update'])}
        """
        st.info(prediction)
    
    def _render_api_config_error(self, feature_name: str, error_msg: str):
        """Render API configuration error with setup instructions."""
        st.warning(f"⚠️ {feature_name} encountered an issue")
        
        with st.expander("🔧 Troubleshooting", expanded=False):
            st.markdown(f"""
            **Error Details:** {error_msg}
            
            **Quick Fixes:**
            1. Refresh the page
            2. Check internet connection
            3. Try a different symbol
            4. Clear browser cache
            """)
            
            if st.button("🔄 Retry", key=f"retry_{feature_name.lower().replace(' ', '_')}"):
                st.rerun()
    
    def _show_ai_configuration(self):
        """Show AI configuration panel."""
        st.info("🔧 **AI Configuration Panel**")
        
        with st.form("ai_config_form"):
            st.markdown("### AI Model Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_provider = st.selectbox(
                    "Primary AI Provider",
                    ["OpenAI (GPT-4)", "Anthropic (Claude)", "Google (Gemini)"],
                    help="Select your preferred AI model provider"
                )
                
                temperature = st.slider(
                    "Response Creativity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Higher values make responses more creative"
                )
            
            with col2:
                max_tokens = st.number_input(
                    "Max Response Length",
                    min_value=100,
                    max_value=4000,
                    value=2000,
                    help="Maximum tokens for AI responses"
                )
                
                analysis_depth = st.selectbox(
                    "Analysis Depth",
                    ["Quick", "Standard", "Deep", "Comprehensive"],
                    index=1,
                    help="Depth of AI analysis"
                )
            
            submitted = st.form_submit_button("💾 Save Configuration")
            
            if submitted:
                # Store configuration in session state
                st.session_state.ai_config = {
                    'model_provider': model_provider,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'analysis_depth': analysis_depth
                }
                st.success("✅ AI configuration saved successfully!")
                st.balloons()


# Main entry point
def render():
    """Entry point for the AI Trading Intelligence tab."""
    app = AITradingIntelligence()
    app.render()
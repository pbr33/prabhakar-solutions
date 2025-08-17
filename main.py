import streamlit as st
import sys
import os

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    """
    The main function to run the Streamlit application.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title="AI Trading Agents Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Import components locally to avoid circular imports
    try:
        from langchain_openai import AzureChatOpenAI
        from ui.styles import apply_custom_css
        from ui.sidebar import render_sidebar
        from ui.tabs import market_analysis, portfolio, auto_trading, pro_dashboard, portfolio_enhanced_main
        from ui.tabs import ai_intelligence  # New AI tab
        from core.trading_engine import AutoTradingEngine
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.error("Please ensure all required modules are installed and properly configured.")
        st.stop()

    # Apply custom CSS
    apply_custom_css()

    st.markdown('<h1 class="main-header">🤖 AI Trading Agents Dashboard</h1>', unsafe_allow_html=True)

    # --- Initialize Session State ---
    if 'trading_engine' not in st.session_state:
        st.session_state.trading_engine = AutoTradingEngine()
    if 'trading_bots' not in st.session_state:
        st.session_state.trading_bots = {}
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    # --- Sidebar ---
    try:
        render_sidebar()
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        st.info("Some sidebar features may not be available.")

    # --- LLM Initialization ---
    api_key = st.session_state.get('azure_api_key')
    endpoint = st.session_state.get('azure_endpoint')
    deployment = st.session_state.get('azure_deployment')
    api_version = st.session_state.get('azure_api_version', '2024-02-01')

    if all([api_key, endpoint, deployment, api_version]) and st.session_state.llm is None:
        try:
            st.session_state.llm = AzureChatOpenAI(
                api_key=api_key, 
                api_version=api_version, 
                azure_endpoint=endpoint,
                deployment_name=deployment, 
                temperature=0.7
            )
            if st.session_state.get('debug_mode'):
                st.success("✅ LLM initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize LLM. Please check credentials. Error: {e}")
    elif not all([api_key, endpoint, deployment, api_version]):
         st.warning("Please provide all Azure OpenAI credentials in the sidebar to enable AI-powered features.")

    # --- Main Application Tabs ---
    tab_ai, tab_market, tab_pro, tab_portfolio, tab_autotrade = st.tabs([
        "🤖 AI Intelligence",      # New AI tab - placed first for prominence
        "📊 Market Analysis",
        "📈 Pro Dashboard",
        "💼 Portfolio View", 
        "🚀 Auto-Trading"
    ])

    with tab_ai:
        try:
            ai_intelligence.render()
        except Exception as e:
            st.error(f"AI Intelligence error: {e}")
            if st.session_state.get('debug_mode'):
                st.exception(e)
            st.info("AI Intelligence features require proper API configuration.")

    with tab_market:
        try:
            market_analysis.render()
        except Exception as e:
            st.error(f"Market Analysis error: {e}")
            if st.session_state.get('debug_mode'):
                st.exception(e)

    with tab_pro:
        try:
            pro_dashboard.render()
        except Exception as e:
            st.error(f"Pro Dashboard error: {e}")
            if st.session_state.get('debug_mode'):
                st.exception(e)

    with tab_portfolio:
        try:
            portfolio_enhanced_main.render()
        except Exception as e:
            st.error(f"Portfolio error: {e}")
            if st.session_state.get('debug_mode'):
                st.exception(e)

    with tab_autotrade:
        try:
            auto_trading.render()
        except Exception as e:
            st.error(f"Auto-Trading error: {e}")
            if st.session_state.get('debug_mode'):
                st.exception(e)

if __name__ == "__main__":
    main()
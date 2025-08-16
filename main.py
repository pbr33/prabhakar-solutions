import streamlit as st
from langchain_openai import AzureChatOpenAI

# Import UI components
from ui.styles import apply_custom_css
from ui.sidebar import render_sidebar
from ui.tabs import market_analysis, portfolio, auto_trading, pro_dashboard

# Import core components
from core.trading_engine import AutoTradingEngine
from core.trading_bot import TradingBot

def main():
    """
    The main function to run the Streamlit application.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title="AI Trading Agents Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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
    render_sidebar()

    # --- LLM Initialization ---
    api_key = st.session_state.get('azure_api_key')
    endpoint = st.session_state.get('azure_endpoint')
    deployment = st.session_state.get('azure_deployment')
    api_version = st.session_state.get('azure_api_version')

    if all([api_key, endpoint, deployment, api_version]) and st.session_state.llm is None:
        try:
            st.session_state.llm = AzureChatOpenAI(
                api_key=api_key, api_version=api_version, azure_endpoint=endpoint,
                deployment_name=deployment, temperature=0.7
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM. Please check credentials. Error: {e}")
    elif not all([api_key, endpoint, deployment, api_version]):
         st.warning("Please provide all Azure OpenAI credentials in the sidebar to enable AI-powered features.")


    # --- Main Application Tabs ---
    tab_market, tab_pro, tab_portfolio, tab_autotrade = st.tabs([
        "📊 Market Analysis",
        " dashboards and Alerts",
        "💼 HF & PE Portfolio View",
        "🚀 Auto-Trading"
    ])

    with tab_market:
        market_analysis.render()

    with tab_pro:
        pro_dashboard.render()

    with tab_portfolio:
        portfolio.render()

    with tab_autotrade:
        auto_trading.render()


if __name__ == "__main__":
    main()

# Method 1: Update your main.py to include a landing page

import streamlit as st
import sys
import os

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def render_landing_page():
    """Render the beautiful landing page"""
    st.markdown("""
    <style>
        .stApp > header {
            background-color: transparent;
        }
        
        .main > div {
            padding-top: 0rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Your beautiful HTML landing page
    landing_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agent RICH - Real-time Investment Capital Hub</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #fff;
                overflow-x: hidden;
            }

            .header {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                padding: 1rem 2rem;
            }

            .nav-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1400px;
                margin: 0 auto;
            }

            .logo-section {
                display: flex;
                align-items: center;
                gap: 1rem;
            }

            .logo {
                width: 50px;
                height: 50px;
                background: #4338ca;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                color: white;
                font-size: 18px;
                box-shadow: 0 8px 32px rgba(67, 56, 202, 0.3);
            }

            .brand-text {
                font-size: 24px;
                font-weight: bold;
                background: linear-gradient(45deg, #fff, #e0e7ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .hero {
                min-height: 80vh;
                display: flex;
                align-items: center;
                padding: 2rem;
                position: relative;
            }

            .hero-container {
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 4rem;
                align-items: center;
            }

            .hero-title {
                font-size: clamp(2.5rem, 5vw, 4rem);
                font-weight: 900;
                line-height: 1.2;
                margin-bottom: 1rem;
                background: linear-gradient(45deg, #fff, #e0e7ff, #c7d2fe);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .hero-subtitle {
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: #e0e7ff;
            }

            .hero-description {
                font-size: 1.2rem;
                line-height: 1.6;
                margin-bottom: 2rem;
                color: rgba(255, 255, 255, 0.9);
            }

            .cta-button {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: linear-gradient(45deg, #10b981, #059669);
                color: white;
                padding: 1rem 2rem;
                border: none;
                border-radius: 12px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
            }

            .cta-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 40px rgba(16, 185, 129, 0.4);
            }

            .hero-avatar {
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }

            .avatar-container {
                width: 250px;
                height: 250px;
                border-radius: 50%;
                background: linear-gradient(45deg, #4338ca, #7c3aed);
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 2rem;
                position: relative;
                box-shadow: 0 20px 60px rgba(67, 56, 202, 0.4);
                animation: pulse 3s ease-in-out infinite;
            }

            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }

            .avatar {
                width: 200px;
                height: 200px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 4rem;
                color: white;
            }

            .avatar-name {
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                color: #e0e7ff;
            }

            @media (max-width: 768px) {
                .hero-container {
                    grid-template-columns: 1fr;
                    text-align: center;
                }
            }
        </style>
    </head>
    <body>
        <header class="header">
            <nav class="nav-container">
                <div class="logo-section">
                    <div class="logo">ECI</div>
                    <div class="brand-text">Agent RICH</div>
                </div>
            </nav>
        </header>

        <section class="hero">
            <div class="hero-container">
                <div class="hero-content">
                    <h1 class="hero-title">Agent RICH</h1>
                    <h2 class="hero-subtitle">Real-time Investment Capital Hub</h2>
                    <p class="hero-description">
                        Harness the power of AI-driven trading agents to maximize your investment potential. 
                        Our cutting-edge platform combines real-time market analysis, intelligent automation, 
                        and comprehensive portfolio management to help you achieve financial success.
                    </p>
                    <button class="cta-button" onclick="launchDashboard()">
                        <i class="fas fa-rocket"></i>
                        Launch Dashboard
                    </button>
                </div>
                <div class="hero-avatar">
                    <div class="avatar-container">
                        <div class="avatar">
                            <i class="fas fa-user-tie"></i>
                        </div>
                    </div>
                    <div class="avatar-name">AI Agent RICH</div>
                </div>
            </div>
        </section>

        <script>
            function launchDashboard() {
                // This will trigger the Streamlit page change
                window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'dashboard'}, '*');
            }
        </script>
    </body>
    </html>
    """
    
    # Display the landing page
    st.components.v1.html(landing_html, height=800, scrolling=True)
    
    # Listen for the dashboard launch
    if st.button("🚀 Launch Dashboard", key="dashboard_btn", help="Enter the AI Trading Platform"):
        st.session_state.page = 'dashboard'
        st.rerun()

def main():
    """
    The main function to run the Streamlit application.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Agent RICH - Real-time Investment Capital Hub",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed" if st.session_state.get('page') == 'landing' else "expanded"
    )

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'

    # Route to different pages
    if st.session_state.page == 'landing':
        render_landing_page()
        return
    
    # If not landing page, continue with your existing app
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

    # Add a back to landing button
    if st.sidebar.button("🏠 Back to Landing"):
        st.session_state.page = 'landing'
        st.rerun()

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
        "🤖 AI Intelligence",
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
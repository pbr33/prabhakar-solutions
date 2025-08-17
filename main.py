# Complete working main.py with landing page and login

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
    
    # Load the agent image from static folder
    avatar_img_base64 = ""
    import base64
    
    # Try multiple possible filenames and extensions
    possible_names = [
        "agent_image.png", "agent_image.jpg", "agent_image.jpeg",
        "agent.png", "agent.jpg", "agent.jpeg",
        "rich.png", "rich.jpg", "rich.jpeg",
        "avatar.png", "avatar.jpg", "avatar.jpeg"
    ]
    
    avatar_loaded = False
    for filename in possible_names:
        try:
            avatar_path = os.path.join(current_dir, "static", filename)
            if os.path.exists(avatar_path):
                with open(avatar_path, "rb") as img_file:
                    avatar_img_base64 = base64.b64encode(img_file.read()).decode()
                    avatar_loaded = True
                    st.success(f"✅ Avatar image loaded: {filename}")
                    break
        except Exception as e:
            continue
    
    if not avatar_loaded:
        st.warning("⚠️ Avatar image not found. Please check:")
        st.info("📁 Make sure your image is in the 'static' folder")
        st.info("📝 Supported names: agent_image.png, agent.png, rich.png, avatar.png")
        st.info("🖼️ Supported formats: .png, .jpg, .jpeg")
        
        # Show current directory and static folder contents
        static_path = os.path.join(current_dir, "static")
        if os.path.exists(static_path):
            files = os.listdir(static_path)
            st.info(f"📂 Files in static folder: {files}")
        else:
            st.error("❌ 'static' folder not found. Please create it and add your image.")

    # Prepare avatar HTML
    if avatar_img_base64:
        avatar_html = f'<img src="data:image/png;base64,{avatar_img_base64}" class="avatar" alt="AI Agent RICH">'
    else:
        avatar_html = '<div class="avatar-icon"><i class="fas fa-user-tie"></i></div>'
    
    # Beautiful HTML landing page
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
                object-fit: cover;
                border: 4px solid rgba(255, 255, 255, 0.2);
            }

            .avatar-icon {
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 4rem;
                color: white;
                width: 200px;
                height: 200px;
                border-radius: 50%;
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
                        """ + avatar_html + """
                    </div>
                    <div class="avatar-name">AI Agent RICH</div>
                </div>
            </div>
        </section>

        <script>
            function launchDashboard() {
                // Create a hidden form and submit it to trigger Streamlit page change
                window.location.href = window.location.href + '?login=true';
            }
        </script>
    </body>
    </html>
    """
    
    # Check URL parameters for login trigger
    query_params = st.experimental_get_query_params()
    if 'login' in query_params:
        st.session_state.page = 'login'
        st.experimental_set_query_params()  # Clear the parameter
        st.rerun()
    
    # Display the landing page
    st.components.v1.html(landing_html, height=800, scrolling=True)

def render_login_page():
    """Render the login page"""
    st.markdown("""
    <style>
        .stApp > header {
            background-color: transparent;
        }
        
        .main > div {
            padding-top: 2rem;
        }
        
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        }
        
        .login-title {
            text-align: center;
            color: white;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            color: white;
            padding: 0.75rem;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        
        .stButton > button {
            width: 100%;
            background: linear-gradient(45deg, #10b981, #059669);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem;
            font-weight: 600;
            font-size: 1rem;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #059669, #047857);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4);
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">🔐 Agent RICH Login</h1>', unsafe_allow_html=True)
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        login_button = st.form_submit_button("🚀 Enter Dashboard")
        
        if login_button:
            if username == "genaiwithprabhakar" and password == "genaiwithprabhakar":
                st.session_state.authenticated = True
                st.session_state.page = 'dashboard'
                st.success("✅ Login successful! Redirecting to dashboard...")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")
                st.info("💡 Hint: Both username and password are the same")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Back to landing button
    if st.button("⬅️ Back to Landing"):
        st.session_state.page = 'landing'
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
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Route to different pages
    if st.session_state.page == 'landing':
        render_landing_page()
        return
    elif st.session_state.page == 'login':
        render_login_page()
        return
    elif st.session_state.page == 'dashboard' and not st.session_state.authenticated:
        # Redirect to login if trying to access dashboard without authentication
        st.session_state.page = 'login'
        st.rerun()
    
    # If not landing or login page, continue with your existing app
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

    # Add navigation buttons in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🏠 Back to Landing"):
        st.session_state.page = 'landing'
        st.rerun()
    
    if st.sidebar.button("🔒 Logout"):
        st.session_state.authenticated = False
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
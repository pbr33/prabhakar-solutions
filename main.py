import streamlit as st
import sys
import os
import time
import pandas as pd
import plotly.express as px

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def render_landing_page():
    """Render the enhanced landing page with all capabilities showcase"""
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
    
    # Load the agent image from static folder (NO DEBUG MESSAGES)
    avatar_img_base64 = ""
    import base64
    
    # Try multiple possible filenames and extensions (silent)
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
                    break
        except Exception:
            continue

    # Prepare avatar HTML
    if avatar_img_base64:
        avatar_html = f'<img src="data:image/png;base64,{avatar_img_base64}" class="avatar" alt="AI Agent RICH">'
    else:
        avatar_html = '<div class="avatar-icon"><i class="fas fa-robot"></i></div>'
    
    # Enhanced HTML landing page with capabilities showcase
    landing_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agent RICH - Real-time Investment Capital Hub</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #fff;
                overflow-x: hidden;
                scroll-behavior: smooth;
            }}

            .header {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                padding: 1rem 2rem;
                position: sticky;
                top: 0;
                z-index: 1000;
            }}

            .nav-container {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                max-width: 1400px;
                margin: 0 auto;
            }}

            .logo-section {{
                display: flex;
                align-items: center;
                gap: 1rem;
            }}

            .logo {{
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
            }}

            .brand-text {{
                font-size: 24px;
                font-weight: bold;
                background: linear-gradient(45deg, #fff, #e0e7ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            .hero {{
                min-height: 90vh;
                display: flex;
                align-items: center;
                padding: 2rem;
                position: relative;
            }}

            .hero-container {{
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 4rem;
                align-items: center;
            }}

            .hero-title {{
                font-size: clamp(2.5rem, 5vw, 4.5rem);
                font-weight: 900;
                line-height: 1.2;
                margin-bottom: 1rem;
                background: linear-gradient(45deg, #fff, #e0e7ff, #c7d2fe);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            .hero-subtitle {{
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: #e0e7ff;
            }}

            .hero-description {{
                font-size: 1.2rem;
                line-height: 1.6;
                margin-bottom: 2rem;
                color: rgba(255, 255, 255, 0.9);
            }}

            .cta-button {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: linear-gradient(45deg, #10b981, #059669);
                color: white;
                padding: 1.2rem 2.5rem;
                border: none;
                border-radius: 12px;
                font-size: 1.2rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                box-shadow: 0 8px 32px rgba(16, 185, 129, 0.4);
                position: relative;
                overflow: hidden;
            }}

            .cta-button:hover {{
                transform: translateY(-3px);
                box-shadow: 0 15px 45px rgba(16, 185, 129, 0.5);
            }}

            .cta-button::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }}

            .cta-button:hover::before {{
                left: 100%;
            }}

            .hero-avatar {{
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }}

            .avatar-container {{
                width: 280px;
                height: 280px;
                border-radius: 50%;
                background: linear-gradient(45deg, #4338ca, #7c3aed);
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 2rem;
                position: relative;
                box-shadow: 0 20px 60px rgba(67, 56, 202, 0.4);
                animation: pulse 3s ease-in-out infinite;
            }}

            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
            }}

            .avatar {{
                width: 220px;
                height: 220px;
                border-radius: 50%;
                object-fit: cover;
                border: 4px solid rgba(255, 255, 255, 0.2);
            }}

            .avatar-icon {{
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 5rem;
                color: white;
                width: 220px;
                height: 220px;
                border-radius: 50%;
            }}

            .avatar-name {{
                font-size: 1.8rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                color: #e0e7ff;
            }}

            .capabilities-section {{
                padding: 5rem 2rem;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
            }}

            .capabilities-container {{
                max-width: 1400px;
                margin: 0 auto;
            }}

            .section-title {{
                text-align: center;
                font-size: 3rem;
                font-weight: 800;
                margin-bottom: 1rem;
                background: linear-gradient(45deg, #fff, #e0e7ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            .section-subtitle {{
                text-align: center;
                font-size: 1.2rem;
                color: rgba(255, 255, 255, 0.8);
                margin-bottom: 4rem;
            }}

            .capabilities-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin-bottom: 4rem;
            }}

            .capability-card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 2rem;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
            }}

            .capability-card:hover {{
                transform: translateY(-10px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                background: rgba(255, 255, 255, 0.15);
            }}

            .capability-icon {{
                font-size: 3rem;
                margin-bottom: 1rem;
                display: block;
            }}

            .capability-title {{
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 1rem;
                color: #fff;
            }}

            .capability-description {{
                color: rgba(255, 255, 255, 0.8);
                line-height: 1.6;
            }}

            .stats-section {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 2rem;
                margin: 4rem 0;
            }}

            .stat-card {{
                text-align: center;
                padding: 2rem;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }}

            .stat-number {{
                font-size: 3rem;
                font-weight: 800;
                color: #10b981;
                display: block;
            }}

            .stat-label {{
                color: rgba(255, 255, 255, 0.8);
                font-size: 1.1rem;
                margin-top: 0.5rem;
            }}

            .demo-video {{
                text-align: center;
                margin: 4rem 0;
            }}

            .floating-elements {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                overflow: hidden;
                pointer-events: none;
            }}

            .floating-element {{
                position: absolute;
                color: rgba(255, 255, 255, 0.1);
                animation: float 6s ease-in-out infinite;
            }}

            @keyframes float {{
                0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
                50% {{ transform: translateY(-20px) rotate(10deg); }}
            }}

            @media (max-width: 768px) {{
                .hero-container {{
                    grid-template-columns: 1fr;
                    text-align: center;
                }}
                
                .capabilities-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .stats-section {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}

            .scroll-indicator {{
                position: absolute;
                bottom: 2rem;
                left: 50%;
                transform: translateX(-50%);
                animation: bounce 2s infinite;
            }}

            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translateX(-50%) translateY(0); }}
                40% {{ transform: translateX(-50%) translateY(-10px); }}
                60% {{ transform: translateX(-50%) translateY(-5px); }}
            }}
        </style>
    </head>
    <body>
        <div class="floating-elements">
            <i class="fas fa-chart-line floating-element" style="top: 10%; left: 10%; font-size: 2rem; animation-delay: 0s;"></i>
            <i class="fas fa-robot floating-element" style="top: 20%; right: 15%; font-size: 1.5rem; animation-delay: 1s;"></i>
            <i class="fas fa-brain floating-element" style="bottom: 30%; left: 20%; font-size: 2.5rem; animation-delay: 2s;"></i>
            <i class="fas fa-coins floating-element" style="top: 60%; right: 25%; font-size: 1.8rem; animation-delay: 3s;"></i>
            <i class="fas fa-rocket floating-element" style="bottom: 20%; right: 10%; font-size: 2rem; animation-delay: 4s;"></i>
        </div>

        <header class="header">
            <nav class="nav-container">
                <div class="logo-section">
                    <div class="logo">AR</div>
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
                        Revolutionary AI-driven trading platform that combines multi-agent intelligence, 
                        real-time market analysis, and institutional-grade portfolio management. 
                        Experience the future of investment management with our cutting-edge AI agents.
                    </p>
                    <button class="cta-button" id="launchBtn">
                        <i class="fas fa-rocket"></i>
                        Launch AI Dashboard
                    </button>
                </div>
                <div class="hero-avatar">
                    <div class="avatar-container">
                        {avatar_html}
                    </div>
                    <div class="avatar-name">AI Agent RICH</div>
                </div>
            </div>
            <div class="scroll-indicator">
                <i class="fas fa-chevron-down" style="color: rgba(255,255,255,0.6); font-size: 1.5rem;"></i>
            </div>
        </section>

        <section class="capabilities-section">
            <div class="capabilities-container">
                <h2 class="section-title">Platform Capabilities</h2>
                <p class="section-subtitle">Discover the power of AI-driven investment management</p>
                
                <div class="capabilities-grid">
                    <div class="capability-card">
                        <span class="capability-icon">🤖</span>
                        <h3 class="capability-title">Multi-Agent AI System</h3>
                        <p class="capability-description">
                            Deploy specialized AI agents - Technical Analyst, Macro Economist, 
                            Sentiment Analyst, and Quant Researcher working simultaneously 
                            to provide comprehensive market intelligence.
                        </p>
                    </div>

                    <div class="capability-card">
                        <span class="capability-icon">📊</span>
                        <h3 class="capability-title">Advanced Market Analysis</h3>
                        <p class="capability-description">
                            Real-time pattern detection, predictive analytics, anomaly detection, 
                            and ARIMA forecasting with intelligent chart annotations and 
                            technical indicator dashboards.
                        </p>
                    </div>

                    <div class="capability-card">
                        <span class="capability-icon">🚀</span>
                        <h3 class="capability-title">Automated Trading Agents</h3>
                        <p class="capability-description">
                            Deploy autonomous trading bots with customizable strategies, 
                            real-time execution, risk management, and comprehensive 
                            portfolio tracking with performance analytics.
                        </p>
                    </div>

                    <div class="capability-card">
                        <span class="capability-icon">💼</span>
                        <h3 class="capability-title">Institutional Portfolio Management</h3>
                        <p class="capability-description">
                            Comprehensive hedge fund and private equity portfolio analytics, 
                            risk assessment, sector exposure analysis, and performance 
                            attribution with institutional-grade reporting.
                        </p>
                    </div>

                    <div class="capability-card">
                        <span class="capability-icon">📄</span>
                        <h3 class="capability-title">Document Intelligence Hub</h3>
                        <p class="capability-description">
                            AI-powered document analysis supporting multiple file formats, 
                            bulk processing, data source integrations (SharePoint, Drive, Box), 
                            and intelligent content extraction.
                        </p>
                    </div>

                    <div class="capability-card">
                        <span class="capability-icon">📰</span>
                        <h3 class="capability-title">AI Report Generator</h3>
                        <p class="capability-description">
                            Automated generation of investment memos, board packs, 
                            quarterly reports, compliance documents, and market 
                            sentiment summaries with customizable templates.
                        </p>
                    </div>

                    <div class="capability-card">
                        <span class="capability-icon">🎭</span>
                        <h3 class="capability-title">Scenario Modeling</h3>
                        <p class="capability-description">
                            Monte Carlo simulations, probability-weighted scenarios, 
                            stress testing, and risk modeling with interactive 
                            visualizations and comprehensive scenario analysis.
                        </p>
                    </div>

                    <div class="capability-card">
                        <span class="capability-icon">🎤</span>
                        <h3 class="capability-title">Voice Trading Assistant</h3>
                        <p class="capability-description">
                            Natural language trading queries, voice commands, 
                            intelligent Q&A system, and conversational AI for 
                            complex trading decisions and market insights.
                        </p>
                    </div>
                </div>

                <div class="stats-section">
                    <div class="stat-card">
                        <span class="stat-number">94.2%</span>
                        <span class="stat-label">AI Accuracy Rate</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">< 3s</span>
                        <span class="stat-label">Analysis Speed</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">24/7</span>
                        <span class="stat-label">Market Monitoring</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">∞</span>
                        <span class="stat-label">Scalability</span>
                    </div>
                </div>

                <div class="demo-video">
                    <h3 style="margin-bottom: 2rem; font-size: 2rem;">See Agent RICH in Action</h3>
                    <div class="video-container" style="position: relative; max-width: 800px; margin: 0 auto; border-radius: 20px; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.3);">
                        <div style="width: 100%; height: 400px; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 20px; display: flex; align-items: center; justify-content: center;">
                            <div style="background: rgba(0,0,0,0.7); padding: 2rem; border-radius: 15px; backdrop-filter: blur(10px); text-align: center;">
                                <h3 style="color: white; margin-bottom: 1rem; font-size: 1.5rem;">🎬 Demo Features</h3>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; text-align: left; color: rgba(255,255,255,0.9); font-size: 0.9rem;">
                                    <div>
                                        <div>🤖 AI Multi-Agent Analysis</div>
                                        <div>📊 Real-time Pattern Detection</div>
                                        <div>🚀 Auto-Trading Bots</div>
                                        <div>💼 Portfolio Analytics</div>
                                    </div>
                                    <div>
                                        <div>📄 Document Intelligence</div>
                                        <div>🎭 Scenario Modeling</div>
                                        <div>🎤 Voice Assistant</div>
                                        <div>📰 AI Report Generation</div>
                                    </div>
                                </div>
                                <div style="margin-top: 1rem; padding: 0.5rem 1rem; background: rgba(16, 185, 129, 0.8); border-radius: 8px; color: white; font-weight: 600;">
                                    ⚡ All Features Running Live
                                </div>
                            </div>
                        </div>
                    </div>
                    <p style="margin-top: 1.5rem; color: rgba(255,255,255,0.8); font-size: 1.1rem;">
                        Experience the power of AI-driven investment management with real-time market analysis, 
                        automated trading agents, and institutional-grade portfolio analytics.
                    </p>
                </div>
            </div>
        </section>

        <script>
            // Add event listener when DOM is loaded
            document.addEventListener('DOMContentLoaded', function() {{
                const launchBtn = document.getElementById('launchBtn');
                if (launchBtn) {{
                    launchBtn.addEventListener('click', function() {{
                        // Create a custom event that Streamlit can listen to
                        const event = new CustomEvent('launchDashboard');
                        window.dispatchEvent(event);
                        
                        // Visual feedback
                        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Launching...';
                        this.style.background = 'linear-gradient(45deg, #7c3aed, #6366f1)';
                        
                        // Fallback - show message if Streamlit doesn't respond
                        setTimeout(() => {{
                            alert('🚀 Redirecting to login page! Click OK to continue.');
                        }}, 500);
                    }});
                }}
                
                // Add floating animation delays
                const floatingElements = document.querySelectorAll('.floating-element');
                floatingElements.forEach((element, index) => {{
                    element.style.animationDelay = index * 0.5 + 's';
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Display the landing page
    st.components.v1.html(landing_html, height=2000, scrolling=True)
    
    # FIXED: Use Streamlit button instead of JavaScript for navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 **Launch AI Dashboard**", 
                    key="main_cta", 
                    use_container_width=True,
                    type="primary"):
            st.session_state.page = 'login'
            st.rerun()

def render_login_page():
    """Render a clean and professional login page with avatar"""
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
            padding: 2.5rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .avatar-section {
            margin-bottom: 2rem;
        }
        
        .login-avatar {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: linear-gradient(45deg, #4338ca, #7c3aed);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem;
            position: relative;
            box-shadow: 0 15px 35px rgba(67, 56, 202, 0.4);
            animation: glow 3s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            0% { box-shadow: 0 15px 35px rgba(67, 56, 202, 0.4); }
            100% { box-shadow: 0 15px 35px rgba(124, 58, 237, 0.6); }
        }
        
        .login-avatar img {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            border: 3px solid rgba(255, 255, 255, 0.2);
        }
        
        .avatar-icon {
            font-size: 3.5rem;
            color: white;
        }
        
        .login-title {
            color: white;
            font-size: 2.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, #fff, #e0e7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .login-subtitle {
            color: rgba(255, 255, 255, 0.8);
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        
        .form-section {
            margin: 2rem 0;
        }
        
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            color: white;
            padding: 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #10b981;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.3);
            background: rgba(255, 255, 255, 0.15);
        }
        
        .stButton > button {
            width: 100%;
            background: linear-gradient(45deg, #10b981, #059669);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            margin: 0.5rem 0;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #059669, #047857);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        }
        
        .guest-button {
            background: linear-gradient(45deg, #6366f1, #4f46e5) !important;
        }
        
        .guest-button:hover {
            background: linear-gradient(45deg, #4f46e5, #4338ca) !important;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
        }
        
        .back-button {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            margin-top: 1rem;
        }
        
        .back-button:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 5px 15px rgba(255, 255, 255, 0.1) !important;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .success-message {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.5);
            border-radius: 12px;
            padding: 1rem;
            color: #10b981;
            text-align: center;
            margin: 1rem 0;
            font-weight: 600;
        }
        
        .hint-section {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .hint-title {
            color: #10b981;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .hint-text {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.8rem;
            line-height: 1.4;
        }
        
        .features-preview {
            margin-top: 2rem;
            text-align: center;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .feature-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 0.75rem;
            border-radius: 8px;
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.8);
        }
        
        .feature-icon {
            font-size: 1.2rem;
            margin-bottom: 0.25rem;
            display: block;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load avatar image
    avatar_img_base64 = ""
    import base64
    
    possible_names = [
        "agent_image.png", "agent_image.jpg", "agent_image.jpeg",
        "agent.png", "agent.jpg", "agent.jpeg",
        "rich.png", "rich.jpg", "rich.jpeg",
        "avatar.png", "avatar.jpg", "avatar.jpeg"
    ]
    
    for filename in possible_names:
        try:
            avatar_path = os.path.join(current_dir, "static", filename)
            if os.path.exists(avatar_path):
                with open(avatar_path, "rb") as img_file:
                    avatar_img_base64 = base64.b64encode(img_file.read()).decode()
                    break
        except Exception:
            continue
    
    # Prepare avatar HTML
    if avatar_img_base64:
        avatar_html = f'<img src="data:image/png;base64,{avatar_img_base64}" alt="AI Agent RICH">'
    else:
        avatar_html = '<i class="fas fa-robot avatar-icon"></i>'
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Avatar and title section
    st.markdown(f"""
    <div class="avatar-section">
        <div class="login-avatar">
            {avatar_html}
        </div>
        <h1 class="login-title">Welcome Back</h1>
        <p class="login-subtitle">Sign in to Agent RICH AI Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form
    with st.form("login_form"):
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")
        password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.form_submit_button("🚀 Sign In")
        with col2:
            guest_mode = st.form_submit_button("👀 Guest Mode")
        
        if login_button:
            if username == "genaiwithprabhakar" and password == "genaiwithprabhakar":
                st.session_state.authenticated = True
                st.session_state.user_type = "full_access"
                st.session_state.page = 'dashboard'
                st.session_state.show_hint = False
                st.markdown('<div class="success-message">✅ Welcome! Redirecting to your dashboard...</div>', unsafe_allow_html=True)
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.login_attempts = st.session_state.get('login_attempts', 0) + 1
                st.session_state.show_hint = True
                st.error("❌ Invalid credentials. Please check your username and password.")
                
                # Show demo credentials after 2 failed attempts
                if st.session_state.login_attempts >= 2:
                    show_demo_credentials()
                
        if guest_mode:
            st.session_state.authenticated = True
            st.session_state.user_type = "guest"
            st.session_state.page = 'dashboard'
            st.markdown('<div class="success-message">👀 Entering Guest Mode... Limited features available.</div>', unsafe_allow_html=True)
            st.info("💡 Guest mode provides read-only access to explore the platform.")
            time.sleep(1)
            st.rerun()
    
    # Hint section (only visible on error or first visit)
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    
    # Show hint after failed attempt
    if st.session_state.get('show_hint', False):
        st.markdown("""
        <div class="hint-section">
            <div class="hint-title">💡 Need Help?</div>
            <div class="hint-text">
                Demo credentials are available for testing.<br>
                Contact admin for access or try Guest Mode to explore features.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Back to landing button
    if st.button("⬅️ Back to Landing", key="back_to_landing"):
        st.session_state.page = 'landing'
        st.rerun()
    
    # Features preview
    st.markdown("""
    <div class="features-preview">
        <h4 style="color: rgba(255,255,255,0.9); margin-bottom: 1rem;">🌟 Platform Features</h4>
        <div class="features-grid">
            <div class="feature-item">
                <span class="feature-icon">🤖</span>
                <div>AI Trading Agents</div>
            </div>
            <div class="feature-item">
                <span class="feature-icon">📊</span>
                <div>Market Analysis</div>
            </div>
            <div class="feature-item">
                <span class="feature-icon">💼</span>
                <div>Portfolio Management</div>
            </div>
            <div class="feature-item">
                <span class="feature-icon">🚀</span>
                <div>Automated Trading</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_demo_portfolio():
    """Render a demo portfolio for guest users"""
    st.subheader("📊 Demo Portfolio Overview")
    
    # Sample portfolio data
    portfolio_data = {
        'Stock': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        'Shares': [50, 25, 75, 30, 40],
        'Price': [150.25, 2500.00, 300.50, 800.00, 3200.00],
        'Value': [7512.50, 62500.00, 22537.50, 24000.00, 128000.00],
        'Change': [2.5, -1.2, 1.8, -3.5, 0.8]
    }
    
    df = pd.DataFrame(portfolio_data)
    df['Total Value'] = df['Shares'] * df['Price']
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Portfolio Value",
            value=f"${df['Total Value'].sum():,.2f}",
            delta="$2,450.00"
        )
    
    with col2:
        st.metric(
            label="Day Change",
            value="$1,234.56",
            delta="0.52%"
        )
    
    with col3:
        st.metric(
            label="Total Positions",
            value=len(df),
            delta=None
        )
    
    with col4:
        st.metric(
            label="Demo Account",
            value="Active",
            delta="Guest Mode"
        )
    
    # Portfolio composition chart
    fig = px.pie(
        df, 
        values='Total Value', 
        names='Stock',
        title="Portfolio Composition",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Holdings table
    st.subheader("📋 Current Holdings")
    st.dataframe(
        df[['Stock', 'Shares', 'Price', 'Total Value', 'Change']],
        use_container_width=True
    )
    
    # Upgrade prompt
    st.markdown("""
    <div style="background: linear-gradient(45deg, #667eea, #764ba2); 
                padding: 1.5rem; border-radius: 12px; margin: 2rem 0; text-align: center;">
        <h3 style="color: white; margin-bottom: 1rem;">🚀 Unlock Full Features</h3>
        <p style="color: rgba(255,255,255,0.9); margin-bottom: 1rem;">
            Get access to AI-powered trading, real-time analysis, and automated bots!
        </p>
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <strong style="color: #10b981;">Demo Credentials:</strong><br>
            <span style="color: white;">Username: genaiwithprabhakar</span><br>
            <span style="color: white;">Password: genaiwithprabhakar</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """
    The main function to run the Streamlit application with enhanced authentication.
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
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None

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
        st.info("Some modules may not be available. Creating fallback components...")
        
        # Create fallback functions if imports fail
        def apply_custom_css():
            st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    text-align: center;
                    font-size: 3rem;
                    font-weight: bold;
                    margin-bottom: 2rem;
                }
                .stApp {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
            </style>
            """, unsafe_allow_html=True)
        
        def render_sidebar():
            st.sidebar.header("⚙️ Configuration")
            st.sidebar.info("Some features require additional modules to be installed.")
        
        # Create mock modules for fallback
        class MockModule:
            def render(self):
                st.info("This feature requires additional dependencies. Please check your installation.")
                st.markdown("""
                ### 📊 Demo Feature Available
                This is a placeholder for the actual module. The real implementation would include:
                - Real-time market data
                - AI-powered analysis
                - Interactive charts and visualizations
                - Trading capabilities
                """)
        
        market_analysis = MockModule()
        portfolio_enhanced_main = MockModule()
        auto_trading = MockModule()
        pro_dashboard = MockModule()
        ai_intelligence = MockModule()
        
        class AutoTradingEngine:
            pass

    # Apply custom CSS
    apply_custom_css()

    # Add user type indicator and navigation in sidebar
    st.sidebar.markdown("---")
    
    # Display user status
    user_type = st.session_state.get('user_type', 'guest')
    if user_type == 'full_access':
        st.sidebar.success("✅ Full Access Mode")
        st.sidebar.markdown("**User:** genaiwithprabhakar")
    else:
        st.sidebar.info("👀 Guest Mode - Limited Features")
        st.sidebar.markdown("*Use demo credentials for full access*")
    
    # Navigation buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🏠 Landing", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()
    
    with col2:
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_type = None
            st.session_state.page = 'landing'
            st.rerun()

    # Main header with user status
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<h1 class="main-header">🤖 AI Trading Agents Dashboard</h1>', unsafe_allow_html=True)
    with header_col2:
        if user_type == 'guest':
            st.markdown("""
            <div style="background: rgba(255, 165, 0, 0.1); border: 1px solid rgba(255, 165, 0, 0.3); 
                        border-radius: 8px; padding: 0.5rem; text-align: center; margin-top: 1rem;">
                <small style="color: #ff8c00;">🔓 Guest Mode</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); 
                        border-radius: 8px; padding: 0.5rem; text-align: center; margin-top: 1rem;">
                <small style="color: #10b981;">🔐 Full Access</small>
            </div>
            """, unsafe_allow_html=True)

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
        st.sidebar.error(f"Sidebar error: {e}")
        st.sidebar.info("Some sidebar features may not be available.")

    # --- LLM Initialization ---
    api_key = st.session_state.get('azure_api_key')
    endpoint = st.session_state.get('azure_endpoint')
    deployment = st.session_state.get('azure_deployment')
    api_version = st.session_state.get('azure_api_version', '2024-02-01')

    # For guest mode, show limited functionality message
    if user_type == 'guest':
        st.info("🔔 **Guest Mode:** Some features are limited. Use demo credentials for full AI-powered functionality.")

    if all([api_key, endpoint, deployment, api_version]) and st.session_state.llm is None:
        try:
            from langchain_openai import AzureChatOpenAI
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
    elif not all([api_key, endpoint, deployment, api_version]) and user_type == 'full_access':
         st.warning("Please provide all Azure OpenAI credentials in the sidebar to enable AI-powered features.")

    # --- Main Application Tabs ---
    # Adjust tab availability based on user type
    if user_type == 'full_access':
        tab_ai, tab_market, tab_pro, tab_portfolio, tab_autotrade = st.tabs([
            "🤖 AI Intelligence",
            "📊 Market Analysis",
            "📈 Pro Dashboard",
            "💼 Portfolio View", 
            "🚀 Auto-Trading"
        ])
    else:
        # Limited tabs for guest users
        tab_market, tab_portfolio = st.tabs([
            "📊 Market Analysis (Demo)",
            "💼 Portfolio View (Demo)"
        ])

    # Render tabs based on user access
    if user_type == 'full_access':
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
    
    else:  # Guest mode
        with tab_market:
            st.info("📊 **Demo Mode:** Market Analysis with limited features")
            try:
                # Render market analysis with limited functionality
                market_analysis.render()
                st.warning("🔒 Some advanced features require full access. Please login with demo credentials.")
            except Exception as e:
                st.error(f"Market Analysis error: {e}")

        with tab_portfolio:
            st.info("💼 **Demo Mode:** Portfolio View with sample data")
            try:
                # Show demo portfolio data
                render_demo_portfolio()
            except Exception as e:
                st.error(f"Portfolio demo error: {e}")

if __name__ == "__main__":
    main()
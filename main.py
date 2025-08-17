import streamlit as st
import sys
import os

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
        <title>Agent RICH - AI-Powered Investment Capital Hub</title>
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

            .video-placeholder {{
                width: 100%;
                max-width: 800px;
                height: 400px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto;
                border: 2px dashed rgba(255, 255, 255, 0.3);
                position: relative;
                overflow: hidden;
            }}

            .video-placeholder::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                animation: shimmer 2s infinite;
            }}

            @keyframes shimmer {{
                0% {{ left: -100%; }}
                100% {{ left: 100%; }}
            }}

            .video-content {{
                text-align: center;
                z-index: 1;
            }}

            .play-button {{
                width: 80px;
                height: 80px;
                background: rgba(16, 185, 129, 0.8);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem;
                cursor: pointer;
                transition: all 0.3s ease;
            }}

            .play-button:hover {{
                background: rgba(16, 185, 129, 1);
                transform: scale(1.1);
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
                    <h2 class="hero-subtitle">AI-Powered Investment Capital Hub</h2>
                    <p class="hero-description">
                        Revolutionary AI-driven trading platform that combines multi-agent intelligence, 
                        real-time market analysis, and institutional-grade portfolio management. 
                        Experience the future of investment management with our cutting-edge AI agents.
                    </p>
                    <button class="cta-button" onclick="launchDashboard()">
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
                        <video 
                            width="100%" 
                            height="400" 
                            autoplay 
                            muted 
                            loop 
                            style="border-radius: 20px; background: #000;"
                            poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 400'%3E%3Crect width='800' height='400' fill='%23667eea'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='white' font-size='24' font-family='Arial'%3EAgent RICH Demo%3C/text%3E%3C/svg%3E">
                            <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAABYhtZGF0AAACrQYF//+p3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDEyNSAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMTIgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz0xIGxvb2thaGVhZF90aHJlYWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFwbWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89MS40MCBhcT0xOjEuMDAAgAAAAWxliIQAMv/+/q3AP4C2AQAAABhnaAKAAAAAAAAAAAAAAAAAABABAAEAAAPoATjllJQABAAD//+X" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; pointer-events: none;">
                            <div style="background: rgba(0,0,0,0.7); padding: 2rem; border-radius: 15px; backdrop-filter: blur(10px);">
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
            function launchDashboard() {{
                // Force navigation to login page
                const currentUrl = new URL(window.location);
                currentUrl.searchParams.set('page', 'login');
                window.location.href = currentUrl.toString();
            }}

            function playDemo() {{
                alert("🎬 Demo video feature coming soon! For now, click 'Launch AI Dashboard' to explore the platform.");
            }}

            // Add smooth scrolling animation
            document.addEventListener('DOMContentLoaded', function() {{
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
    
    # Check URL parameters for navigation
    query_params = st.query_params
    if 'page' in query_params and query_params['page'] == 'login':
        st.session_state.page = 'login'
        st.query_params.clear()  # Clear the parameter
        st.rerun()
    
    # Display the landing page
    st.components.v1.html(landing_html, height=2000, scrolling=True)

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
        page_title="Agent RICH - AI-Powered Investment Capital Hub",
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
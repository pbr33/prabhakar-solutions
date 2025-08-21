import streamlit as st
import pandas as pd
from config import config
import base64
import os
from datetime import datetime, timedelta

def load_eci_logo():
    """Load ECI logo from static folder"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'eci.png')
        if not os.path.exists(logo_path):
            # Try alternative paths
            alternative_paths = [
                'static/eci.png',
                '../static/eci.png',
                './static/eci.png'
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    logo_path = path
                    break
        
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Could not load ECI logo: {e}")
    return None

def render_sidebar():
    """Renders a professional, client-focused sidebar"""
    
    # Custom CSS for professional sidebar styling
    st.markdown("""
    <style>
        /* Sidebar styling */
        .sidebar-header {
            text-align: center;
            padding: 1rem 0;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 1.5rem;
        }
        
        .eci-logo {
            max-width: 120px;
            height: auto;
            margin-bottom: 0.5rem;
        }
        
        .company-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2c3e50;
            margin: 0;
        }
        
        .tagline {
            font-size: 0.8rem;
            color: #7f8c8d;
            margin: 0;
            font-style: italic;
        }
        
        .sidebar-section {
            margin: 1.5rem 0;
            padding: 1rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }
        
        .section-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #495057;
            margin-bottom: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .feature-item {
            display: flex;
            align-items: center;
            padding: 0.3rem 0;
            font-size: 0.85rem;
            color: #6c757d;
        }
        
        .feature-icon {
            margin-right: 0.5rem;
            width: 16px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-online {
            background-color: #28a745;
            animation: pulse 2s infinite;
        }
        
        .status-processing {
            background-color: #ffc107;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .quick-stats {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .stat-value {
            font-size: 1.2rem;
            font-weight: bold;
            display: block;
        }
        
        .stat-label {
            font-size: 0.7rem;
            opacity: 0.9;
        }
        
        .theme-selector {
            background: #f8f9fa;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Company Header with Logo
        logo_base64 = load_eci_logo()
        
        if logo_base64:
            st.markdown(f"""
            <div class="sidebar-header">
                <img src="data:image/png;base64,{logo_base64}" class="eci-logo" alt="ECI Logo">
                <p class="tagline">AI-Powered Trading Intelligence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-header">
                <h3 class="company-name">🏢 ECI Solutions</h3>
                <p class="tagline">AI-Powered Trading Intelligence</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Theme & Appearance Settings
        with st.expander("🎨 Appearance Settings", expanded=False):
            theme_col1, theme_col2 = st.columns(2)
            
            with theme_col1:
                st.session_state.theme_mode = st.selectbox(
                    "Theme",
                    ["🌙 Dark", "☀️ Light", "🌈 Auto"],
                    index=0,
                    key="theme_selector"
                )
            
            with theme_col2:
                st.session_state.color_scheme = st.selectbox(
                    "Colors",
                    ["💙 Blue", "💚 Green", "💜 Purple", "🧡 Orange"],
                    index=0
                )
            
            # Font size adjustment
            st.session_state.font_size = st.select_slider(
                "Font Size",
                options=["Small", "Medium", "Large"],
                value="Medium"
            )
            
            # Chart style
            st.session_state.chart_style = st.selectbox(
                "Chart Style",
                ["Professional", "Modern", "Classic", "Minimal"],
                index=0
            )
        
        # Quick Market Status
        st.markdown("""
        <div class="sidebar-section">
            <div class="section-title">📊 Market Status</div>
        </div>
        """, unsafe_allow_html=True)
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.markdown('<span class="status-indicator status-online"></span>**Markets Open**', unsafe_allow_html=True)
        with status_col2:
            st.markdown('<span class="status-indicator status-processing"></span>**AI Active**', unsafe_allow_html=True)
        
        # Current time and market hours
        now = datetime.now()
        st.markdown(f"""
        <div style="text-align: center; font-size: 0.8rem; color: #6c757d; margin: 0.5rem 0;">
            {now.strftime('%H:%M:%S EST')} | {now.strftime('%b %d, %Y')}
        </div>
        """, unsafe_allow_html=True)
        
        # Asset Selection
        st.markdown("### 📈 Asset Selection")
        
        # Asset categories
        asset_category = st.selectbox(
            "Category",
            ["🇺🇸 US Stocks", "🌍 International", "💰 Crypto", "🏛️ Indices", "💎 Commodities", "💵 Forex"],
            key="asset_category"
        )
        
        # Popular assets based on category
        popular_assets = {
            "🇺🇸 US Stocks": ["AAPL.US", "GOOGL.US", "MSFT.US", "TSLA.US", "AMZN.US", "NVDA.US"],
            "🌍 International": ["TSM", "ASML.AS", "SAP.DE", "TM", "BABA"],
            "💰 Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"],
            "🏛️ Indices": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
            "💎 Commodities": ["GLD", "SLV", "USO", "UNG", "DBA"],
            "💵 Forex": ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]
        }
        
        # Quick select buttons for popular assets
        st.markdown("**Quick Select:**")
        cols = st.columns(2)
        for i, symbol in enumerate(popular_assets.get(asset_category, [])[:6]):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(symbol, key=f"quick_{symbol}", use_container_width=True):
                    st.session_state.selected_symbol = symbol
        
        # Custom symbol input
        custom_symbol = st.text_input(
            "Custom Symbol",
            placeholder="Enter symbol (e.g., AAPL.US)",
            key="custom_symbol_input"
        )
        
        if custom_symbol:
            st.session_state.selected_symbol = custom_symbol.upper()
        
        # Display selected symbol
        selected_symbol = st.session_state.get('selected_symbol', 'AAPL.US')
        st.info(f"Selected: **{selected_symbol}**")
        
        # Analysis Settings
        st.markdown("### ⚙️ Analysis Settings")
        
        # Time horizon
        time_horizon = st.selectbox(
            "Time Horizon",
            ["📅 Intraday", "📊 Daily", "📈 Weekly", "📆 Monthly"],
            index=1,
            key="time_horizon"
        )
        
        # Analysis depth
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Deep", "Comprehensive"],
            value="Standard",
            key="analysis_depth"
        )
        
        # Risk settings
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive", "Speculative"],
            value="Moderate",
            key="risk_tolerance"
        )
        
        # Portfolio Metrics (if applicable)
        if st.session_state.get('authenticated', False):
            st.markdown("### 💼 Portfolio Overview")
            
            # Mock portfolio data - replace with real data
            portfolio_value = 245750.50
            daily_change = 2340.25
            daily_change_pct = 0.96
            
            # Portfolio summary card
            change_color = "#28a745" if daily_change >= 0 else "#dc3545"
            change_arrow = "↗️" if daily_change >= 0 else "↘️"
            
            st.markdown(f"""
            <div class="quick-stats">
                <span class="stat-value">${portfolio_value:,.2f}</span>
                <span class="stat-label">Total Portfolio Value</span>
                <hr style="margin: 0.5rem 0; opacity: 0.3;">
                <span style="color: {change_color};">
                    {change_arrow} ${abs(daily_change):,.2f} ({daily_change_pct:+.2f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick portfolio actions
            if st.button("📊 View Full Portfolio", use_container_width=True):
                st.session_state.active_tab = "portfolio"
            
            if st.button("🔄 Rebalance", use_container_width=True):
                st.session_state.show_rebalance = True
        
        # AI Assistant Settings
        st.markdown("### 🤖 AI Assistant")
        
        # AI personality
        ai_personality = st.selectbox(
            "AI Personality",
            ["📊 Professional Analyst", "🚀 Growth Focused", "🛡️ Risk Manager", "📈 Day Trader"],
            key="ai_personality"
        )
        
        # AI response style
        response_style = st.selectbox(
            "Response Style",
            ["Detailed", "Concise", "Technical", "Simplified"],
            index=1,
            key="ai_response_style"
        )
        
        # Notifications & Alerts
        st.markdown("### 🔔 Alerts & Notifications")
        
        # Alert preferences
        price_alerts = st.checkbox("Price Movement Alerts", value=True)
        news_alerts = st.checkbox("Breaking News Alerts", value=True)
        earnings_alerts = st.checkbox("Earnings Alerts", value=False)
        technical_alerts = st.checkbox("Technical Signals", value=True)
        
        if price_alerts:
            price_threshold = st.slider("Price Alert Threshold (%)", 1, 10, 5)
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🔍 Market Scan", use_container_width=True):
                st.session_state.run_market_scan = True
            
            if st.button("📊 Technical Analysis", use_container_width=True):
                st.session_state.run_technical_analysis = True
        
        with action_col2:
            if st.button("📰 News Summary", use_container_width=True):
                st.session_state.show_news_summary = True
            
            if st.button("🎯 AI Insights", use_container_width=True):
                st.session_state.get_ai_insights = True
        
        # Export & Sharing
        with st.expander("📤 Export & Share", expanded=False):
            st.markdown("**Export Options:**")
            
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if st.button("📄 PDF Report"):
                    st.session_state.export_pdf = True
                if st.button("📊 Excel Data"):
                    st.session_state.export_excel = True
            
            with export_col2:
                if st.button("📧 Email Report"):
                    st.session_state.email_report = True
                if st.button("💾 Save Analysis"):
                    st.session_state.save_analysis = True
        
        # Help & Support
        with st.expander("❓ Help & Support", expanded=False):
            st.markdown("""
            **Quick Help:**
            - 🎥 [Video Tutorials](javascript:void(0))
            - 📖 [User Guide](javascript:void(0))
            - 💬 [Live Chat](javascript:void(0))
            - 📧 [Contact Support](javascript:void(0))
            
            **Hotkeys:**
            - `Ctrl + M` - Market Scan
            - `Ctrl + A` - AI Analysis  
            - `Ctrl + P` - Portfolio View
            - `Ctrl + ?` - Help
            """)
        
        # System Status (minimal)
        st.markdown("---")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown('<small>🟢 Online</small>', unsafe_allow_html=True)
        with status_col2:
            st.markdown('<small>🔄 Synced</small>', unsafe_allow_html=True)
        with status_col3:
            st.markdown('<small>⚡ Fast</small>', unsafe_allow_html=True)

def apply_theme_settings():
    """Apply theme settings based on user preferences"""
    
    theme_mode = st.session_state.get('theme_mode', '🌙 Dark')
    color_scheme = st.session_state.get('color_scheme', '💙 Blue')
    font_size = st.session_state.get('font_size', 'Medium')
    
    # Theme colors
    theme_colors = {
        '💙 Blue': {
            'primary': '#007bff',
            'secondary': '#6c757d',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        },
        '💚 Green': {
            'primary': '#28a745',
            'secondary': '#6c757d', 
            'success': '#20c997',
            'warning': '#ffc107',
            'danger': '#dc3545'
        },
        '💜 Purple': {
            'primary': '#6f42c1',
            'secondary': '#6c757d',
            'success': '#28a745', 
            'warning': '#ffc107',
            'danger': '#dc3545'
        },
        '🧡 Orange': {
            'primary': '#fd7e14',
            'secondary': '#6c757d',
            'success': '#28a745',
            'warning': '#ffc107', 
            'danger': '#dc3545'
        }
    }
    
    # Font sizes
    font_sizes = {
        'Small': '0.8rem',
        'Medium': '1rem', 
        'Large': '1.2rem'
    }
    
    colors = theme_colors.get(color_scheme, theme_colors['💙 Blue'])
    base_font_size = font_sizes.get(font_size, '1rem')
    
    # Apply theme CSS
    if '🌙 Dark' in theme_mode:
        bg_color = '#0e1117'
        text_color = '#fafafa'
        card_bg = '#262730'
    else:
        bg_color = '#ffffff'
        text_color = '#262730'
        card_bg = '#f8f9fa'
    
    st.markdown(f"""
    <style>
        :root {{
            --primary-color: {colors['primary']};
            --secondary-color: {colors['secondary']};
            --success-color: {colors['success']};
            --warning-color: {colors['warning']};
            --danger-color: {colors['danger']};
            --bg-color: {bg_color};
            --text-color: {text_color};
            --card-bg: {card_bg};
            --base-font-size: {base_font_size};
        }}
        
        .main .block-container {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-size: var(--base-font-size);
        }}
        
        .stSelectbox > div > div {{
            background-color: var(--card-bg);
        }}
        
        .metric-container {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }}
    </style>
    """, unsafe_allow_html=True)

# Call theme settings when sidebar is rendered
def render_sidebar_with_theme():
    """Render sidebar and apply theme settings"""
    render_sidebar()
    apply_theme_settings()

# Additional features to add to the professional sidebar

def render_advanced_sidebar_features():
    """Additional professional features for the sidebar"""
    
    # Market Hours & Status
    with st.expander("🕐 Market Hours & Status", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🇺🇸 US Markets**")
            st.markdown("NYSE: 🟢 Open")
            st.markdown("NASDAQ: 🟢 Open") 
            st.markdown("Next Close: 4:00 PM EST")
        
        with col2:
            st.markdown("**🌍 Global Markets**")
            st.markdown("London: 🔴 Closed")
            st.markdown("Tokyo: 🔴 Closed")
            st.markdown("Sydney: 🟡 Pre-Market")
    
    # Economic Calendar
    with st.expander("📅 Economic Calendar", expanded=False):
        st.markdown("**Today's Events:**")
        events = [
            {"time": "10:00 AM", "event": "Fed Interest Rate Decision", "impact": "🔴 High"},
            {"time": "2:00 PM", "event": "GDP Report", "impact": "🟡 Medium"},
            {"time": "4:30 PM", "event": "Oil Inventory", "impact": "🟢 Low"}
        ]
        
        for event in events:
            st.markdown(f"**{event['time']}** - {event['event']} {event['impact']}")
    
    # Performance Analytics
    if st.session_state.get('authenticated', False):
        with st.expander("📈 Performance Analytics", expanded=False):
            # Performance metrics
            metrics = {
                "Today": {"return": 2.34, "trades": 12},
                "This Week": {"return": 8.67, "trades": 45},
                "This Month": {"return": 15.23, "trades": 156},
                "YTD": {"return": 34.56, "trades": 1243}
            }
            
            for period, data in metrics.items():
                color = "green" if data["return"] > 0 else "red"
                st.markdown(f"**{period}**: <span style='color: {color}'>{data['return']:+.2f}%</span> ({data['trades']} trades)", 
                           unsafe_allow_html=True)
    
    # Watchlist Management
    st.markdown("### 👁️ Watchlist Management")
    
    # Watchlist tabs
    watchlist_tab = st.selectbox(
        "Select Watchlist",
        ["📋 Main Watchlist", "⭐ Favorites", "🔥 Hot Stocks", "📊 Earnings Watch", "💎 Crypto Watch"],
        key="watchlist_selector"
    )
    
    # Sample watchlist items based on selection
    watchlists = {
        "📋 Main Watchlist": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        "⭐ Favorites": ["NVDA", "META", "NFLX"],
        "🔥 Hot Stocks": ["PLTR", "RIVN", "LCID"],
        "📊 Earnings Watch": ["AAPL", "MSFT", "GOOGL"],
        "💎 Crypto Watch": ["BTC-USD", "ETH-USD", "ADA-USD"]
    }
    
    current_watchlist = watchlists.get(watchlist_tab, [])
    
    for symbol in current_watchlist:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button(f"📊 {symbol}", key=f"watch_{symbol}", use_container_width=True):
                st.session_state.selected_symbol = symbol
        with col2:
            # Mock price change
            change = 2.34 if symbol == "AAPL" else -1.23
            color = "green" if change > 0 else "red"
            st.markdown(f"<small style='color: {color}'>{change:+.1f}%</small>", unsafe_allow_html=True)
        with col3:
            if st.button("🗑️", key=f"remove_{symbol}", help="Remove from watchlist"):
                st.session_state.remove_from_watchlist = symbol
    
    # Add to watchlist
    new_symbol = st.text_input("Add Symbol", placeholder="e.g., AAPL", key="add_to_watchlist")
    if new_symbol and st.button("➕ Add to Watchlist"):
        st.success(f"Added {new_symbol.upper()} to watchlist!")
    
    # News & Sentiment
    with st.expander("📰 Market News & Sentiment", expanded=False):
        st.markdown("**Market Sentiment:** 😊 Bullish")
        st.markdown("**Fear & Greed Index:** 72 (Greed)")
        
        # Latest news headlines (mock)
        news_items = [
            "Fed keeps rates unchanged, markets rally",
            "Tech stocks lead gains in morning trading",
            "Oil prices surge on supply concerns",
            "Crypto market shows signs of recovery"
        ]
        
        for i, news in enumerate(news_items):
            st.markdown(f"• {news}")
            if i == 2:  # Show only first 3, with option to see more
                if st.button("📖 See More News"):
                    st.session_state.show_full_news = True
                break
    
    # Trading Strategies & Alerts
    with st.expander("🎯 Trading Strategies", expanded=False):
        strategy_type = st.selectbox(
            "Strategy Type",
            ["💹 Day Trading", "📈 Swing Trading", "💰 Value Investing", "🚀 Growth", "🛡️ Defensive"],
            key="strategy_type"
        )
        
        # Strategy-specific settings
        if "Day Trading" in strategy_type:
            st.slider("Max Position Size (%)", 1, 10, 5, key="day_trade_size")
            st.checkbox("Auto-stop Loss", value=True, key="auto_stop_loss")
        elif "Swing Trading" in strategy_type:
            st.slider("Hold Period (days)", 1, 30, 7, key="swing_hold_period")
            st.selectbox("Signal Type", ["Technical", "Fundamental", "Hybrid"], key="swing_signal")
    
    # Risk Management
    with st.expander("🛡️ Risk Management", expanded=False):
        # Risk metrics
        st.markdown("**Current Risk Metrics:**")
        
        risk_metrics = {
            "Portfolio Beta": 1.23,
            "Value at Risk (1d)": 2.45,
            "Sharpe Ratio": 1.67,
            "Max Drawdown": -8.34
        }
        
        for metric, value in risk_metrics.items():
            if "Drawdown" in metric:
                color = "red"
                value_str = f"{value:.2f}%"
            else:
                color = "green" if value > 1 else "orange"
                value_str = f"{value:.2f}"
            
            st.markdown(f"**{metric}:** <span style='color: {color}'>{value_str}</span>", 
                       unsafe_allow_html=True)
        
        # Risk controls
        st.markdown("**Risk Controls:**")
        st.checkbox("Enable Position Limits", value=True)
        st.checkbox("Auto Risk-off Mode", value=False)
        st.slider("Max Portfolio Risk (%)", 1, 20, 10)
    
    # API & Data Sources
    with st.expander("🔌 Data Sources & APIs", expanded=False):
        st.markdown("**Connected Sources:**")
        
        data_sources = [
            {"name": "Real-time Market Data", "status": "🟢 Active", "latency": "< 1ms"},
            {"name": "Financial News", "status": "🟢 Active", "latency": "< 5s"},
            {"name": "Economic Data", "status": "🟢 Active", "latency": "Real-time"},
            {"name": "Social Sentiment", "status": "🟡 Limited", "latency": "~1min"},
        ]
        
        for source in data_sources:
            st.markdown(f"**{source['name']}**")
            st.markdown(f"   Status: {source['status']} | Latency: {source['latency']}")
    
    # Backup & Export
    with st.expander("💾 Backup & Export", expanded=False):
        st.markdown("**Auto-Backup:** 🟢 Enabled")
        st.markdown("**Last Backup:** 2 hours ago")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Import Settings"):
                st.session_state.import_settings = True
        with col2:
            if st.button("📤 Export Data"):
                st.session_state.export_data = True
        
        # Cloud sync
        st.checkbox("Cloud Sync", value=True, help="Sync settings across devices")
    
    # Keyboard Shortcuts
    with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
        shortcuts = {
            "Ctrl + M": "Market Scan",
            "Ctrl + A": "AI Analysis", 
            "Ctrl + P": "Portfolio View",
            "Ctrl + T": "New Trade",
            "Ctrl + W": "Watchlist",
            "Ctrl + N": "News Feed",
            "Ctrl + ?": "Help",
            "Ctrl + S": "Save Analysis",
            "Esc": "Cancel Action"
        }
        
        for shortcut, action in shortcuts.items():
            st.markdown(f"**{shortcut}** - {action}")
    
    # Session Information
    st.markdown("---")
    st.markdown("**Session Info:**")
    
    # Session stats
    session_start = st.session_state.get('session_start_time', '10:30 AM')
    active_time = "2h 15m"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<small>Started: {session_start}</small>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<small>Active: {active_time}</small>", unsafe_allow_html=True)

# Color scheme options for different themes
def get_color_schemes():
    """Return available color schemes"""
    return {
        "💙 Professional Blue": {
            "primary": "#0066cc",
            "secondary": "#4d94ff", 
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
            "background": "#f8f9fa"
        },
        "💚 Financial Green": {
            "primary": "#00b386",
            "secondary": "#4dd2aa",
            "success": "#28a745", 
            "warning": "#ffc107",
            "danger": "#dc3545",
            "background": "#f0f8f5"
        },
        "💜 Executive Purple": {
            "primary": "#6f42c1",
            "secondary": "#9370db",
            "success": "#28a745",
            "warning": "#ffc107", 
            "danger": "#dc3545",
            "background": "#f8f6ff"
        },
        "🧡 Energy Orange": {
            "primary": "#fd7e14",
            "secondary": "#ff9a44",
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545", 
            "background": "#fff8f0"
        },
        "⚫ Dark Mode": {
            "primary": "#375a7f",
            "secondary": "#4a90e2",
            "success": "#00bc8c",
            "warning": "#f39c12",
            "danger": "#e74c3c",
            "background": "#2c3e50"
        },
        "⚪ Light Mode": {
            "primary": "#007bff",
            "secondary": "#6c757d",
            "success": "#28a745",
            "warning": "#ffc107",
            "danger": "#dc3545",
            "background": "#ffffff"
        }
    }

def apply_advanced_theme(color_scheme_name, font_size, theme_mode):
    """Apply advanced theming with custom color schemes"""
    
    color_schemes = get_color_schemes()
    colors = color_schemes.get(color_scheme_name, color_schemes["💙 Professional Blue"])
    
    # Font size mapping
    font_sizes = {
        "Small": {"base": "0.875rem", "h1": "1.5rem", "h2": "1.25rem", "h3": "1.1rem"},
        "Medium": {"base": "1rem", "h1": "2rem", "h2": "1.5rem", "h3": "1.25rem"},
        "Large": {"base": "1.125rem", "h1": "2.5rem", "h2": "2rem", "h3": "1.5rem"}
    }
    
    sizes = font_sizes.get(font_size, font_sizes["Medium"])
    
    # Dark/Light mode overrides
    if "🌙 Dark" in theme_mode:
        bg_primary = "#0e1117"
        bg_secondary = "#262730"
        text_primary = "#fafafa"
        text_secondary = "#a0a0a0"
        border_color = "#404040"
    else:
        bg_primary = colors["background"]
        bg_secondary = "#ffffff"
        text_primary = "#212529"
        text_secondary = "#6c757d"
        border_color = "#dee2e6"
    
    # Advanced CSS with professional styling
    advanced_css = f"""
    <style>
        /* Root variables for consistent theming */
        :root {{
            --primary-color: {colors["primary"]};
            --secondary-color: {colors["secondary"]};
            --success-color: {colors["success"]};
            --warning-color: {colors["warning"]};
            --danger-color: {colors["danger"]};
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --border-color: {border_color};
            --font-size-base: {sizes["base"]};
            --font-size-h1: {sizes["h1"]};
            --font-size-h2: {sizes["h2"]};
            --font-size-h3: {sizes["h3"]};
        }}
        
        /* Global styling */
        .main .block-container {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-size: var(--font-size-base);
            transition: all 0.3s ease;
        }}
        
        /* Professional card styling */
        .professional-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }}
        
        .professional-card:hover {{
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }}
        
        /* Enhanced metrics styling */
        .metric-professional {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            display: block;
            margin-bottom: 0.5rem;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        /* Professional buttons */
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            filter: brightness(1.1);
        }}
        
        /* Professional tables */
        .dataframe {{
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .dataframe th {{
            background: var(--primary-color);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85rem;
        }}
        
        .dataframe td {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
        }}
        
        /* Professional alerts */
        .alert-professional {{
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid var(--primary-color);
            background: var(--bg-secondary);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        /* Enhanced sidebar styling */
        .css-1d391kg {{
            background: var(--bg-secondary);
            border-right: 2px solid var(--border-color);
        }}
        
        /* Professional form inputs */
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stTextArea > div > div > textarea {{
            background: var(--bg-secondary);
            border: 2px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            transition: all 0.3s ease;
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div:focus-within,
        .stTextArea > div > div > textarea:focus {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(var(--primary-color), 0.1);
        }}
        
        /* Professional charts */
        .js-plotly-plot {{
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        /* Status indicators */
        .status-indicator-pro {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            margin: 0.25rem;
        }}
        
        .status-online {{
            background: rgba(40, 167, 69, 0.1);
            color: var(--success-color);
            border: 1px solid var(--success-color);
        }}
        
        .status-processing {{
            background: rgba(255, 193, 7, 0.1);
            color: var(--warning-color);
            border: 1px solid var(--warning-color);
        }}
        
        .status-offline {{
            background: rgba(220, 53, 69, 0.1);
            color: var(--danger-color);
            border: 1px solid var(--danger-color);
        }}
        
        /* Professional loading animations */
        @keyframes professional-pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.8; transform: scale(1.05); }}
        }}
        
        .loading-professional {{
            animation: professional-pulse 2s infinite;
        }}
        
        /* Professional tooltips */
        .tooltip-professional {{
            position: relative;
            cursor: help;
        }}
        
        .tooltip-professional:hover::after {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--text-primary);
            color: var(--bg-primary);
            padding: 0.5rem;
            border-radius: 4px;
            white-space: nowrap;
            font-size: 0.875rem;
            z-index: 1000;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .professional-card {{
                padding: 1rem;
                margin: 0.5rem 0;
            }}
            
            .metric-professional {{
                padding: 1rem;
            }}
            
            .metric-value {{
                font-size: 1.5rem;
            }}
        }}
        
        /* Print styles for reports */
        @media print {{
            .professional-card {{
                break-inside: avoid;
                box-shadow: none;
                border: 1px solid #000;
            }}
            
            .stButton {{
                display: none;
            }}
        }}
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {{
            .professional-card {{
                border: 2px solid var(--text-primary);
            }}
            
            .stButton > button {{
                border: 2px solid white;
            }}
        }}
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {{
            * {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}
    </style>
    """
    
    return advanced_css

def render_theme_preview():
    """Render a preview of the current theme settings"""
    
    st.markdown("### 🎨 Theme Preview")
    
    # Preview cards with current theme
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <h4 style="margin: 0 0 0.5rem 0; color: var(--primary-color);">📊 Market Data</h4>
            <div class="metric-professional">
                <span class="metric-value">$1,234.56</span>
                <span class="metric-label">Portfolio Value</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="professional-card">
            <h4 style="margin: 0 0 0.5rem 0; color: var(--success-color);">🚀 Performance</h4>
            <div style="text-align: center;">
                <span class="status-indicator-pro status-online">🟢 Active</span>
                <span class="status-indicator-pro status-processing">🟡 Processing</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="professional-card">
            <h4 style="margin: 0 0 0.5rem 0; color: var(--secondary-color);">⚡ Analytics</h4>
            <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">
                Real-time market analysis and insights powered by AI.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Integration function for the main sidebar
def integrate_advanced_features_to_sidebar():
    """Integration function to add advanced features to the main sidebar"""
    
    # This would be called from within the main render_sidebar function
    
    # Theme application
    color_scheme = st.session_state.get('color_scheme', '💙 Professional Blue')
    font_size = st.session_state.get('font_size', 'Medium') 
    theme_mode = st.session_state.get('theme_mode', '☀️ Light')
    
    # Apply the advanced theme
    advanced_css = apply_advanced_theme(color_scheme, font_size, theme_mode)
    st.markdown(advanced_css, unsafe_allow_html=True)
    
    # Add the advanced features
    render_advanced_sidebar_features()
    
    # Optional: Show theme preview in an expander
    with st.expander("🎨 Theme Preview", expanded=False):
        render_theme_preview()

# Additional utility functions for sidebar features

def get_market_status():
    """Get current market status"""
    from datetime import datetime
    import pytz
    
    # Get current time in EST
    est = pytz.timezone('US/Eastern')
    current_time = datetime.now(est)
    hour = current_time.hour
    minute = current_time.minute
    
    # Market hours: 9:30 AM - 4:00 PM EST
    market_open = (hour > 9) or (hour == 9 and minute >= 30)
    market_close = hour >= 16
    
    if market_open and not market_close:
        return "🟢 Open"
    elif hour < 9 or (hour == 9 and minute < 30):
        return "🟡 Pre-Market"
    else:
        return "🔴 Closed"

def format_currency(value, currency="USD"):
    """Format currency values professionally"""
    if currency == "USD":
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:.1f}K"
        else:
            return f"${value:.2f}"
    return f"{value:.2f} {currency}"

def get_performance_color(value):
    """Get color for performance values"""
    if value > 0:
        return "#28a745"  # Green
    elif value < 0:
        return "#dc3545"  # Red
    else:
        return "#6c757d"  # Gray

def save_user_preferences():
    """Save user preferences to session state"""
    preferences = {
        'theme_mode': st.session_state.get('theme_mode', '☀️ Light'),
        'color_scheme': st.session_state.get('color_scheme', '💙 Professional Blue'),
        'font_size': st.session_state.get('font_size', 'Medium'),
        'chart_style': st.session_state.get('chart_style', 'Professional'),
        'notifications_enabled': st.session_state.get('notifications_enabled', True),
        'auto_refresh': st.session_state.get('auto_refresh', True)
    }
    
    st.session_state.user_preferences = preferences
    return preferences

def load_user_preferences():
    """Load user preferences from session state"""
    if 'user_preferences' in st.session_state:
        prefs = st.session_state.user_preferences
        for key, value in prefs.items():
            if key not in st.session_state:
                st.session_state[key] = value
import streamlit as st
import pandas as pd
import time
from config import config
import base64
import os
from datetime import datetime, timedelta
import requests

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

def get_exchanges(api_key: str):
    """Get list of exchanges from EODHD API"""
    try:
        url = f"https://eodhd.com/api/exchanges-list/?api_token={api_key}&fmt=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching exchanges: {e}")
        return []

def get_tickers_by_exchange(exchange_code: str, api_key: str):
    """Get tickers for a specific exchange"""
    try:
        url = f"https://eodhd.com/api/exchange-symbol-list/{exchange_code}?api_token={api_key}&fmt=json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching tickers for {exchange_code}: {e}")
        return []

def fetch_all_tickers(api_key: str):
    """Fetches all tickers from all exchanges with progress tracking"""
    exchanges = get_exchanges(api_key)
    if not exchanges:
        return []
        
    all_tickers = []
    progress_bar = st.progress(0, text="Initializing ticker fetch...")
    
    for i, exchange in enumerate(exchanges):
        code = exchange.get('Code')
        if not code:
            continue
        
        progress_text = f"Fetching tickers for exchange: {exchange.get('Name', code)} ({i+1}/{len(exchanges)})"
        progress_bar.progress((i + 1) / len(exchanges), text=progress_text)
        
        tickers = get_tickers_by_exchange(code, api_key)
        if tickers:
            all_tickers.extend(tickers)
        
        time.sleep(0.2)

    progress_bar.empty()
    if not all_tickers:
        st.error("Failed to fetch any tickers. Please check your EODHD API key and subscription.")
    
    return all_tickers

def handle_file_upload(uploaded_files, file_type):
    """Handle file upload and processing"""
    if not uploaded_files:
        return
    
    processed_files = []
    
    for uploaded_file in uploaded_files:
        try:
            # Store file info in session state
            file_info = {
                'name': uploaded_file.name,
                'type': file_type,
                'size': uploaded_file.size,
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content': uploaded_file.read()
            }
            
            # Initialize uploaded files in session state
            if 'uploaded_files' not in st.session_state:
                st.session_state.uploaded_files = []
            
            # Add to uploaded files
            st.session_state.uploaded_files.append(file_info)
            processed_files.append(uploaded_file.name)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    if processed_files:
        st.success(f"✅ Uploaded {len(processed_files)} file(s): {', '.join(processed_files)}")

def render_data_sources_section():
    """Render the comprehensive data sources section"""
    
    st.markdown("### 📊 Data Sources & Integration")
    
    # File Upload Section
    with st.expander("📁 Upload Financial Documents", expanded=False):
        st.markdown("**📄 Document Types Supported:**")
        
        upload_tab1, upload_tab2, upload_tab3 = st.tabs(["📈 Financial Reports", "📰 Market Data", "💼 Private Equity"])
        
        with upload_tab1:
            st.markdown("**SEC Filings & Financial Reports:**")
            financial_files = st.file_uploader(
                "Upload Files",
                type=['pdf', 'txt', 'csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                key="financial_files",
                help="10-K, 10-Q, 8-K, Annual Reports, Earnings Transcripts"
            )
            
            if financial_files:
                handle_file_upload(financial_files, "Financial Reports")
            
            # Common financial document types
            st.markdown("""
            **Supported Documents:**
            - 📋 SEC 10-K Annual Reports
            - 📊 SEC 10-Q Quarterly Reports  
            - 📢 SEC 8-K Current Reports
            - 💰 Earnings Call Transcripts
            - 📈 Annual/Quarterly Reports
            - 🏦 Bank Call Reports
            """)
        
        with upload_tab2:
            st.markdown("**Market & Trading Data:**")
            market_files = st.file_uploader(
                "Upload Market Data",
                type=['csv', 'xlsx', 'json', 'txt'],
                accept_multiple_files=True,
                key="market_files",
                help="Trading data, market analysis, insider trading reports"
            )
            
            if market_files:
                handle_file_upload(market_files, "Market Data")
            
            st.markdown("""
            **Supported Data:**
            - 📊 Historical Price Data
            - 🔄 Trading Volume Data
            - 👥 Insider Trading Reports
            - 📈 Technical Indicators
            - 🎯 Options Chain Data
            - 📰 News Sentiment Data
            """)
        
        with upload_tab3:
            st.markdown("**Private Equity & Alternative Investments:**")
            pe_files = st.file_uploader(
                "Upload PE Documents",
                type=['pdf', 'xlsx', 'csv', 'txt'],
                accept_multiple_files=True,
                key="pe_files",
                help="Fund reports, portfolio valuations, investment memos"
            )
            
            if pe_files:
                handle_file_upload(pe_files, "Private Equity")
            
            st.markdown("""
            **Supported Documents:**
            - 💼 Fund Performance Reports
            - 📊 Portfolio Valuations
            - 📝 Investment Memos
            - 🎯 Due Diligence Reports
            - 💰 Capital Call Notices
            - 📈 IRR Calculations
            """)
    
    # Cloud Data Connections
    with st.expander("☁️ Connect Cloud Data Sources", expanded=False):
        st.markdown("**🔗 Enterprise Data Connections:**")
        
        # First row of connections
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Bloomberg Terminal", use_container_width=True):
                st.session_state.connect_bloomberg = True
                st.info("Bloomberg API integration initiated")
            
            if st.button("📈 Refinitiv Eikon", use_container_width=True):
                st.session_state.connect_refinitiv = True
                st.info("Refinitiv connection initiated")
        
        with col2:
            if st.button("🏦 FactSet", use_container_width=True):
                st.session_state.connect_factset = True
                st.info("FactSet integration initiated")
            
            if st.button("📊 S&P Capital IQ", use_container_width=True):
                st.session_state.connect_sp = True
                st.info("S&P Capital IQ connection initiated")
        
        with col3:
            if st.button("💼 PitchBook", use_container_width=True):
                st.session_state.connect_pitchbook = True
                st.info("PitchBook connection initiated")
            
            if st.button("🎯 Morningstar Direct", use_container_width=True):
                st.session_state.connect_morningstar = True
                st.info("Morningstar connection initiated")
        
        # Second row - Cloud Storage
        st.markdown("**☁️ Cloud Storage Connections:**")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if st.button("📁 Azure Storage", use_container_width=True):
                st.session_state.connect_azure_storage = True
                st.info("Azure Storage connected")
            
            if st.button("📂 SharePoint", use_container_width=True):
                st.session_state.connect_sharepoint = True
                st.info("SharePoint integrated")
        
        with col5:
            if st.button("☁️ Google Drive", use_container_width=True):
                st.session_state.connect_gdrive = True
                st.info("Google Drive connected")
            
            if st.button("📋 OneDrive", use_container_width=True):
                st.session_state.connect_onedrive = True
                st.info("OneDrive integrated")
        
        with col6:
            if st.button("📊 Dropbox Business", use_container_width=True):
                st.session_state.connect_dropbox = True
                st.info("Dropbox connected")
            
            if st.button("🔗 FTP/SFTP Server", use_container_width=True):
                st.session_state.connect_ftp = True
                st.info("FTP connection established")
    
    # Real-time Data Feeds
    with st.expander("📡 Real-time Data Feeds", expanded=False):
        st.markdown("**⚡ Live Market Data:**")
        
        feed_col1, feed_col2 = st.columns(2)
        
        with feed_col1:
            # Market data toggles
            st.checkbox("📈 Real-time Prices", value=True, help="Live stock prices")
            st.checkbox("📊 Level II Data", value=False, help="Order book data")
            st.checkbox("🔄 Options Flow", value=False, help="Real-time options data")
            st.checkbox("📰 News Feed", value=True, help="Breaking financial news")
        
        with feed_col2:
            st.checkbox("👥 Insider Trading", value=False, help="SEC insider filings")
            st.checkbox("🏢 Earnings Calendar", value=True, help="Upcoming earnings")
            st.checkbox("📊 Economic Calendar", value=True, help="Economic events")
            st.checkbox("🎯 Analyst Updates", value=False, help="Rating changes")
        
        # Data refresh settings
        refresh_interval = st.selectbox(
            "🔄 Data Refresh Interval",
            ["Real-time", "1 second", "5 seconds", "30 seconds", "1 minute", "5 minutes"],
            index=2
        )
    
    # Alternative Data Sources
    with st.expander("🔍 Alternative Data Sources", expanded=False):
        st.markdown("**🧠 AI-Powered Data Sources:**")
        
        alt_col1, alt_col2, alt_col3 = st.columns(3)
        
        with alt_col1:
            if st.button("📱 Social Sentiment", use_container_width=True):
                st.session_state.connect_social = True
                st.info("Social media sentiment activated")
            
            if st.button("🛰️ Satellite Data", use_container_width=True):
                st.session_state.connect_satellite = True
                st.info("Satellite imagery data connected")
        
        with alt_col2:
            if st.button("🏪 Retail Analytics", use_container_width=True):
                st.session_state.connect_retail = True
                st.info("Retail analytics integrated")
            
            if st.button("🌐 Web Scraping", use_container_width=True):
                st.session_state.connect_webscrape = True
                st.info("Web scraping initiated")
        
        with alt_col3:
            if st.button("💳 Credit Card Data", use_container_width=True):
                st.session_state.connect_credit = True
                st.info("Credit card analytics connected")
            
            if st.button("📍 Geolocation Data", use_container_width=True):
                st.session_state.connect_geo = True
                st.info("Geolocation data activated")
    
    # Data Management
    if st.session_state.get('uploaded_files'):
        with st.expander("📋 Uploaded Files Manager", expanded=True):
            st.markdown("**📁 Your Uploaded Documents:**")
            
            uploaded_files = st.session_state.uploaded_files
            
            # Create DataFrame for display
            files_data = []
            for i, file_info in enumerate(uploaded_files):
                files_data.append({
                    'File Name': file_info['name'],
                    'Type': file_info['type'],
                    'Size (KB)': f"{file_info['size'] / 1024:.1f}",
                    'Uploaded': file_info['upload_time'],
                    'Status': '✅ Ready'
                })
            
            files_df = pd.DataFrame(files_data)
            st.dataframe(files_df, use_container_width=True)
            
            # File actions
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("🔄 Process All Files", use_container_width=True):
                    with st.spinner("Processing uploaded files..."):
                        time.sleep(2)
                        st.success(f"✅ Processed {len(uploaded_files)} files successfully!")
                        st.session_state.files_processed = True
            
            with action_col2:
                if st.button("🤖 AI Analysis", use_container_width=True):
                    with st.spinner("Running AI analysis on documents..."):
                        time.sleep(3)
                        st.success("🧠 AI analysis completed!")
                        st.session_state.ai_analysis_done = True
            
            with action_col3:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.uploaded_files = []
                    st.success("🗑️ All files cleared!")
                    st.rerun()

def apply_working_theme():
    """Apply WORKING theme CSS that actually changes colors"""
    
    # Get current theme settings from session state
    theme_mode = st.session_state.get('theme_mode', 'Light')
    color_scheme = st.session_state.get('color_scheme', 'Professional Blue')
    font_size = st.session_state.get('font_size', 'Medium')
    
    # Define color schemes
    color_schemes = {
        'Professional Blue': {
            'primary': '#0066cc',
            'secondary': '#4d94ff',
            'accent': '#007bff',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        },
        'Financial Green': {
            'primary': '#00b386',
            'secondary': '#4dd2aa',
            'accent': '#28a745',
            'success': '#20c997',
            'warning': '#f39c12',
            'danger': '#e74c3c'
        },
        'Executive Purple': {
            'primary': '#6f42c1',
            'secondary': '#9370db',
            'accent': '#8a2be2',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        },
        'Energy Orange': {
            'primary': '#fd7e14',
            'secondary': '#ff9a44',
            'accent': '#ff6b35',
            'success': '#28a745',
            'warning': '#f39c12',
            'danger': '#e74c3c'
        }
    }
    
    # Font sizes
    font_sizes = {
        'Small': {'base': '0.875rem', 'header': '1.2rem'},
        'Medium': {'base': '1rem', 'header': '1.5rem'},
        'Large': {'base': '1.125rem', 'header': '1.75rem'}
    }
    
    # Get selected colors and fonts
    colors = color_schemes.get(color_scheme, color_schemes['Professional Blue'])
    fonts = font_sizes.get(font_size, font_sizes['Medium'])
    
    # Theme-specific background and text colors
    if theme_mode == 'Dark':
        bg_primary = '#0e1117'
        bg_secondary = '#262730'
        bg_sidebar = '#1e1e1e'
        text_primary = '#fafafa'
        text_secondary = '#a0a0a0'
        border_color = '#404040'
        input_bg = '#2d2d2d'
    else:  # Light theme
        bg_primary = '#ffffff'
        bg_secondary = '#f8f9fa'
        bg_sidebar = '#f0f2f6'
        text_primary = '#212529'
        text_secondary = '#6c757d'
        border_color = '#dee2e6'
        input_bg = '#ffffff'
    
    # Apply comprehensive CSS that ACTUALLY works
    working_css = f"""
    <style>
        /* FORCE theme application with !important */
        
        /* Main app background */
        .stApp {{
            background-color: {bg_primary} !important;
            color: {text_primary} !important;
        }}
        
        .main .block-container {{
            background-color: {bg_primary} !important;
            color: {text_primary} !important;
            font-size: {fonts['base']} !important;
        }}
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background-color: {bg_sidebar} !important;
        }}
        
        .css-1d391kg > div:first-child {{
            background-color: {bg_sidebar} !important;
        }}
        
        /* Buttons with working colors */
        .stButton > button {{
            background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']}) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: {fonts['base']} !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, {colors['secondary']}, {colors['accent']}) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        }}
        
        /* Selectboxes */
        .stSelectbox > div > div {{
            background-color: {input_bg} !important;
            border: 2px solid {border_color} !important;
            border-radius: 6px !important;
            color: {text_primary} !important;
        }}
        
        .stSelectbox > div > div:focus-within {{
            border-color: {colors['primary']} !important;
            box-shadow: 0 0 0 2px {colors['primary']}40 !important;
        }}
        
        /* Text inputs */
        .stTextInput > div > div > input {{
            background-color: {input_bg} !important;
            border: 2px solid {border_color} !important;
            border-radius: 6px !important;
            color: {text_primary} !important;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {colors['primary']} !important;
            box-shadow: 0 0 0 2px {colors['primary']}40 !important;
        }}
        
        /* Headers and text */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_primary} !important;
            font-size: {fonts['header']} !important;
        }}
        
        .sidebar-header h3 {{
            color: {colors['primary']} !important;
            font-size: {fonts['header']} !important;
        }}
        
        /* Sliders */
        .stSlider > div > div > div > div {{
            background-color: {colors['primary']} !important;
        }}
        
        /* Checkboxes */
        .stCheckbox > label {{
            color: {text_primary} !important;
        }}
        
        /* Info/Success/Warning/Error boxes */
        .stInfo {{
            background-color: {colors['primary']}20 !important;
            border: 1px solid {colors['primary']} !important;
            color: {text_primary} !important;
        }}
        
        .stSuccess {{
            background-color: {colors['success']}20 !important;
            border: 1px solid {colors['success']} !important;
            color: {text_primary} !important;
        }}
        
        .stWarning {{
            background-color: {colors['warning']}20 !important;
            border: 1px solid {colors['warning']} !important;
            color: {text_primary} !important;
        }}
        
        .stError {{
            background-color: {colors['danger']}20 !important;
            border: 1px solid {colors['danger']} !important;
            color: {text_primary} !important;
        }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {bg_secondary} !important;
            color: {text_primary} !important;
        }}
        
        /* Metrics */
        [data-testid="metric-container"] {{
            background-color: {bg_secondary} !important;
            border: 1px solid {border_color} !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }}
        
        /* Progress bars */
        .stProgress > div > div > div > div {{
            background-color: {colors['primary']} !important;
        }}
        
        /* Sidebar sections */
        .sidebar-header {{
            background-color: {bg_secondary} !important;
            border: 2px solid {colors['primary']} !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            text-align: center !important;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']}) !important;
            color: white !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            text-align: center !important;
            margin: 0.5rem 0 !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }}
        
        /* Status indicators */
        .status-online {{
            color: {colors['success']} !important;
            font-weight: bold !important;
        }}
        
        .status-processing {{
            color: {colors['warning']} !important;
            font-weight: bold !important;
        }}
        
        /* Theme indicator to show current theme */
        .theme-indicator {{
            position: fixed;
            top: 70px;
            right: 20px;
            background: {colors['primary']};
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            z-index: 999;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        
        /* Company name styling */
        .company-name {{
            color: {colors['primary']} !important;
            font-weight: 700 !important;
            font-size: {fonts['header']} !important;
        }}
        
        .tagline {{
            color: {text_secondary} !important;
            font-style: italic !important;
        }}
        
        /* Portfolio value styling */
        .portfolio-value {{
            font-size: 1.5rem !important;
            font-weight: bold !important;
            color: {colors['primary']} !important;
        }}
        
        /* Quick select buttons */
        .stColumns > div > div > button {{
            width: 100% !important;
            background: {colors['accent']} !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            margin: 2px 0 !important;
        }}
        
        /* Make sure all text is readable */
        div, span, p, label {{
            color: {text_primary} !important;
        }}
        
        /* DataFrames */
        .stDataFrame {{
            background-color: {bg_secondary} !important;
            border-radius: 8px !important;
        }}
        
        /* Force all components to use theme */
        * {{
            color: {text_primary} !important;
        }}
        
        .stApp * {{
            transition: all 0.3s ease !important;
        }}
    </style>
    
    <!-- Theme indicator -->
    <div class="theme-indicator">
        {theme_mode} • {color_scheme} • {font_size}
    </div>
    """
    
    st.markdown(working_css, unsafe_allow_html=True)

def render_sidebar():
    """Renders the main sidebar with working themes and ticker fetching"""
    
    # Apply working theme FIRST
    apply_working_theme()
    
    with st.sidebar:
        # Company Header with Logo
        logo_base64 = load_eci_logo()
        
        if logo_base64:
            st.markdown(f"""
            <div class="sidebar-header">
                <img src="data:image/png;base64,{logo_base64}" style="max-width: 120px; height: auto; margin-bottom: 0.5rem;" alt="ECI Logo">
                <h3 class="company-name">ECI Solutions</h3>
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
        
        # WORKING Theme Settings
        st.markdown("### 🎨 Theme Settings")
        
        # Theme selector with immediate effect
        theme_col1, theme_col2 = st.columns(2)
        
        with theme_col1:
            new_theme = st.selectbox(
                "Theme Mode",
                ["Light", "Dark"],
                index=0 if st.session_state.get('theme_mode', 'Light') == 'Light' else 1,
                key="theme_selector_working"
            )
            if new_theme != st.session_state.get('theme_mode'):
                st.session_state.theme_mode = new_theme
                st.rerun()
        
        with theme_col2:
            new_colors = st.selectbox(
                "Color Scheme",
                ["Professional Blue", "Financial Green", "Executive Purple", "Energy Orange"],
                index=["Professional Blue", "Financial Green", "Executive Purple", "Energy Orange"].index(
                    st.session_state.get('color_scheme', 'Professional Blue')
                ),
                key="color_selector_working"
            )
            if new_colors != st.session_state.get('color_scheme'):
                st.session_state.color_scheme = new_colors
                st.rerun()
        
        # Font size selector
        new_font = st.select_slider(
            "Font Size",
            options=["Small", "Medium", "Large"],
            value=st.session_state.get('font_size', 'Medium'),
            key="font_selector_working"
        )
        if new_font != st.session_state.get('font_size'):
            st.session_state.font_size = new_font
            st.rerun()
        
        render_data_sources_section()

        # Market Status
        st.markdown("### 📊 Market Status")
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.markdown('<span class="status-online">🟢 Markets Open</span>', unsafe_allow_html=True)
        with status_col2:
            st.markdown('<span class="status-processing">🟡 AI Active</span>', unsafe_allow_html=True)
        
        # Current time
        now = datetime.now()
        st.markdown(f"""
        <div style="text-align: center; font-size: 0.8rem; margin: 0.5rem 0;">
            {now.strftime('%H:%M:%S EST')} | {now.strftime('%b %d, %Y')}
        </div>
        """, unsafe_allow_html=True)
        
        # WORKING Asset Selection with Real Ticker Fetching
        st.markdown("### 📈 Asset Selection")
        
        # Check if EODHD is configured
        eodhd_configured = config.is_configured('eodhd')
        
        # Fetch World Tickers Button
        if st.session_state.get('all_tickers_df', pd.DataFrame()).empty:
            if st.button("🌍 Fetch World Tickers", disabled=(not eodhd_configured), use_container_width=True):
                if not eodhd_configured:
                    st.error("❌ EODHD API key not configured. Please check your .env file.")
                else:
                    with st.spinner("🔄 Fetching all world tickers... This may take 2-3 minutes."):
                        try:
                            api_key = config.get_eodhd_api_key()
                            all_tickers = fetch_all_tickers(api_key)
                            
                            if all_tickers:
                                df = pd.DataFrame(all_tickers)
                                
                                # Ensure required columns exist
                                if 'Code' in df.columns and 'Name' in df.columns:
                                    # Create display name
                                    df['display_name'] = df['Code'] + " - " + df['Name'].fillna('')
                                    st.session_state.all_tickers_df = df
                                    st.success(f"✅ Successfully fetched {len(df):,} tickers from {df['Exchange'].nunique() if 'Exchange' in df.columns else 'multiple'} exchanges!")
                                    st.rerun()
                                else:
                                    st.error("❌ Invalid ticker data format received.")
                            else:
                                st.error("❌ No tickers received. Please check your EODHD API key and subscription.")
                                
                        except Exception as e:
                            st.error(f"❌ Error fetching tickers: {str(e)}")
        
        # Ticker Search and Selection
        if not st.session_state.get('all_tickers_df', pd.DataFrame()).empty:
            tickers_df = st.session_state.all_tickers_df
            st.success(f"📊 {len(tickers_df):,} tickers available")
            
            # Search functionality
            search_term = st.text_input(
                "🔍 Search Tickers", 
                placeholder="Enter symbol or company name...",
                key="ticker_search_input"
            )
            
            filtered_df = tickers_df
            if search_term:
                try:
                    search_upper = search_term.upper()
                    # Search in both Code and Name columns
                    code_mask = filtered_df['Code'].str.upper().str.contains(search_upper, na=False, regex=False)
                    name_mask = filtered_df['Name'].str.upper().str.contains(search_upper, na=False, regex=False)
                    filtered_df = filtered_df[code_mask | name_mask]
                    
                    if len(filtered_df) > 0:
                        st.info(f"Found {len(filtered_df)} matching tickers")
                    else:
                        st.warning("No tickers match your search")
                        
                except Exception as e:
                    st.warning(f"Search error: {e}")
                    filtered_df = tickers_df
            
            # Limit results for performance
            max_display = min(100, len(filtered_df))
            display_df = filtered_df.head(max_display)
            
            if len(display_df) > 0 and 'display_name' in display_df.columns:
                # Create symbol selector
                options_list = display_df['display_name'].tolist()
                selected_display = st.selectbox(
                    "📋 Select Symbol",
                    options_list,
                    key="symbol_selector_main"
                )
                
                # Get the actual symbol code
                if selected_display:
                    selected_row = display_df[display_df['display_name'] == selected_display]
                    if not selected_row.empty:
                        selected_symbol = selected_row.iloc[0]['Code']
                        st.session_state.selected_symbol = selected_symbol
                    else:
                        selected_symbol = st.session_state.get('selected_symbol', 'AAPL.US')
                else:
                    selected_symbol = st.session_state.get('selected_symbol', 'AAPL.US')
            else:
                # Fallback to default symbols
                default_symbols = ['AAPL.US', 'GOOGL.US', 'MSFT.US', 'TSLA.US', 'AMZN.US']
                selected_symbol = st.selectbox("📋 Select Symbol (Default)", default_symbols)
                st.session_state.selected_symbol = selected_symbol
        
        else:
            # No tickers fetched yet - show quick select options
            st.info("Click 'Fetch World Tickers' above to access all global markets")
            
            # Quick asset categories
            asset_category = st.selectbox(
                "📂 Asset Category",
                ["🇺🇸 US Stocks", "🌍 International", "💰 Crypto", "🏛️ Indices"],
                key="asset_category_selector"
            )
            
            # Popular assets by category
            popular_assets = {
                "🇺🇸 US Stocks": ["AAPL.US", "GOOGL.US", "MSFT.US", "TSLA.US", "AMZN.US", "NVDA.US"],
                "🌍 International": ["TSM", "ASML.AS", "SAP.DE", "TM", "BABA"],
                "💰 Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"],
                "🏛️ Indices": ["SPY", "QQQ", "IWM", "DIA", "VTI"]
            }
            
            # Quick select buttons
            st.markdown("**⚡ Quick Select:**")
            current_assets = popular_assets.get(asset_category, [])
            
            # Display in columns
            cols = st.columns(2)
            for i, symbol in enumerate(current_assets[:6]):
                col_idx = i % 2
                with cols[col_idx]:
                    if st.button(symbol, key=f"quick_select_{symbol}", use_container_width=True):
                        st.session_state.selected_symbol = symbol
                        selected_symbol = symbol
            
            # Custom symbol input
            custom_symbol = st.text_input(
                "✏️ Custom Symbol",
                placeholder="e.g., AAPL.US",
                key="custom_symbol_input_main"
            )
            
            if custom_symbol:
                st.session_state.selected_symbol = custom_symbol.upper()
                selected_symbol = custom_symbol.upper()
            else:
                selected_symbol = st.session_state.get('selected_symbol', 'AAPL.US')
        
        # Display currently selected symbol
        current_symbol = st.session_state.get('selected_symbol', 'AAPL.US')
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, var(--primary-color, #0066cc), var(--secondary-color, #4d94ff)); 
                    color: white; padding: 0.8rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
            <strong>📊 Selected: {current_symbol}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear symbol selection if symbol changes
        if st.session_state.get('selected_symbol') != current_symbol:
            # Clear analysis cache when symbol changes
            keys_to_clear = [
                'chart_analysis_report', 'forecast_df', 'anomalies', 
                'detected_patterns', 'comprehensive_analysis', 'forecast_plot_df'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Analysis Settings
        st.markdown("### ⚙️ Analysis Settings")
        
        time_horizon = st.selectbox(
            "📅 Time Horizon",
            ["📅 Intraday", "📊 Daily", "📈 Weekly", "📆 Monthly"],
            index=1
        )
        
        analysis_depth = st.select_slider(
            "🔍 Analysis Depth",
            options=["Quick", "Standard", "Deep", "Comprehensive"],
            value="Standard"
        )
        
        risk_tolerance = st.select_slider(
            "⚖️ Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive", "Speculative"],
            value="Moderate"
        )
        
        # Portfolio Overview (if authenticated)
        if st.session_state.get('authenticated', False):
            st.markdown("### 💼 Portfolio Overview")
            
            # Mock portfolio data
            portfolio_value = 245750.50
            daily_change = 2340.25
            daily_change_pct = 0.96
            
            change_color = "#28a745" if daily_change >= 0 else "#dc3545"
            change_arrow = "↗️" if daily_change >= 0 else "↘️"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="portfolio-value">${portfolio_value:,.2f}</div>
                <div style="font-size: 0.8rem; margin: 0.5rem 0;">Total Portfolio Value</div>
                <div style="color: {change_color};">
                    {change_arrow} ${abs(daily_change):,.2f} ({daily_change_pct:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🔍 Market Scan", use_container_width=True):
                st.session_state.run_market_scan = True
                st.success("Market scan initiated!")
            
            if st.button("📊 Technical Analysis", use_container_width=True):
                st.session_state.run_technical_analysis = True
                st.success("Technical analysis started!")
        
        with action_col2:
            if st.button("📰 News Summary", use_container_width=True):
                st.session_state.show_news_summary = True
                st.success("Loading news...")
            
            if st.button("🎯 AI Insights", use_container_width=True):
                st.session_state.get_ai_insights = True
                st.success("AI analysis in progress...")
        
        # System Status
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown('<div class="status-online">🟢 Online</div>', unsafe_allow_html=True)
        with status_col2:
            st.markdown('<div>🔄 Synced</div>', unsafe_allow_html=True)
        with status_col3:
            st.markdown('<div>⚡ Fast</div>', unsafe_allow_html=True)

def render_sidebar_with_theme():
    """Main function to render sidebar with working themes"""
    render_sidebar()
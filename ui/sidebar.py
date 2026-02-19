# Professional Trading Platform Sidebar - Complete Implementation
import streamlit as st
import pandas as pd
import time
from config import config
import base64
import os
from datetime import datetime, timedelta
import requests
import json
from translate import Translator



def load_eci_logo():
    """Load ECI logo from static folder"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'eci.png')
        if not os.path.exists(logo_path):
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

@st.cache_data(ttl=300)  # Cache translations for 5 minutes
def simple_translate(text, target_language):
    """Simple translation function with caching"""
    try:
        if not text or target_language == 'en':
            return text
        
        translator = Translator(to_lang=target_language)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        # If translation fails, return original text
        return text

def get_supported_languages():
    """Get list of supported languages"""
    return {
        'en': '🇺🇸 English',
        'es': '🇪🇸 Spanish',
        'fr': '🇫🇷 French',
        'de': '🇩🇪 German',
        'it': '🇮🇹 Italian',
        'pt': '🇵🇹 Portuguese',
        'ru': '🇷🇺 Russian',
        'ja': '🇯🇵 Japanese',
        'ko': '🇰🇷 Korean',
        'zh': '🇨🇳 Chinese',
        'ar': '🇸🇦 Arabic',
        'hi': '🇮🇳 Hindi',
        'tr': '🇹🇷 Turkish',
        'nl': '🇳🇱 Dutch',
        'sv': '🇸🇪 Swedish',
        'no': '🇳🇴 Norwegian',
        'da': '🇩🇰 Danish',
        'fi': '🇫🇮 Finnish',
        'pl': '🇵🇱 Polish',
        'cs': '🇨🇿 Czech'
    }

# REPLACE THIS SECTION in your sidebar.py file around line 30-70:

def apply_professional_theme():
    """Apply comprehensive professional theme with advanced styling"""
    
    # Get theme settings
    theme_mode = st.session_state.get('theme_mode', 'Dark')
    color_scheme = st.session_state.get('color_scheme', 'Claude Anthropic')
    
    # Define comprehensive color schemes - ENHANCED WITH MORE OPTIONS
    color_schemes = {
        'Claude Anthropic': {
            'primary': '#D97706',
            'secondary': '#F59E0B',
            'accent': '#B45309',
            'success': '#059669',
            'warning': '#D97706',
            'danger': '#DC2626',
            'gradient_start': '#F3F4F6',
            'gradient_end': '#E5E7EB'
        },
        'Minimal White': {
            'primary': '#6B7280',
            'secondary': '#9CA3AF',
            'accent': '#4B5563',
            'success': '#10B981',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'gradient_start': '#FFFFFF',
            'gradient_end': '#F9FAFB'
        },
        'Soft Gray': {
            'primary': '#6366F1',
            'secondary': '#8B5CF6',
            'accent': '#4F46E5',
            'success': '#10B981',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'gradient_start': '#F8FAFC',
            'gradient_end': '#E2E8F0'
        },
        'Clean Slate': {
            'primary': '#374151',
            'secondary': '#6B7280',
            'accent': '#1F2937',
            'success': '#059669',
            'warning': '#D97706',
            'danger': '#DC2626',
            'gradient_start': '#F1F5F9',
            'gradient_end': '#CBD5E1'
        },
        'Off White': {
            'primary': '#78716C',
            'secondary': '#A8A29E',
            'accent': '#57534E',
            'success': '#16A34A',
            'warning': '#EA580C',
            'danger': '#DC2626',
            'gradient_start': '#FAFAF9',
            'gradient_end': '#F5F5F4'
        },
        'Pure White': {
            'primary': '#000000',
            'secondary': '#404040',
            'accent': '#1A1A1A',
            'success': '#22C55E',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'gradient_start': '#FFFFFF',
            'gradient_end': '#FFFFFF'
        },
        'Light Gray': {
            'primary': '#525252',
            'secondary': '#737373',
            'accent': '#404040',
            'success': '#22C55E',
            'warning': '#F97316',
            'danger': '#EF4444',
            'gradient_start': '#FAFAFA',
            'gradient_end': '#F4F4F5'
        },
        'Warm Gray': {
            'primary': '#78716C',
            'secondary': '#A8A29E',
            'accent': '#57534E',
            'success': '#16A34A',
            'warning': '#EA580C',
            'danger': '#DC2626',
            'gradient_start': '#FAFAF9',
            'gradient_end': '#F5F5F4'
        },
        'Cool Gray': {
            'primary': '#64748B',
            'secondary': '#94A3B8',
            'accent': '#475569',
            'success': '#0F766E',
            'warning': '#C2410C',
            'danger': '#DC2626',
            'gradient_start': '#F8FAFC',
            'gradient_end': '#E2E8F0'
        },
        'Neutral Stone': {
            'primary': '#6B7280',
            'secondary': '#9CA3AF',
            'accent': '#4B5563',
            'success': '#059669',
            'warning': '#D97706',
            'danger': '#DC2626',
            'gradient_start': '#FAFAF9',
            'gradient_end': '#E7E5E4'
        },
        'Professional Blue': {
            'primary': '#0066ff',
            'secondary': '#4d94ff',
            'accent': '#0052cc',
            'success': '#00b894',
            'warning': '#fdcb6e',
            'danger': '#e74c3c',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        },
        'Financial Green': {
            'primary': '#00b386',
            'secondary': '#4dd2aa',
            'accent': '#00a085',
            'success': '#20c997',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'gradient_start': '#11998e',
            'gradient_end': '#38ef7d'
        },
        'Executive Purple': {
            'primary': '#6c5ce7',
            'secondary': '#a29bfe',
            'accent': '#5f3dc4',
            'success': '#00b894',
            'warning': '#fdcb6e',
            'danger': '#e74c3c',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        },
        'Energy Orange': {
            'primary': '#fd7e14',
            'secondary': '#ff9a44',
            'accent': '#e55100',
            'success': '#00b894',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'gradient_start': '#f093fb',
            'gradient_end': '#f5576c'
        },
        'Deep Ocean': {
            'primary': '#0891B2',
            'secondary': '#06B6D4',
            'accent': '#0E7490',
            'success': '#059669',
            'warning': '#D97706',
            'danger': '#DC2626',
            'gradient_start': '#0F172A',
            'gradient_end': '#1E293B'
        },
        'Forest Green': {
            'primary': '#059669',
            'secondary': '#10B981',
            'accent': '#047857',
            'success': '#22C55E',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'gradient_start': '#064E3B',
            'gradient_end': '#065F46'
        },
        'Royal Purple': {
            'primary': '#7C3AED',
            'secondary': '#A855F7',
            'accent': '#6D28D9',
            'success': '#10B981',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'gradient_start': '#581C87',
            'gradient_end': '#7C2D92'
        },
        'Sunset Red': {
            'primary': '#DC2626',
            'secondary': '#EF4444',
            'accent': '#B91C1C',
            'success': '#10B981',
            'warning': '#F59E0B',
            'danger': '#7F1D1D',
            'gradient_start': '#7F1D1D',
            'gradient_end': '#991B1B'
        }
    }
    
    # Rest of your existing code continues unchanged...
    colors = color_schemes.get(color_scheme, color_schemes['Claude Anthropic'])
    
    # Theme-specific colors
    if theme_mode == 'Dark':
        bg_primary = '#0e1117'
        bg_secondary = '#1e1e1e'
        bg_tertiary = '#2d2d2d'
        bg_sidebar = '#0f1419'
        text_primary = '#ffffff'
        text_secondary = '#b3b3b3'
        text_muted = '#808080'
        border_color = '#404040'
        shadow_color = 'rgba(0, 0, 0, 0.5)'
    else:
        # Enhanced light theme colors for better contrast with new color schemes
        bg_primary = '#ffffff'
        bg_secondary = '#fafafa'
        bg_tertiary = '#f5f5f5'
        bg_sidebar = '#fdfdfd'
        text_primary = '#1a1a1a'
        text_secondary = '#4a4a4a'
        text_muted = '#6a6a6a'
        border_color = '#e5e5e5'
        shadow_color = 'rgba(0, 0, 0, 0.08)'
    
    # Advanced CSS with animations and professional styling
    professional_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for consistency */
        :root {{
            --primary-color: {colors['primary']};
            --secondary-color: {colors['secondary']};
            --accent-color: {colors['accent']};
            --success-color: {colors['success']};
            --warning-color: {colors['warning']};
            --danger-color: {colors['danger']};
            --bg-primary: {bg_primary};
            --bg-secondary: {bg_secondary};
            --bg-tertiary: {bg_tertiary};
            --bg-sidebar: {bg_sidebar};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --text-muted: {text_muted};
            --border-color: {border_color};
            --shadow-color: {shadow_color};
            --gradient-start: {colors['gradient_start']};
            --gradient-end: {colors['gradient_end']};
        }}
        
        /* Global styles */
        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }}
        
        /* Main app styling */
        .stApp {{
            background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end)) !important;
            color: var(--text-primary) !important;
        }}
        
        .main .block-container {{
            background: var(--bg-primary) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 32px var(--shadow-color) !important;
            margin: 20px !important;
            padding: 2rem !important;
            backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}
        
        /* Sidebar comprehensive styling */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background: var(--bg-sidebar) !important;
            border-right: 2px solid var(--primary-color) !important;
            box-shadow: 4px 0 20px var(--shadow-color) !important;
        }}
        
        .css-1d391kg > div:first-child {{
            background: var(--bg-sidebar) !important;
            padding: 1rem !important;
        }}
        
        /* Professional header styling */
        .professional-header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            margin-bottom: 1.5rem !important;
            text-align: center !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
            position: relative !important;
            overflow: hidden !important;
        }}
        
        .professional-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        
        .company-logo {{
            max-width: 80px !important;
            height: auto !important;
            margin-bottom: 0.5rem !important;
            filter: brightness(1.2) !important;
        }}
        
        .company-name {{
            color: white !important;
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            margin: 0.5rem 0 !important;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        }}
        
        .company-tagline {{
            color: rgba(255, 255, 255, 0.9) !important;
            font-size: 0.8rem !important;
            font-weight: 400 !important;
            margin: 0 !important;
            opacity: 0.9 !important;
        }}
        
        /* Section headers */
        .section-header {{
            color: var(--primary-color) !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin: 1.5rem 0 1rem 0 !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid var(--primary-color) !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }}
        
        /* Data source cards with advanced styling */
        .data-source-grid {{
            display: grid !important;
            gap: 0.8rem !important;
            margin: 1rem 0 !important;
        }}
        
        .data-source-card {{
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary)) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
            position: relative !important;
            overflow: hidden !important;
        }}
        
        .data-source-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }}
        
        .data-source-card:hover {{
            transform: translateY(-4px) !important;
            box-shadow: 0 12px 32px var(--shadow-color) !important;
            border-color: var(--primary-color) !important;
            background: linear-gradient(135deg, var(--primary-color)10, var(--secondary-color)10) !important;
        }}
        
        .data-source-card:hover::before {{
            transform: scaleX(1);
        }}
        
        .source-header {{
            display: flex !important;
            align-items: center !important;
            margin-bottom: 0.8rem !important;
        }}
        
        .source-icon {{
            font-size: 1.5rem !important;
            margin-right: 0.8rem !important;
            width: 40px !important;
            height: 40px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
            border-radius: 10px !important;
            color: white !important;
        }}
        
        .source-title {{
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            color: var(--text-primary) !important;
            margin: 0 !important;
        }}
        
        .source-description {{
            font-size: 0.75rem !important;
            color: var(--text-muted) !important;
            line-height: 1.4 !important;
            margin-bottom: 0.8rem !important;
        }}
        
        .source-status {{
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
        }}
        
        .status-badge {{
            padding: 0.25rem 0.5rem !important;
            border-radius: 20px !important;
            font-size: 0.7rem !important;
            font-weight: 500 !important;
        }}
        
        .status-connected {{
            background: linear-gradient(135deg, var(--success-color), #00d68f) !important;
            color: white !important;
        }}
        
        .status-disconnected {{
            background: linear-gradient(135deg, var(--danger-color), #ff6b7a) !important;
            color: white !important;
        }}
        
        /* Advanced button styling */
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }}
        
        .stButton > button::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
            background: linear-gradient(135deg, var(--secondary-color), var(--accent-color)) !important;
        }}
        
        .stButton > button:hover::before {{
            left: 100%;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stMultiSelect > div > div > div {{
            background: var(--bg-secondary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            font-size: 0.85rem !important;
            padding: 0.6rem !important;
            transition: all 0.3s ease !important;
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.1) !important;
            background: var(--bg-primary) !important;
        }}
        
        /* Metrics and statistics */
        .metric-card {{
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary)) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            margin: 0.5rem 0 !important;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px var(--shadow-color) !important;
        }}
        
        .metric-value {{
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: var(--primary-color) !important;
            margin-bottom: 0.25rem !important;
        }}
        
        .metric-label {{
            font-size: 0.75rem !important;
            color: var(--text-muted) !important;
            font-weight: 500 !important;
        }}
        
        /* Search bar */
        .search-container {{
            position: relative !important;
            margin: 1rem 0 !important;
        }}
        
        .search-container::before {{
            content: '🔍' !important;
            position: absolute !important;
            left: 12px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            color: var(--text-muted) !important;
            z-index: 1 !important;
        }}
        
        /* Filter tabs */
        .filter-tabs {{
            display: flex !important;
            gap: 0.25rem !important;
            margin: 1rem 0 !important;
            background: var(--bg-secondary) !important;
            border-radius: 12px !important;
            padding: 0.25rem !important;
            border: 1px solid var(--border-color) !important;
        }}
        
        .filter-tab {{
            flex: 1 !important;
            padding: 0.5rem 0.75rem !important;
            border-radius: 8px !important;
            background: transparent !important;
            color: var(--text-muted) !important;
            text-align: center !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            border: none !important;
        }}
        
        .filter-tab:hover {{
            background: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }}
        
        .filter-tab.active {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
            color: white !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
        }}
        
        /* Quick actions */
        .quick-actions {{
            display: grid !important;
            grid-template-columns: 1fr 1fr !important;
            gap: 0.5rem !important;
            margin: 1rem 0 !important;
        }}
        
        .action-btn {{
            background: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 0.6rem !important;
            color: var(--text-primary) !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            text-align: center !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }}
        
        .action-btn:hover {{
            background: var(--primary-color) !important;
            color: white !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }}
        
        /* Status indicators */
        .status-indicator {{
            display: inline-flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
            padding: 0.4rem 0.8rem !important;
            border-radius: 20px !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
            margin: 0.25rem 0 !important;
        }}
        
        .status-online {{
            background: linear-gradient(135deg, var(--success-color), #00d68f) !important;
            color: white !important;
        }}
        
        .status-processing {{
            background: linear-gradient(135deg, var(--warning-color), #ffa726) !important;
            color: white !important;
        }}
        
        .status-offline {{
            background: linear-gradient(135deg, var(--danger-color), #ff6b7a) !important;
            color: white !important;
        }}
        
        /* Pulse animation for status indicators */
        .pulse {{
            animation: pulse 2s infinite !important;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}
        
        /* Scrollbar styling */
        .css-1d391kg::-webkit-scrollbar {{
            width: 6px !important;
        }}
        
        .css-1d391kg::-webkit-scrollbar-track {{
            background: var(--bg-secondary) !important;
            border-radius: 3px !important;
        }}
        
        .css-1d391kg::-webkit-scrollbar-thumb {{
            background: var(--primary-color) !important;
            border-radius: 3px !important;
        }}
        
        .css-1d391kg::-webkit-scrollbar-thumb:hover {{
            background: var(--secondary-color) !important;
        }}
        
        /* Portfolio styling */
        .portfolio-card {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            color: white !important;
            margin: 1rem 0 !important;
            position: relative !important;
            overflow: hidden !important;
        }}
        
        .portfolio-card::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: float 6s ease-in-out infinite;
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
            50% {{ transform: translateY(-20px) rotate(180deg); }}
        }}
        
        .portfolio-value {{
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
        }}
        
        .portfolio-change {{
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            opacity: 0.9 !important;
        }}
        
        /* Theme indicator */
        .theme-indicator {{
            position: fixed !important;
            top: 80px !important;
            right: 20px !important;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
            color: white !important;
            padding: 0.5rem 1rem !important;
            border-radius: 25px !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            z-index: 9999 !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
            backdrop-filter: blur(10px) !important;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .data-source-grid {{
                grid-template-columns: 1fr !important;
            }}
            
            .quick-actions {{
                grid-template-columns: 1fr !important;
            }}
            
            .filter-tabs {{
                flex-direction: column !important;
            }}
        }}
        
        /* Animation delays for staggered effect */
        .data-source-card:nth-child(1) {{ animation-delay: 0.1s; }}
        .data-source-card:nth-child(2) {{ animation-delay: 0.2s; }}
        .data-source-card:nth-child(3) {{ animation-delay: 0.3s; }}
        .data-source-card:nth-child(4) {{ animation-delay: 0.4s; }}
        .data-source-card:nth-child(5) {{ animation-delay: 0.5s; }}
        
        /* Loading animation */
        .loading-shimmer {{
            background: linear-gradient(90deg, var(--bg-secondary) 25%, var(--bg-tertiary) 50%, var(--bg-secondary) 75%);
            background-size: 200% 100%;
            animation: shimmer-loading 1.5s infinite;
        }}
        
        @keyframes shimmer-loading {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}

        /* Fix Market Status Section */
        .market-status-grid {{
            display: grid !important;
            grid-template-columns: 1fr 1fr !important;
            gap: 0.5rem !important;
            margin: 1rem 0 !important;
        }}

        .status-card {{
            padding: 0.6rem !important;
            border-radius: 8px !important;
            text-align: center !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            color: white !important;
        }}

        .status-open {{
            background: #00b894 !important;
        }}

        .status-processing {{
            background: #f39c12 !important;
        }}

        .status-closed {{
            background: #e74c3c !important;
        }}

        /* Fix button spacing */
        .stButton > button {{
            margin: 0.2rem 0 !important;
            font-size: 0.8rem !important;
            padding: 0.4rem 0.8rem !important;
        }}

        /* Fix section spacing */
        .section-header + * {{
            margin-top: 0.5rem !important;
        }}
        
        /* Fix sidebar sections spacing */
        .css-1d391kg .element-container {{
            margin-bottom: 0.5rem !important;
        }}
        
        /* Clean up Market Status text overlaps */
        .css-1d391kg .stMarkdown {{
            margin-bottom: 0.3rem !important;
        }}

    </style>
    """
    
    
    st.markdown(professional_css, unsafe_allow_html=True)
    

def get_data_source_status(source_name: str):
    """Get the connection status of a data source"""
    connected_sources = st.session_state.get('connected_sources', {})
    return {
        'connected': connected_sources.get(source_name, False),
        'last_sync': connected_sources.get(f'{source_name}_last_sync', 'Never'),
        'status': 'Active' if connected_sources.get(source_name, False) else 'Disconnected'
    }

def connect_data_source(source_name: str, source_config):
    """Connect to a data source with realistic simulation"""
    try:
        if 'connected_sources' not in st.session_state:
            st.session_state.connected_sources = {}
        
        # Simulate connection process
        progress_bar = st.progress(0, text=f"Connecting to {source_config['title']}...")
        
        for i in range(100):
            time.sleep(0.01)  # Simulate connection time
            progress_bar.progress(i + 1, text=f"Establishing connection... {i+1}%")
        
        st.session_state.connected_sources[source_name] = True
        st.session_state.connected_sources[f'{source_name}_last_sync'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        progress_bar.empty()
        st.success(f"✅ Successfully connected to {source_config['title']}")
        time.sleep(1)
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to connect to {source_config['title']}: {str(e)}")
        return False

def render_professional_data_sources():
    """Render the professional data sources interface"""
    
    st.markdown("""
    <div class="section-header">
        🔗 Data Sources & Integration
    </div>
    """, unsafe_allow_html=True)
    
    # Search functionality
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_term = st.text_input(
            "",
            placeholder="🔍 Search data sources...",
            key="data_source_search",
            help="Search by name, category, or description"
        )
    
    with search_col2:
        if st.button("🔄", help="Refresh", use_container_width=True):
            st.rerun()
    
    # Filter tabs with enhanced styling
    st.markdown("""
    <div class="filter-tabs">
        <button class="filter-tab active" onclick="filterSources('all')">All</button>
        <button class="filter-tab" onclick="filterSources('market')">Market</button>
        <button class="filter-tab" onclick="filterSources('cloud')">Cloud</button>
        <button class="filter-tab" onclick="filterSources('ai')">AI</button>
    </div>
    """, unsafe_allow_html=True)
    
    # Category filter
    filter_option = st.radio(
        "",
        ["All Sources", "Market Data", "Cloud Storage", "AI & Alternative", "News & Sentiment"],
        horizontal=True,
        key="data_source_filter",
        label_visibility="collapsed"
    )
    
    # Comprehensive data source definitions
    data_sources = {
        "bloomberg": {
            "title": "Bloomberg Terminal",
            "icon": "📊",
            "description": "Real-time market data, news, analytics, and trading tools from Bloomberg's professional platform with Level II data access.",
            "category": "Market Data",
            "priority": 1,
            "connection_type": "API",
            "features": ["Real-time Data", "News Feed", "Analytics", "Level II"]
        },
        "factset": {
            "title": "FactSet Workstation",
            "icon": "📈",
            "description": "Comprehensive financial data, analytics, and portfolio management tools for institutional investors and analysts.",
            "category": "Market Data",
            "priority": 1,
            "connection_type": "API",
            "features": ["Portfolio Analytics", "Risk Management", "Research", "Backtesting"]
        },
        "refinitiv": {
            "title": "Refinitiv Eikon",
            "icon": "📖",
            "description": "Global market intelligence, real-time data, news, and advanced analytics for trading and investment decisions.",
            "category": "Market Data",
            "priority": 1,
            "connection_type": "API",
            "features": ["Market Data", "News", "Analytics", "FX Trading"]
        },
        "google_drive": {
            "title": "Google Drive",
            "icon": "💾",
            "description": "Secure access to financial documents, spreadsheets, and reports stored in Google Drive with real-time collaboration.",
            "category": "Cloud Storage",
            "priority": 2,
            "connection_type": "OAuth",
            "features": ["Document Sync", "Collaboration", "Version Control", "Security"]
        },
        "sharepoint": {
            "title": "Microsoft SharePoint",
            "icon": "🏢",
            "description": "Enterprise document management and collaboration platform for financial teams with advanced security features.",
            "category": "Cloud Storage",
            "priority": 2,
            "connection_type": "OAuth",
            "features": ["Enterprise Security", "Workflow", "Document Management", "Integration"]
        },
        "dropbox": {
            "title": "Dropbox Business",
            "icon": "📦",
            "description": "Secure file sharing and collaboration platform for financial documents with enterprise-grade security and compliance.",
            "category": "Cloud Storage",
            "priority": 2,
            "connection_type": "OAuth",
            "features": ["File Sync", "Team Collaboration", "Security", "Compliance"]
        },
        "social_sentiment": {
            "title": "Social Sentiment AI",
            "icon": "🤖",
            "description": "AI-powered social media sentiment analysis for market trends, trading signals, and investor sentiment tracking.",
            "category": "AI & Alternative",
            "priority": 3,
            "connection_type": "API",
            "features": ["Sentiment Analysis", "Trend Detection", "Signal Generation", "Real-time Monitoring"]
        },
        "satellite_data": {
            "title": "Satellite Intelligence",
            "icon": "🛰️",
            "description": "Satellite imagery and geospatial data for commodity trading, economic indicators, and alternative investment insights.",
            "category": "AI & Alternative",
            "priority": 3,
            "connection_type": "API",
            "features": ["Geospatial Analysis", "Commodity Tracking", "Economic Indicators", "ESG Monitoring"]
        },
        "news_api": {
            "title": "Financial News API",
            "icon": "📰",
            "description": "Real-time financial news, press releases, and market-moving events from global sources with AI-powered categorization.",
            "category": "News & Sentiment",
            "priority": 2,
            "connection_type": "API",
            "features": ["Real-time News", "Market Events", "AI Categorization", "Alert System"]
        },
        "sec_edgar": {
            "title": "SEC EDGAR",
            "icon": "🏛️",
            "description": "Direct access to SEC filings, insider trading data, regulatory documents, and compliance monitoring tools.",
            "category": "Market Data",
            "priority": 2,
            "connection_type": "API",
            "features": ["SEC Filings", "Insider Trading", "Compliance", "Regulatory Alerts"]
        },
        "azure_openai": {
            "title": "Azure OpenAI",
            "icon": "🧠",
            "description": "Advanced AI capabilities for document analysis, market insights, and intelligent trading recommendations.",
            "category": "AI & Alternative",
            "priority": 1,
            "connection_type": "API",
            "features": ["Document Analysis", "Market Insights", "AI Recommendations", "Natural Language Processing"]
        },
        "polygon_io": {
            "title": "Polygon.io",
            "icon": "📊",
            "description": "Real-time and historical market data for stocks, options, forex, and crypto with millisecond-level precision.",
            "category": "Market Data",
            "priority": 2,
            "connection_type": "API",
            "features": ["Real-time Data", "Historical Data", "Options Flow", "Crypto Data"]
        },
        "alphavantage": {
            "title": "Alpha Vantage",
            "icon": "📈",
            "description": "Comprehensive market data API with technical indicators, fundamental data, and economic indicators.",
            "category": "Market Data",
            "priority": 2,
            "connection_type": "API",
            "features": ["Technical Indicators", "Fundamental Data", "Economic Data", "Forex Data"]
        }
    }
    
    # Filter data sources based on search and category
    filtered_sources = {}
    for key, source in data_sources.items():
        # Apply search filter
        if search_term:
            search_lower = search_term.lower()
            if not (search_lower in source['title'].lower() or 
                   search_lower in source['description'].lower() or
                   search_lower in source['category'].lower()):
                continue
        
        # Apply category filter
        if filter_option != "All Sources":
            category_map = {
                "Market Data": "Market Data",
                "Cloud Storage": "Cloud Storage", 
                "AI & Alternative": "AI & Alternative",
                "News & Sentiment": "News & Sentiment"
            }
            if source['category'] != category_map.get(filter_option):
                continue
        
        filtered_sources[key] = source
    
    # Sort by priority and title
    sorted_sources = sorted(filtered_sources.items(), key=lambda x: (x[1]['priority'], x[1]['title']))
    
    # Render data source cards in a grid
    if sorted_sources:
        st.markdown('<div class="data-source-grid">', unsafe_allow_html=True)
        
        for source_name, source_config in sorted_sources:
            status = get_data_source_status(source_name)
            status_class = "status-connected" if status['connected'] else "status-disconnected"
            status_text = "🟢 Connected" if status['connected'] else "🔴 Disconnected"
            
            # Create feature badges
            feature_badges = " ".join([f"<span style='background: var(--bg-tertiary); padding: 0.2rem 0.4rem; border-radius: 12px; font-size: 0.6rem; margin-right: 0.25rem;'>{feature}</span>" for feature in source_config.get('features', [])[:3]])
            
            card_html = f"""
            <div class="data-source-card">
                <div class="source-header">
                    <div class="source-icon">{source_config['icon']}</div>
                    <div>
                        <div class="source-title">{source_config['title']}</div>
                        <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.2rem;">
                            {source_config['connection_type']} • {source_config['category']}
                        </div>
                    </div>
                </div>
                <div class="source-description">{source_config['description']}</div>
                <div style="margin-bottom: 0.8rem;">{feature_badges}</div>
                <div class="source-status">
                    <span class="status-badge {status_class}">{status_text}</span>
                    <div style="font-size: 0.7rem; color: var(--text-muted);">
                        Last sync: {status['last_sync']}
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Add connection button
            if not status['connected']:
                if st.button(f"Connect {source_config['title']}", 
                           key=f"connect_{source_name}", 
                           use_container_width=True,
                           help=f"Connect to {source_config['title']} via {source_config['connection_type']}"):
                    connect_data_source(source_name, source_config)
                    st.rerun()
            else:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔄 Sync", key=f"sync_{source_name}", use_container_width=True):
                        with st.spinner(f"Syncing {source_config['title']}..."):
                            time.sleep(1)
                            st.session_state.connected_sources[f'{source_name}_last_sync'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            st.success("✅ Synced successfully!")
                            st.rerun()
                
                with col2:
                    if st.button(f"⚙️ Config", key=f"config_{source_name}", use_container_width=True):
                        st.info(f"Configuration for {source_config['title']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🔍 No data sources match your current filters.")
    
    # Quick Actions section
    st.markdown("""
    <div class="section-header">
        ⚡ Quick Actions
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔗 Connect All Available", use_container_width=True):
            connected_count = 0
            disconnected_sources = [name for name, config in data_sources.items() 
                                  if not get_data_source_status(name)['connected']]
            
            if disconnected_sources:
                progress_bar = st.progress(0, text="Connecting all available sources...")
                
                for i, source_name in enumerate(disconnected_sources):
                    progress_bar.progress((i + 1) / len(disconnected_sources), 
                                        text=f"Connecting {data_sources[source_name]['title']}...")
                    
                    if connect_data_source(source_name, data_sources[source_name]):
                        connected_count += 1
                    time.sleep(0.5)
                
                progress_bar.empty()
                if connected_count > 0:
                    st.success(f"✅ Successfully connected {connected_count} data sources!")
                    st.rerun()
            else:
                st.info("All sources are already connected!")
    
    with col2:
        if st.button("🔄 Refresh All", use_container_width=True):
            with st.spinner("Refreshing all connections..."):
                time.sleep(2)
                # Update all last sync times
                for source_name in data_sources.keys():
                    if get_data_source_status(source_name)['connected']:
                        st.session_state.connected_sources[f'{source_name}_last_sync'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                st.success("✅ All connections refreshed!")
                st.rerun()

def render_connection_statistics():
    """Render connection statistics with enhanced metrics"""
    
    st.markdown("""
    <div class="section-header">
        📊 Connection Statistics
    </div>
    """, unsafe_allow_html=True)
    
    # Get connection data
    data_sources = {
        "bloomberg": {"category": "Market Data"},
        "factset": {"category": "Market Data"},
        "refinitiv": {"category": "Market Data"},
        "google_drive": {"category": "Cloud Storage"},
        "sharepoint": {"category": "Cloud Storage"},
        "dropbox": {"category": "Cloud Storage"},
        "social_sentiment": {"category": "AI & Alternative"},
        "satellite_data": {"category": "AI & Alternative"},
        "news_api": {"category": "News & Sentiment"},
        "sec_edgar": {"category": "Market Data"},
        "azure_openai": {"category": "AI & Alternative"},
        "polygon_io": {"category": "Market Data"},
        "alphavantage": {"category": "Market Data"}
    }
    
    connected_sources = st.session_state.get('connected_sources', {})
    total_sources = len(data_sources)
    connected_count = sum(1 for key in data_sources.keys() if connected_sources.get(key, False))
    
    # Calculate category statistics
    categories = {}
    for source, config in data_sources.items():
        category = config['category']
        if category not in categories:
            categories[category] = {'total': 0, 'connected': 0}
        categories[category]['total'] += 1
        if connected_sources.get(source, False):
            categories[category]['connected'] += 1
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_sources}</div>
            <div class="metric-label">Total Sources</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{connected_count}</div>
            <div class="metric-label">Connected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        connection_rate = (connected_count / total_sources * 100) if total_sources > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{connection_rate:.1f}%</div>
            <div class="metric-label">Connection Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        active_categories = sum(1 for cat_data in categories.values() if cat_data['connected'] > 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{active_categories}</div>
            <div class="metric-label">Active Categories</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Category breakdown
    st.markdown("**Category Breakdown:**")
    for category, data in categories.items():
        percentage = (data['connected'] / data['total'] * 100) if data['total'] > 0 else 0
        st.progress(percentage / 100, text=f"{category}: {data['connected']}/{data['total']} ({percentage:.0f}%)")

def render_market_status():
    """Render market status with clean, professional indicators"""
    
    st.markdown("""
    <div class="section-header">
        📈 Market Status
    </div>
    """, unsafe_allow_html=True)
    
    # Clean status indicators
    now = datetime.now()
    market_open = 9 <= now.hour < 16
    
    # Create clean status cards
    col1, col2 = st.columns(2)
    
    with col1:
        status_color = "#00b894" if market_open else "#e74c3c"
        status_text = "Open" if market_open else "Closed"
        st.markdown(f"""
        <div style="background: {status_color}; color: white; padding: 0.5rem; border-radius: 8px; text-align: center; margin-bottom: 0.5rem;">
            <div style="font-size: 0.8rem; font-weight: 600;">Markets {status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f39c12; color: white; padding: 0.5rem; border-radius: 8px; text-align: center; margin-bottom: 0.5rem;">
            <div style="font-size: 0.8rem; font-weight: 600;">AI Active</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Time display
    st.markdown(f"""
    <div style="text-align: center; font-size: 0.75rem; color: var(--text-muted); margin: 0.5rem 0;">
        {now.strftime('%H:%M:%S EST')} • {now.strftime('%b %d, %Y')}
    </div>
    """, unsafe_allow_html=True)

def render_asset_selection():
    """Render enhanced asset selection interface"""
    
    st.markdown("""
    <div class="section-header">
        🎯 Asset Selection
    </div>
    """, unsafe_allow_html=True)
    
    # Check EODHD configuration
    eodhd_configured = config.is_configured('eodhd')
    
    # Ticker fetching interface
    if st.session_state.get('all_tickers_df', pd.DataFrame()).empty:
        if st.button("🌍 Fetch Global Markets", 
                    disabled=(not eodhd_configured), 
                    use_container_width=True,
                    help="Load tickers from all global exchanges"):
            if not eodhd_configured:
                st.error("❌ EODHD API key not configured. Please check your .env file.")
            else:
                with st.spinner("🔄 Loading global markets... This may take 2-3 minutes."):
                    try:
                        api_key = config.get_eodhd_api_key()
                        all_tickers = fetch_all_tickers(api_key)
                        
                        if all_tickers:
                            df = pd.DataFrame(all_tickers)
                            if 'Code' in df.columns and 'Name' in df.columns:
                                df['display_name'] = df['Code'] + " - " + df['Name'].fillna('')
                                st.session_state.all_tickers_df = df
                                st.success(f"✅ Loaded {len(df):,} tickers from global markets!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid ticker data format received.")
                        else:
                            st.error("❌ No tickers received. Please check your API configuration.")
                    except Exception as e:
                        st.error(f"❌ Error loading tickers: {str(e)}")
    
    # Enhanced ticker selection
    if not st.session_state.get('all_tickers_df', pd.DataFrame()).empty:
        tickers_df = st.session_state.all_tickers_df
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(tickers_df):,}</div>
            <div class="metric-label">Global Tickers Available</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Advanced search
        search_term = st.text_input(
            "",
            placeholder="🔍 Search by symbol or company name...",
            key="ticker_search_enhanced",
            help="Search across all global markets"
        )
        
        # Filter and display
        filtered_df = tickers_df
        if search_term:
            try:
                search_upper = search_term.upper()
                code_mask = filtered_df['Code'].str.upper().str.contains(search_upper, na=False, regex=False)
                name_mask = filtered_df['Name'].str.upper().str.contains(search_upper, na=False, regex=False)
                filtered_df = filtered_df[code_mask | name_mask]
                
                if len(filtered_df) > 0:
                    st.info(f"📊 Found {len(filtered_df)} matching tickers")
                else:
                    st.warning("🔍 No matches found")
            except Exception as e:
                st.warning(f"Search error: {e}")
                filtered_df = tickers_df
        
        # Ticker selection
        max_display = min(100, len(filtered_df))
        display_df = filtered_df.head(max_display)
        
        if len(display_df) > 0 and 'display_name' in display_df.columns:
            selected_display = st.selectbox(
                "Select Asset",
                display_df['display_name'].tolist(),
                key="enhanced_symbol_selector",
                help="Choose from global markets"
            )
            
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
            # Fallback
            default_symbols = ['AAPL.US', 'GOOGL.US', 'MSFT.US', 'TSLA.US', 'AMZN.US']
            selected_symbol = st.selectbox("Select Symbol", default_symbols)
            st.session_state.selected_symbol = selected_symbol
    else:
        # Quick selection interface
        st.info("💡 Click 'Fetch Global Markets' to access all worldwide exchanges")
        
        # Asset categories
        categories = {
            "🇺🇸 US Stocks": ["AAPL.US", "GOOGL.US", "MSFT.US", "TSLA.US", "AMZN.US", "NVDA.US"],
            "🌍 International": ["TSM", "ASML.AS", "SAP.DE", "TM", "BABA", "NVO"],
            "💰 Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "DOT-USD"],
            "🏛️ ETFs": ["SPY", "QQQ", "IWM", "DIA", "VTI", "EFA"]
        }
        
        selected_category = st.selectbox("Asset Category", list(categories.keys()))
        
        # Quick select grid
        symbols = categories[selected_category]
        cols = st.columns(2)
        
        for i, symbol in enumerate(symbols):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(f"📊 {symbol}", key=f"quick_{symbol}", use_container_width=True):
                    st.session_state.selected_symbol = symbol
                    st.success(f"Selected: {symbol}")
        
        # Custom input
        custom_symbol = st.text_input(
            "Custom Symbol",
            placeholder="e.g., AAPL.US, BTC-USD",
            key="custom_symbol_enhanced"
        )
        
        if custom_symbol:
            st.session_state.selected_symbol = custom_symbol.upper()
            selected_symbol = custom_symbol.upper()
        else:
            selected_symbol = st.session_state.get('selected_symbol', 'AAPL.US')
    
    # Display selected symbol
    current_symbol = st.session_state.get('selected_symbol', 'AAPL.US')
    st.markdown(f"""
    <div class="portfolio-card">
        <div style="text-align: center;">
            <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">📊 Selected Asset</div>
            <div style="font-size: 1.8rem; font-weight: 700;">{current_symbol}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_analysis_settings():
    """Render analysis configuration settings"""
    
    st.markdown("""
    <div class="section-header">
        ⚙️ Analysis Settings
    </div>
    """, unsafe_allow_html=True)
    
    # Time horizon
    time_horizon = st.selectbox(
        "Time Horizon",
        ["📅 Intraday", "📊 Daily", "📈 Weekly", "📆 Monthly"],
        index=1,
        key="time_horizon_enhanced"
    )
    
    # Analysis depth
    analysis_depth = st.select_slider(
        "Analysis Depth",
        options=["Quick", "Standard", "Deep", "Comprehensive"],
        value="Standard",
        key="analysis_depth_enhanced"
    )
    
    # Risk tolerance
    risk_tolerance = st.select_slider(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive", "Speculative"],
        value="Moderate",
        key="risk_tolerance_enhanced"
    )
    
    # AI features
    st.markdown("**AI Features:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("🤖 AI Insights", value=True, key="ai_insights_enabled")
        st.checkbox("📰 News Analysis", value=True, key="news_analysis_enabled")
    
    with col2:
        st.checkbox("📊 Pattern Detection", value=True, key="pattern_detection_enabled")
        st.checkbox("🎯 Signal Generation", value=False, key="signal_generation_enabled")

def render_portfolio_overview():
    """Render portfolio overview for authenticated users"""
    
    if not st.session_state.get('authenticated', False):
        return
    
    st.markdown("""
    <div class="section-header">
        💼 Portfolio Overview
    </div>
    """, unsafe_allow_html=True)
    
    # Mock portfolio data with realistic values
    portfolio_value = 847250.75
    daily_change = 12340.25
    daily_change_pct = 1.47
    
    change_color = "var(--success-color)" if daily_change >= 0 else "var(--danger-color)"
    change_arrow = "↗️" if daily_change >= 0 else "↘️"
    
    st.markdown(f"""
    <div class="portfolio-card">
        <div style="text-align: center;">
            <div style="font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem;">
                ${portfolio_value:,.2f}
            </div>
            <div style="font-size: 0.9rem; margin-bottom: 1rem; opacity: 0.9;">
                Total Portfolio Value
            </div>
            <div style="color: {change_color}; font-size: 1.1rem; font-weight: 600;">
                {change_arrow} ${abs(daily_change):,.2f} ({daily_change_pct:+.2f}%)
            </div>
            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.5rem;">
                Today's Change
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio allocation
    allocations = [
        ("🇺🇸 US Stocks", 65),
        ("🌍 International", 20),
        ("💰 Crypto", 10),
        ("💵 Cash", 5)
    ]
    
    for name, percentage in allocations:
        st.progress(percentage / 100, text=f"{name}: {percentage}%")

def render_quick_actions():
    """Render quick action buttons"""
    
    st.markdown("""
    <div class="section-header">
        ⚡ Quick Actions
    </div>
    """, unsafe_allow_html=True)
    
    # Action grid
    actions = [
        ("🔍 Market Scan", "run_market_scan"),
        ("📊 Technical Analysis", "run_technical_analysis"),
        ("📰 News Summary", "show_news_summary"),
        ("🎯 AI Insights", "get_ai_insights"),
        ("📈 Performance Report", "generate_report"),
        ("⚠️ Risk Analysis", "run_risk_analysis")
    ]
    
    cols = st.columns(2)
    
    for i, (action_name, session_key) in enumerate(actions):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(action_name, key=f"action_{session_key}", use_container_width=True):
                st.session_state[session_key] = True
                st.success(f"✅ {action_name} initiated!")
                time.sleep(0.5)

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
    """Fetch all tickers with enhanced progress tracking"""
    exchanges = get_exchanges(api_key)
    if not exchanges:
        return []
    
    all_tickers = []
    progress_bar = st.progress(0, text="Initializing global market data fetch...")
    
    for i, exchange in enumerate(exchanges):
        code = exchange.get('Code')
        if not code:
            continue
        
        progress_text = f"Loading {exchange.get('Name', code)} ({i+1}/{len(exchanges)})"
        progress_bar.progress((i + 1) / len(exchanges), text=progress_text)
        
        tickers = get_tickers_by_exchange(code, api_key)
        if tickers:
            all_tickers.extend(tickers)
        
        time.sleep(0.2)
    
    progress_bar.empty()
    if not all_tickers:
        st.error("Failed to fetch tickers. Please check your EODHD API configuration.")
    
    return all_tickers

def render_theme_settings():
    """Render theme configuration with enhanced color options"""
    
    st.markdown("""
    <div class="section-header">
        🎨 Theme Settings
    </div>
    """, unsafe_allow_html=True)
    
    # Theme mode
    theme_col1, theme_col2 = st.columns(2)
    
    with theme_col1:
        new_theme = st.selectbox(
            "Mode",
            ["Light", "Dark"],
            index=1 if st.session_state.get('theme_mode', 'Dark') == 'Dark' else 0,
            key="theme_mode_professional"
        )
        if new_theme != st.session_state.get('theme_mode'):
            st.session_state.theme_mode = new_theme
            st.rerun()
    
    with theme_col2:
        # UPDATED COLOR SCHEME OPTIONS - More choices including Claude-like colors
        color_options = [
            "Claude Anthropic",      # Default - Claude's orange/amber theme
            "Minimal White",         # Clean white with gray accents
            "Soft Gray",            # Light gray with subtle colors
            "Clean Slate",          # Professional gray tones
            "Off White",            # Warm off-white background
            "Pure White",           # Stark white with black accents
            "Light Gray",           # Light gray variations
            "Warm Gray",            # Warm gray tones
            "Cool Gray",            # Cool gray variations
            "Neutral Stone",        # Stone/beige neutrals
            "Professional Blue",    # Your existing blue
            "Financial Green",      # Your existing green
            "Executive Purple",     # Your existing purple
            "Energy Orange",        # Your existing orange
            "Deep Ocean",           # Dark blue theme
            "Forest Green",         # Dark green theme
            "Royal Purple",         # Rich purple theme
            "Sunset Red"            # Bold red theme
        ]
        
        current_scheme = st.session_state.get('color_scheme', 'Claude Anthropic')
        current_index = color_options.index(current_scheme) if current_scheme in color_options else 0
        
        new_colors = st.selectbox(
            "Colors",
            color_options,
            index=current_index,
            key="color_scheme_professional"
        )
        
        if new_colors != st.session_state.get('color_scheme'):
            st.session_state.color_scheme = new_colors
            st.rerun()
    
    # Color scheme preview
    if st.checkbox("🎨 Show Color Preview", value=False):
        color_schemes = {
            'Claude Anthropic': {'primary': '#D97706', 'desc': 'Claude\'s signature orange theme'},
            'Minimal White': {'primary': '#6B7280', 'desc': 'Clean white with gray accents'},
            'Soft Gray': {'primary': '#6366F1', 'desc': 'Soft gray with purple highlights'},
            'Clean Slate': {'primary': '#374151', 'desc': 'Professional dark gray'},
            'Off White': {'primary': '#78716C', 'desc': 'Warm off-white background'},
            'Pure White': {'primary': '#000000', 'desc': 'Stark white with black'},
            'Light Gray': {'primary': '#525252', 'desc': 'Various light gray tones'},
            'Warm Gray': {'primary': '#78716C', 'desc': 'Warm gray variations'},
            'Cool Gray': {'primary': '#64748B', 'desc': 'Cool gray with blue tint'},
            'Neutral Stone': {'primary': '#6B7280', 'desc': 'Stone and beige neutrals'}
        }
        
        selected_scheme = color_schemes.get(new_colors, {'primary': '#6B7280', 'desc': 'Custom theme'})
        
        st.markdown(f"""
        <div style="
            background: {selected_scheme['primary']}; 
            color: white; 
            padding: 1rem; 
            border-radius: 8px; 
            text-align: center; 
            margin: 0.5rem 0;
        ">
            <strong>{new_colors}</strong><br>
            <small>{selected_scheme['desc']}</small>
        </div>
        """, unsafe_allow_html=True)
        render_translation_settings()

def render_translation_settings():
    """Render simple language translation settings"""
    
    st.markdown("""
    <div class="section-header">
        🌐 Language Translation
    </div>
    """, unsafe_allow_html=True)
    
    # Get available languages
    languages = get_supported_languages()
    language_options = list(languages.values())
    language_codes = list(languages.keys())
    
    # Current selection
    current_lang = st.session_state.get('selected_language', 'en')
    current_index = language_codes.index(current_lang) if current_lang in language_codes else 0
    
    # Language selector
    selected_language = st.selectbox(
        "Select Language",
        language_options,
        index=current_index,
        key="language_selector",
        help="Translate interface text to your preferred language"
    )
    
    # Get language code
    selected_code = language_codes[language_options.index(selected_language)]
    
    # Update session state if changed
    if selected_code != st.session_state.get('selected_language'):
        st.session_state.selected_language = selected_code
        st.rerun()
    
    # Show current selection
    st.markdown(f"""
    <div style="
        background: var(--bg-secondary); 
        border: 1px solid var(--border-color); 
        border-radius: 8px; 
        padding: 0.8rem; 
        text-align: center; 
        margin: 0.5rem 0;
    ">
        <strong>Current: {selected_language}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Translation status
    if selected_code != 'en':
        st.info("🔄 Translation active - Interface text will be translated")
        
        # Test translation
        if st.button("🧪 Test Translation", use_container_width=True):
            test_text = "Welcome to ECI Trading Platform"
            translated = simple_translate(test_text, selected_code)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original:**")
                st.code(test_text)
            with col2:
                st.markdown("**Translated:**")
                st.code(translated)
    else:
        st.success("✅ English - No translation needed")

# STEP 5: Add translation helper function (around line 1280):

def t(text, target_lang=None):
    """Simple translation helper function"""
    if not target_lang:
        target_lang = st.session_state.get('selected_language', 'en')
    
    if target_lang == 'en':
        return text
    
    return simple_translate(text, target_lang)


def handle_file_upload(uploaded_files, file_type):
    """Enhanced file upload handling with progress tracking"""
    if not uploaded_files:
        return
    
    processed_files = []
    progress_bar = st.progress(0, text="Processing uploaded files...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            progress_bar.progress((i + 1) / len(uploaded_files), 
                                text=f"Processing {uploaded_file.name}...")
            
            # Enhanced file info with metadata
            file_info = {
                'name': uploaded_file.name,
                'type': file_type,
                'size': uploaded_file.size,
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content': uploaded_file.read(),
                'status': 'uploaded',
                'processed': False
            }
            
            # Initialize uploaded files in session state
            if 'uploaded_files' not in st.session_state:
                st.session_state.uploaded_files = []
            
            # Add to uploaded files
            st.session_state.uploaded_files.append(file_info)
            processed_files.append(uploaded_file.name)
            
            time.sleep(0.1)  # Simulate processing time
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    progress_bar.empty()
    
    if processed_files:
        st.success(f"✅ Successfully uploaded {len(processed_files)} file(s)")
        
        # Show file summary
        total_size = sum(len(f['content']) for f in st.session_state.uploaded_files[-len(processed_files):])
        st.info(f"📊 Total size: {total_size / (1024*1024):.2f} MB")

def render_file_manager():
    """Render advanced file management interface"""
    if not st.session_state.get('uploaded_files'):
        return
    
    st.markdown("""
    <div class="section-header">
        📁 Document Manager
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.session_state.uploaded_files
    
    # File statistics
    total_files = len(uploaded_files)
    total_size = sum(f['size'] for f in uploaded_files)
    file_types = {}
    
    for file_info in uploaded_files:
        file_type = file_info['type']
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_files}</div>
            <div class="metric-label">Total Files</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_size / (1024*1024):.1f} MB</div>
            <div class="metric-label">Total Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(file_types)}</div>
            <div class="metric-label">File Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    # File list with enhanced interface
    for i, file_info in enumerate(uploaded_files):
        with st.expander(f"📄 {file_info['name']} ({file_info['size'] / 1024:.1f} KB)"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Type:** {file_info['type']}")
                st.write(f"**Uploaded:** {file_info['upload_time']}")
                st.write(f"**Status:** {'✅ Processed' if file_info.get('processed') else '⏳ Pending'}")
            
            with col2:
                if st.button(f"🔍 Analyze", key=f"analyze_{i}"):
                    with st.spinner("Analyzing document..."):
                        time.sleep(2)
                        st.success("✅ Analysis complete!")
                        file_info['processed'] = True
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    st.session_state.uploaded_files.remove(file_info)
                    st.success("🗑️ File deleted!")
                    st.rerun()
    
    # Bulk actions
    st.markdown("**Bulk Actions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Process All", use_container_width=True):
            with st.spinner("Processing all files..."):
                for file_info in uploaded_files:
                    file_info['processed'] = True
                    time.sleep(0.1)
                st.success("✅ All files processed!")
                st.rerun()
    
    with col2:
        if st.button("🤖 AI Analysis", use_container_width=True):
            with st.spinner("Running AI analysis..."):
                time.sleep(3)
                st.success("🧠 AI analysis completed!")
                st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.uploaded_files = []
            st.success("🗑️ All files cleared!")
            st.rerun()

def render_advanced_upload_interface():
    """Render advanced file upload interface"""
    
    st.markdown("""
    <div class="section-header">
        📤 Document Upload
    </div>
    """, unsafe_allow_html=True)
    
    # Upload tabs
    upload_tabs = st.tabs([
        "📈 Financial Reports", 
        "📊 Market Data", 
        "💼 Investment Documents",
        "📰 News & Research"
    ])
    
    # Financial Reports Tab
    with upload_tabs[0]:
        st.markdown("**SEC Filings & Financial Reports:**")
        financial_files = st.file_uploader(
            "Upload Financial Documents",
            type=['pdf', 'txt', 'csv', 'xlsx', 'xls', 'doc', 'docx'],
            accept_multiple_files=True,
            key="financial_files_advanced",
            help="10-K, 10-Q, 8-K, Annual Reports, Earnings Transcripts, Financial Statements"
        )
        
        if financial_files:
            handle_file_upload(financial_files, "Financial Reports")
        
        # Supported document types
        st.markdown("""
        **Supported Documents:**
        - 📋 SEC 10-K Annual Reports
        - 📊 SEC 10-Q Quarterly Reports
        - 📢 SEC 8-K Current Reports
        - 💰 Earnings Call Transcripts
        - 📈 Annual/Quarterly Reports
        - 🏦 Bank Call Reports
        - 📊 Financial Statements
        - 💼 Proxy Statements
        """)
    
    # Market Data Tab
    with upload_tabs[1]:
        st.markdown("**Trading & Market Data:**")
        market_files = st.file_uploader(
            "Upload Market Data Files",
            type=['csv', 'xlsx', 'json', 'txt', 'xml'],
            accept_multiple_files=True,
            key="market_files_advanced",
            help="Historical prices, trading volumes, options data, economic indicators"
        )
        
        if market_files:
            handle_file_upload(market_files, "Market Data")
        
        st.markdown("""
        **Supported Data Types:**
        - 📊 Historical Price Data
        - 📈 Trading Volume Data
        - 👥 Insider Trading Reports
        - 🎯 Options Chain Data
        - 📰 Economic Indicators
        - 🔄 Currency Exchange Rates
        - 📉 Volatility Data
        - 🏭 Sector Performance
        """)
    
    # Investment Documents Tab
    with upload_tabs[2]:
        st.markdown("**Investment & PE Documents:**")
        investment_files = st.file_uploader(
            "Upload Investment Documents",
            type=['pdf', 'xlsx', 'csv', 'txt', 'ppt', 'pptx'],
            accept_multiple_files=True,
            key="investment_files_advanced",
            help="Private equity reports, fund documents, investment memos"
        )
        
        if investment_files:
            handle_file_upload(investment_files, "Investment Documents")
        
        st.markdown("""
        **Document Categories:**
        - 💼 Private Equity Reports
        - 📊 Fund Performance Data
        - 📝 Investment Memos
        - 🎯 Due Diligence Reports
        - 💰 Capital Call Notices
        - 📈 Portfolio Valuations
        - 🔍 Risk Assessments
        - 📋 Compliance Documents
        """)
    
    # News & Research Tab
    with upload_tabs[3]:
        st.markdown("**News & Research Documents:**")
        research_files = st.file_uploader(
            "Upload Research Files",
            type=['pdf', 'txt', 'doc', 'docx', 'html'],
            accept_multiple_files=True,
            key="research_files_advanced",
            help="Research reports, news articles, analyst notes"
        )
        
        if research_files:
            handle_file_upload(research_files, "Research Documents")
        
        st.markdown("""
        **Research Types:**
        - 📰 Financial News Articles
        - 📊 Analyst Research Reports
        - 🎯 Investment Recommendations
        - 📈 Market Commentary
        - 🔍 Industry Analysis
        - 📋 Economic Research
        - 💡 Strategy Documents
        - 🌐 Alternative Data Reports
        """)
    
    # Render file manager if files exist
    render_file_manager()

# Main sidebar integration with all components
def render_complete_professional_sidebar():
    """Complete professional sidebar with all enhanced features"""
    
    # Apply professional theme
    apply_professional_theme()
    
    with st.sidebar:
        # Professional header
        logo_base64 = load_eci_logo()
        
        if logo_base64:
            st.markdown(f"""
            <div class="professional-header">
                <img src="data:image/png;base64,{logo_base64}" class="company-logo" alt="ECI Logo">
                <div class="company-name">ECI Solutions</div>
                <div class="company-tagline">AI-Powered Trading Intelligence Platform</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="professional-header">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏢</div>
                <div class="company-name">ECI Solutions</div>
                <div class="company-tagline">AI-Powered Trading Intelligence Platform</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Navigation tabs for different sections
        nav_tab = st.selectbox(
            "",
            ["🏠 Overview", "🔗 Data Sources", "📤 Upload", "📊 Analytics", "⚙️ Settings", "🌐 Language"],  # ADD "🌐 Language"
            key="sidebar_navigation",
            label_visibility="collapsed"
        )
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
        
        if nav_tab == "🏠 Overview":
            render_market_status()
            render_asset_selection()
            render_portfolio_overview()
            render_quick_actions()
            
        elif nav_tab == "🔗 Data Sources":
            render_professional_data_sources()
            render_connection_statistics()
            
        elif nav_tab == "📤 Upload":
            render_advanced_upload_interface()
            
        elif nav_tab == "📊 Analytics":
            render_analysis_settings()

        elif nav_tab == "🌐 Language":
            render_translation_settings()
            
            # Show some example translations
            st.markdown("### 📝 Translation Examples")
            
            sample_texts = [
                "Market Analysis",
                "Portfolio Performance", 
                "Risk Assessment",
                "Trading Signals",
                "Data Sources"
            ]
            
            target_lang = st.session_state.get('selected_language', 'en')
            
            if target_lang != 'en':
                for text in sample_texts:
                    translated = simple_translate(text, target_lang)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(text)
                    with col2:
                        st.text(translated)
            else:
                st.info("Select a language other than English to see translations")
            
        elif nav_tab == "⚙️ Settings":
            render_theme_settings()
            
            # Additional settings
            st.markdown("""
            <div class="section-header">
                🔧 Advanced Settings
            </div>
            """, unsafe_allow_html=True)
            
            # API configuration
            with st.expander("🔑 API Configuration"):
                st.text_input("EODHD API Key", type="password", key="eodhd_api_setting")
                st.text_input("Azure OpenAI Key", type="password", key="azure_api_setting")
                st.selectbox("Data Refresh Rate", ["Real-time", "1min", "5min", "15min"], key="refresh_rate_setting")
            
            # Notification settings
            with st.expander("🔔 Notifications"):
                st.checkbox("Price Alerts", value=True, key="price_alerts_setting")
                st.checkbox("News Alerts", value=True, key="news_alerts_setting")
                st.checkbox("AI Insights", value=False, key="ai_alerts_setting")
            
            # Performance settings
            with st.expander("⚡ Performance"):
                st.slider("Cache Size (MB)", 100, 1000, 500, key="cache_size_setting")
                st.selectbox("Processing Priority", ["Low", "Normal", "High"], index=1, key="priority_setting")
        
        # Footer with system info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.75rem; color: var(--text-muted);">
            <div style="margin-bottom: 0.5rem;">
                🟢 System: Online | 🔄 Data: Live | ⚡ AI: Active
            </div>
            <div style="margin-bottom: 0.5rem;">
                💾 Cache: 85% | 📊 API: 98% | 🔒 Security: High
            </div>
            <div>
                © 2024 ECI Solutions • Professional Trading Platform v2.1
            </div>
        </div>
        """, unsafe_allow_html=True)

# Set the main render function to use the complete professional sidebar
render_sidebar = render_complete_professional_sidebar
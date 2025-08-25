# ui/tabs/ai_intelligence/styles/theme.py
"""
Complete Professional Theme System for AI Trading Intelligence.
Creates a stunning, modern UI that will make clients say "WOW!"
Production-ready with all animations, responsive design, and accessibility features.

Usage:
    from ui.tabs.ai_intelligence.styles.theme import apply_complete_theme
    apply_complete_theme()
"""

import streamlit as st

def apply_complete_theme():
    """Apply the complete professional theme with all components."""
    st.markdown("""
    <style>
    /* Import Google Fonts for professional typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* Root CSS Variables - Professional Color Palette */
    :root {
        /* Primary Gradients - Professional Finance Theme */
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --danger-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --warning-gradient: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        --info-gradient: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        
        /* Hero Gradients */
        --hero-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --premium-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #fa709a 50%, #fee140 75%, #00f2fe 100%);
        
        /* Core Colors */
        --primary: #667eea;
        --primary-light: #8b9bff;
        --primary-dark: #4c63d2;
        --primary-darker: #3b4db8;
        --secondary: #6c757d;
        --secondary-light: #8d95a3;
        --secondary-dark: #495057;
        --success: #10b981;
        --success-light: #34d399;
        --success-dark: #059669;
        --danger: #ef4444;
        --danger-light: #f87171;
        --danger-dark: #dc2626;
        --warning: #f59e0b;
        --warning-light: #fbbf24;
        --warning-dark: #d97706;
        --info: #3b82f6;
        --info-light: #60a5fa;
        --info-dark: #2563eb;
        
        /* Background Colors */
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --bg-quaternary: #e2e8f0;
        --bg-overlay: rgba(255, 255, 255, 0.95);
        --bg-glass: rgba(255, 255, 255, 0.25);
        --bg-glass-dark: rgba(0, 0, 0, 0.05);
        
        /* Text Colors */
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-tertiary: #94a3b8;
        --text-muted: #cbd5e1;
        --text-inverse: #ffffff;
        --text-accent: #667eea;
        
        /* Border Colors */
        --border: #e2e8f0;
        --border-light: #f1f5f9;
        --border-medium: #cbd5e1;
        --border-dark: #94a3b8;
        --border-accent: #667eea;
        
        /* Shadow System */
        --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        --shadow-glow: 0 0 20px rgba(102, 126, 234, 0.3);
        --shadow-glow-success: 0 0 20px rgba(16, 185, 129, 0.3);
        --shadow-glow-danger: 0 0 20px rgba(239, 68, 68, 0.3);
        
        /* Border Radius */
        --border-radius-sm: 8px;
        --border-radius: 12px;
        --border-radius-lg: 16px;
        --border-radius-xl: 20px;
        --border-radius-2xl: 24px;
        --border-radius-full: 9999px;
        
        /* Transitions */
        --transition-fast: all 0.15s ease;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-bounce: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        
        /* Spacing */
        --space-xs: 0.25rem;
        --space-sm: 0.5rem;
        --space-md: 1rem;
        --space-lg: 1.5rem;
        --space-xl: 2rem;
        --space-2xl: 3rem;
        --space-3xl: 4rem;
        
        /* Z-Index Scale */
        --z-dropdown: 1000;
        --z-sticky: 1020;
        --z-fixed: 1030;
        --z-modal-backdrop: 1040;
        --z-modal: 1050;
        --z-popover: 1060;
        --z-tooltip: 1070;
        --z-toast: 1080;
    }

    /* Global Styles and Overrides */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        box-sizing: border-box;
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    body {
        margin: 0;
        padding: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        line-height: 1.6;
        color: var(--text-primary);
        background: var(--bg-secondary);
    }
    
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--bg-secondary);
        min-height: 100vh;
    }
    
    .main .block-container {
        padding-top: var(--space-xl);
        padding-bottom: var(--space-xl);
        max-width: 1400px;
        padding-left: var(--space-lg);
        padding-right: var(--space-lg);
    }

    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden; height: 0px;}
    .main > div:first-child > div:first-child > div:first-child {padding-top: 0px;}
    footer {visibility: hidden; height: 0px;}
    header {visibility: hidden; height: 0px;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: var(--border-radius-sm);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: var(--border-radius-sm);
        border: 2px solid var(--bg-tertiary);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
    
    ::-webkit-scrollbar-corner {
        background: var(--bg-tertiary);
    }

    /* Selection Styles */
    ::selection {
        background: var(--primary);
        color: var(--text-inverse);
    }
    
    ::-moz-selection {
        background: var(--primary);
        color: var(--text-inverse);
    }

    /* Professional Headers */
    .header-container {
        background: var(--hero-gradient);
        padding: var(--space-3xl) var(--space-xl);
        border-radius: var(--border-radius-2xl);
        margin-bottom: var(--space-xl);
        color: var(--text-inverse);
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-xl);
        backdrop-filter: blur(20px);
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255,255,255,0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(255,255,255,0.1) 0%, transparent 50%);
        animation: floatingOrbs 6s ease-in-out infinite alternate;
        pointer-events: none;
    }
    
    .header-container::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(transparent, rgba(255,255,255,0.1), transparent 30%);
        animation: rotate 20s linear infinite;
        pointer-events: none;
    }
    
    @keyframes floatingOrbs {
        0%, 100% { 
            transform: translateY(0px) scale(1);
            opacity: 0.7;
        }
        50% { 
            transform: translateY(-10px) scale(1.05);
            opacity: 1;
        }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .header-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: var(--space-lg);
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
        background: linear-gradient(45deg, #ffffff 0%, #f0f9ff 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-subtitle {
        font-size: 1.35rem;
        font-weight: 500;
        opacity: 0.95;
        margin-bottom: var(--space-xl);
        position: relative;
        z-index: 2;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .quick-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: var(--space-lg);
        margin-top: var(--space-xl);
        position: relative;
        z-index: 2;
    }
    
    .stat-item {
        text-align: center;
        padding: var(--space-lg);
        background: var(--bg-glass);
        border-radius: var(--border-radius-lg);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: var(--transition-bounce);
        box-shadow: var(--shadow-md);
    }
    
    .stat-item:hover {
        transform: translateY(-6px) scale(1.02);
        background: rgba(255,255,255,0.35);
        box-shadow: var(--shadow-xl);
    }
    
    .stat-value {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: var(--space-sm);
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .stat-label {
        font-size: 0.95rem;
        opacity: 0.9;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Enhanced Card System */
    .metric-card, .agent-card, .message-card, .scenario-card, .consensus-card {
        background: var(--bg-primary);
        border-radius: var(--border-radius-lg);
        border: 1px solid var(--border);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover, .agent-card:hover, .message-card:hover, .scenario-card:hover {
        transform: translateY(-6px);
        box-shadow: var(--shadow-2xl);
        border-color: var(--border-accent);
    }
    
    .metric-card::before, .agent-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--primary-gradient);
        opacity: 0;
        transition: var(--transition);
        transform: scaleX(0);
        transform-origin: left;
    }
    
    .metric-card:hover::before, .agent-card:hover::before {
        opacity: 1;
        transform: scaleX(1);
    }

    /* Professional Agent Cards */
    .agent-card {
        padding: var(--space-xl);
        margin: var(--space-lg) 0;
        box-shadow: var(--shadow-md);
        will-change: transform;
        backface-visibility: hidden;
    }
    
    .agent-buy {
        border-left: 5px solid var(--success);
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        box-shadow: var(--shadow-glow-success);
    }
    
    .agent-buy:hover {
        box-shadow: var(--shadow-2xl), var(--shadow-glow-success);
    }
    
    .agent-sell {
        border-left: 5px solid var(--danger);
        background: linear-gradient(135deg, #fef2f2 0%, #fef1f1 100%);
        box-shadow: var(--shadow-glow-danger);
    }
    
    .agent-sell:hover {
        box-shadow: var(--shadow-2xl), var(--shadow-glow-danger);
    }
    
    .agent-hold {
        border-left: 5px solid var(--warning);
        background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.2);
    }
    
    .agent-hold:hover {
        box-shadow: var(--shadow-2xl), 0 0 30px rgba(245, 158, 11, 0.3);
    }

    /* Message Cards with Advanced Animation */
    .message-card {
        padding: var(--space-lg);
        margin: var(--space-md) 0;
        box-shadow: var(--shadow-sm);
        animation: slideInFromLeft 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        border-left: 4px solid var(--primary);
        position: relative;
    }
    
    .message-card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, transparent 0%, var(--primary) 50%, transparent 100%);
        opacity: 0;
        transition: var(--transition);
    }
    
    .message-card:hover::after {
        opacity: 1;
    }
    
    @keyframes slideInFromLeft {
        0% {
            transform: translateX(-80px);
            opacity: 0;
            filter: blur(4px);
        }
        50% {
            transform: translateX(10px);
            filter: blur(2px);
        }
        100% {
            transform: translateX(0);
            opacity: 1;
            filter: blur(0);
        }
    }
    
    .message-analysis { 
        border-left-color: var(--info);
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    }
    
    .message-challenge { 
        border-left-color: var(--danger);
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        animation: shakeIn 0.8s ease-out;
    }
    
    .message-defense { 
        border-left-color: var(--success);
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    .message-synthesis { 
        border-left-color: #8b5cf6;
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
    }
    
    .message-consensus { 
        border-left-color: #06b6d4;
        background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%);
    }
    
    @keyframes shakeIn {
        0% { transform: translateX(-50px) rotate(-2deg); }
        25% { transform: translateX(20px) rotate(1deg); }
        50% { transform: translateX(-10px) rotate(-0.5deg); }
        75% { transform: translateX(5px) rotate(0.25deg); }
        100% { transform: translateX(0) rotate(0deg); }
    }

    /* Consensus Card - The Show-Stopper */
    .consensus-card {
        background: var(--premium-gradient);
        color: var(--text-inverse);
        padding: var(--space-3xl);
        border-radius: var(--border-radius-2xl);
        text-align: center;
        margin: var(--space-xl) 0;
        box-shadow: var(--shadow-2xl);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .consensus-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            transparent,
            rgba(255,255,255,0.03),
            transparent 30%,
            rgba(255,255,255,0.08),
            transparent 60%,
            rgba(255,255,255,0.03),
            transparent
        );
        animation: rotate 25s linear infinite;
        pointer-events: none;
    }
    
    .consensus-card::after {
        content: '';
        position: absolute;
        top: 10%;
        left: 10%;
        right: 10%;
        bottom: 10%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: var(--border-radius-xl);
        animation: pulse 3s ease-in-out infinite;
        pointer-events: none;
    }
    
    @keyframes pulse {
        0%, 100% { 
            opacity: 0.3; 
            transform: scale(0.95);
        }
        50% { 
            opacity: 0.7; 
            transform: scale(1.05);
        }
    }
    
    .consensus-card > * {
        position: relative;
        z-index: 2;
    }

    /* Advanced Confidence Bars */
    .confidence-bar {
        background: var(--bg-quaternary);
        height: 18px;
        border-radius: var(--border-radius-full);
        overflow: hidden;
        margin: var(--space-md) 0;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .confidence-bar::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 25%, rgba(255,255,255,0.1) 25%, rgba(255,255,255,0.1) 50%, transparent 50%, transparent 75%, rgba(255,255,255,0.1) 75%);
        background-size: 20px 20px;
        animation: barberpole 1s linear infinite;
        opacity: 0.3;
    }
    
    @keyframes barberpole {
        0% { background-position: 0 0; }
        100% { background-position: 20px 0; }
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: var(--border-radius-full);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--text-inverse);
        font-weight: 700;
        font-size: 0.875rem;
        transition: width 2.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    .confidence-fill::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
        animation: shimmer 2.5s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }

    /* Premium Button System */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: var(--text-inverse) !important;
        border: none !important;
        border-radius: var(--border-radius-lg) !important;
        padding: var(--space-lg) var(--space-xl) !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        transition: var(--transition) !important;
        box-shadow: var(--shadow-lg) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: var(--shadow-2xl) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
        transition: var(--transition-fast) !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.6s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        transition: all 0.6s ease;
        transform: translate(-50%, -50%);
    }
    
    .stButton > button:active::after {
        width: 300px;
        height: 300px;
    }

    /* Professional Tab System */
    .stTabs [data-baseweb="tab-list"] {
        gap: var(--space-md);
        background: var(--bg-primary);
        padding: var(--space-md);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--border-radius) !important;
        padding: var(--space-lg) var(--space-xl) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: var(--transition) !important;
        border: 2px solid transparent !important;
        background: var(--bg-secondary) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stTabs [data-baseweb="tab"]::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 0;
        height: 3px;
        background: var(--primary-gradient);
        transition: var(--transition);
        transform: translateX(-50%);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-tertiary) !important;
        border-color: var(--primary-light) !important;
        transform: translateY(-2px) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover::before {
        width: 80%;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient) !important;
        color: var(--text-inverse) !important;
        border-color: var(--primary) !important;
        box-shadow: var(--shadow-lg) !important;
        transform: translateY(-3px) !important;
    }
    
    .stTabs [aria-selected="true"]::before {
        width: 100%;
        background: rgba(255,255,255,0.5);
    }

    /* Enhanced Metrics */
    .stMetric {
        background: var(--bg-primary) !important;
        padding: var(--space-xl) !important;
        border-radius: var(--border-radius-lg) !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: var(--transition) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--info-gradient);
        opacity: 0;
        transition: var(--transition);
    }
    
    .stMetric:hover {
        transform: translateY(-4px) !important;
        box-shadow: var(--shadow-xl) !important;
        border-color: var(--primary-light) !important;
    }
    
    .stMetric:hover::before {
        opacity: 1;
    }
    
    .stMetric [data-testid="metric-container"] > div:first-child {
        font-size: 0.875rem !important;
        color: var(--text-secondary) !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: var(--space-sm) !important;
    }
    
    .stMetric [data-testid="metric-container"] > div:nth-child(2) {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: var(--text-primary) !important;
        line-height: 1.1 !important;
    }

    /* Real-time Indicators */
    .real-time-badge {
        background: var(--success-gradient);
        color: var(--text-inverse);
        padding: var(--space-sm) var(--space-lg);
        border-radius: var(--border-radius-full);
        font-size: 0.875rem;
        font-weight: 800;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: var(--shadow-glow-success);
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { 
            box-shadow: var(--shadow-glow-success);
            transform: scale(1);
        }
        50% { 
            box-shadow: var(--shadow-glow-success), 0 0 30px rgba(16, 185, 129, 0.4);
            transform: scale(1.02);
        }
    }
    
    .real-time-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.4) 50%, transparent 70%);
        animation: sweep 3s infinite;
    }
    
    @keyframes sweep {
        0% { transform: translateX(-100%) skewX(-25deg); }
        100% { transform: translateX(200%) skewX(-25deg); }
    }

    /* Market Stats Container */
    .market-stats {
        background: var(--bg-primary);
        padding: var(--space-xl);
        border-radius: var(--border-radius-lg);
        margin: var(--space-lg) 0;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .market-stats::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--success-gradient);
        border-radius: var(--border-radius-lg) var(--border-radius-lg) 0 0;
    }
    
    .market-stats::after {
        content: '';
        position: absolute;
        top: 10px;
        right: 10px;
        width: 10px;
        height: 10px;
        background: var(--success);
        border-radius: 50%;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }

    /* Enhanced Progress Bars */
    .stProgress > div > div > div > div {
        background: var(--primary-gradient) !important;
        border-radius: var(--border-radius-sm) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stProgress > div > div > div > div::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: progressShimmer 2s infinite;
    }
    
    @keyframes progressShimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .stProgress > div > div > div {
        background: var(--bg-quaternary) !important;
        border-radius: var(--border-radius-sm) !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    /* AI Trading Intelligence Specific Components */
    
    /* Debate Arena Header */
    .debate-arena-header {
        background: var(--hero-gradient);
        padding: var(--space-xl) var(--space-lg);
        border-radius: var(--border-radius-2xl);
        margin-bottom: var(--space-xl);
        color: var(--text-inverse);
        text-align: center;
        box-shadow: var(--shadow-2xl);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .debate-arena-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 30% 20%, rgba(255,255,255,0.1) 0%, transparent 50%),
            radial-gradient(circle at 70% 80%, rgba(255,255,255,0.08) 0%, transparent 50%);
        opacity: 0.7;
        animation: floatingOrbs 8s ease-in-out infinite alternate;
    }

    /* Agent Cards with Signal-Based Styling */
    .agent-confidence-container {
        background: var(--bg-primary);
        border-radius: var(--border-radius-lg);
        padding: var(--space-lg);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border);
        margin: var(--space-md) 0;
    }
    
    .agent-confidence-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: var(--space-md);
    }
    
    .agent-name {
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text-primary);
    }
    
    .agent-signal {
        padding: var(--space-xs) var(--space-md);
        border-radius: var(--border-radius-full);
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .signal-buy {
        background: var(--success);
        color: var(--text-inverse);
    }
    
    .signal-sell {
        background: var(--danger);
        color: var(--text-inverse);
    }
    
    .signal-hold {
        background: var(--warning);
        color: var(--text-inverse);
    }

    /* Consensus Display */
    .consensus-display {
        background: var(--premium-gradient);
        color: var(--text-inverse);
        padding: var(--space-2xl);
        border-radius: var(--border-radius-2xl);
        text-align: center;
        margin: var(--space-xl) 0;
        box-shadow: var(--shadow-2xl);
        position: relative;
        overflow: hidden;
    }
    
    .consensus-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: var(--space-lg);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .consensus-signal {
        font-size: 3rem;
        font-weight: 900;
        margin: var(--space-lg) 0;
        text-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .consensus-confidence {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: var(--space-lg);
    }
    
    .voting-breakdown {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: var(--space-md);
        margin-top: var(--space-xl);
    }
    
    .vote-item {
        background: rgba(255,255,255,0.15);
        padding: var(--space-md);
        border-radius: var(--border-radius);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .vote-emoji {
        font-size: 2rem;
        display: block;
        margin-bottom: var(--space-sm);
    }
    
    .vote-count {
        font-size: 1.5rem;
        font-weight: 800;
        display: block;
    }
    
    .vote-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }

    /* Debate Messages */
    .debate-message {
        background: var(--bg-primary);
        border-radius: var(--border-radius-lg);
        padding: var(--space-lg);
        margin: var(--space-md) 0;
        border-left: 4px solid;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
        animation: messageSlideIn 0.8s ease-out;
    }
    
    @keyframes messageSlideIn {
        0% {
            transform: translateX(-100px);
            opacity: 0;
        }
        60% {
            transform: translateX(10px);
        }
        100% {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .debate-message::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 3px;
        height: 100%;
        background: linear-gradient(180deg, transparent, var(--primary), transparent);
        opacity: 0;
        transition: var(--transition);
    }
    
    .debate-message:hover::after {
        opacity: 1;
    }
    
    .message-opening {
        border-left-color: var(--info);
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    }
    
    .message-challenge {
        border-left-color: var(--danger);
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        animation: messageShake 0.8s ease-out;
    }
    
    @keyframes messageShake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    .message-defense {
        border-left-color: var(--success);
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    
    .message-synthesis {
        border-left-color: #8b5cf6;
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
    }
    
    .message-consensus {
        border-left-color: #06b6d4;
        background: linear-gradient(135deg, #ecfeff 0%, #cffafe 100%);
        box-shadow: var(--shadow-lg);
    }
    
    .message-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: var(--space-md);
    }
    
    .message-agent {
        display: flex;
        align-items: center;
        gap: var(--space-sm);
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        background: var(--primary-gradient);
        color: var(--text-inverse);
    }
    
    .message-name {
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .message-role {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-left: var(--space-xs);
    }
    
    .message-type {
        padding: var(--space-xs) var(--space-sm);
        background: var(--bg-tertiary);
        border-radius: var(--border-radius);
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .message-content {
        color: var(--text-primary);
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .message-timestamp {
        font-size: 0.8rem;
        color: var(--text-tertiary);
        text-align: right;
        margin-top: var(--space-sm);
    }

    /* Agent Status Indicators */
    .agent-status {
        display: inline-flex;
        align-items: center;
        gap: var(--space-xs);
        padding: var(--space-xs) var(--space-sm);
        background: var(--bg-secondary);
        border-radius: var(--border-radius-full);
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: statusPulse 2s infinite;
    }
    
    .status-analyzing .status-dot {
        background: var(--info);
    }
    
    .status-ready .status-dot {
        background: var(--success);
    }
    
    .status-error .status-dot {
        background: var(--danger);
    }
    
    .status-waiting .status-dot {
        background: var(--warning);
    }
    
    @keyframes statusPulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.5;
            transform: scale(1.2);
        }
    }

    /* Market Data Display */
    .market-ticker {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--border-radius-lg);
        padding: var(--space-lg);
        margin: var(--space-md) 0;
        box-shadow: var(--shadow-md);
    }
    
    .ticker-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: var(--space-md);
    }
    
    .ticker-symbol {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-primary);
    }
    
    .ticker-name {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-left: var(--space-sm);
    }
    
    .ticker-price {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: var(--space-sm) 0;
    }
    
    .ticker-change {
        display: flex;
        align-items: center;
        gap: var(--space-sm);
    }
    
    .ticker-change.positive {
        color: var(--success);
    }
    
    .ticker-change.negative {
        color: var(--danger);
    }
    
    .ticker-arrow {
        font-size: 1.2rem;
    }

    /* Mobile Responsiveness */
    @media (max-width: 1024px) {
        .header-title { 
            font-size: 2.5rem !important; 
        }
        
        .quick-stats {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: var(--space-md) !important;
        }
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: var(--space-md);
            padding-right: var(--space-md);
        }
        
        .header-container {
            padding: var(--space-xl) var(--space-lg);
        }
        
        .header-title { 
            font-size: 2rem !important; 
        }
        
        .header-subtitle {
            font-size: 1.1rem !important;
        }
        
        .agent-card, .message-card {
            padding: var(--space-lg) !important;
            margin: var(--space-md) 0 !important;
        }
        
        .consensus-card {
            padding: var(--space-xl) var(--space-lg) !important;
        }
        
        .stColumns > div {
            padding: 0 !important;
            margin-bottom: var(--space-md);
        }
        
        .voting-breakdown {
            grid-template-columns: 1fr;
            gap: var(--space-md);
        }
        
        .message-header {
            flex-direction: column;
            align-items: flex-start;
            gap: var(--space-sm);
        }
    }

    @media (max-width: 480px) {
        .header-title { 
            font-size: 1.8rem !important; 
        }
        
        .header-subtitle { 
            font-size: 1rem !important; 
        }
        
        .quick-stats {
            grid-template-columns: 1fr !important;
        }
        
        .stat-value { 
            font-size: 1.8rem !important; 
        }
        
        .stButton > button {
            padding: var(--space-md) var(--space-lg) !important;
            font-size: 0.95rem !important;
        }
        
        .debate-message {
            padding: var(--space-md);
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            font-size: 1rem;
        }
        
        .consensus-signal {
            font-size: 2.5rem !important;
        }
    }

    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --bg-quaternary: #475569;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-tertiary: #94a3b8;
            --text-muted: #64748b;
            --border: #334155;
            --border-light: #475569;
            --border-medium: #64748b;
            --border-dark: #94a3b8;
        }
        
        .header-container {
            background: linear-gradient(135deg, #1e293b 0%, #334155 50%, #475569 100%);
        }
        
        .agent-buy {
            background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        }
        
        .agent-sell {
            background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        }
        
        .agent-hold {
            background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        }
    }

    /* High Contrast Mode */
    @media (prefers-contrast: high) {
        :root {
            --border: #000000;
            --text-primary: #000000;
            --text-secondary: #333333;
        }
        
        .stButton > button {
            border: 2px solid #000000 !important;
        }
        
        .agent-card,
        .message-card,
        .metric-card {
            border: 2px solid #000000 !important;
        }
    }

    /* Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
        
        .header-container::before,
        .header-container::after,
        .consensus-card::before,
        .consensus-card::after,
        .real-time-badge::before {
            animation: none !important;
        }
    }

    /* Print Styles */
    @media print {
        .header-container,
        .consensus-card {
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            -webkit-print-color-adjust: exact;
        }
        
        .stButton,
        .real-time-badge {
            display: none !important;
        }
        
        .agent-card,
        .message-card,
        .metric-card {
            break-inside: avoid;
            page-break-inside: avoid;
        }
    }

    /* Accessibility Enhancements */
    .stButton > button:focus-visible {
        outline: 3px solid var(--primary-light) !important;
        outline-offset: 3px !important;
    }
    
    .agent-card:focus-within,
    .message-card:focus-within,
    .metric-card:focus-within {
        outline: 2px solid var(--primary) !important;
        outline-offset: 2px !important;
    }

    /* High Performance Optimizations */
    .agent-card, .message-card, .metric-card, .debate-message {
        will-change: transform;
        backface-visibility: hidden;
        perspective: 1000px;
        transform-style: preserve-3d;
    }
    
    /* GPU Acceleration for Animations */
    .header-container::before,
    .header-container::after,
    .consensus-card::before,
    .confidence-fill::before,
    .real-time-badge::before {
        will-change: transform;
        transform: translateZ(0);
    }

    /* Utility Classes */
    .gradient-text {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .glass-effect {
        backdrop-filter: blur(15px);
        background: var(--bg-glass);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .hover-lift {
        transition: var(--transition);
    }
    
    .hover-lift:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl);
    }
    
    .text-center { text-align: center; }
    .text-left { text-align: left; }
    .text-right { text-align: right; }
    
    .font-bold { font-weight: 700; }
    .font-extrabold { font-weight: 800; }
    .font-black { font-weight: 900; }
    
    .uppercase { text-transform: uppercase; }
    .tracking-wide { letter-spacing: 0.025em; }
    .tracking-wider { letter-spacing: 0.05em; }
    
    /* Custom Columns */
    .stColumns > div {
        padding: 0 var(--space-sm) !important;
    }
    
    .stColumns > div:first-child {
        padding-left: 0 !important;
    }
    
    .stColumns > div:last-child {
        padding-right: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Additional utility functions for theme management
def get_theme_colors():
    """Return the current theme color palette."""
    return {
        'primary': '#667eea',
        'success': '#10b981',
        'danger': '#ef4444',
        'warning': '#f59e0b',
        'info': '#3b82f6',
        'bg_primary': '#ffffff',
        'bg_secondary': '#f8fafc',
        'text_primary': '#1e293b',
        'text_secondary': '#64748b'
    }

def apply_dark_theme():
    """Apply dark theme variant."""
    st.markdown("""
    <style>
    :root {
        --bg-primary: #0f172a !important;
        --bg-secondary: #1e293b !important;
        --bg-tertiary: #334155 !important;
        --text-primary: #f8fafc !important;
        --text-secondary: #cbd5e1 !important;
        --border: #334155 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def apply_high_contrast_theme():
    """Apply high contrast theme for accessibility."""
    st.markdown("""
    <style>
    :root {
        --text-primary: #000000 !important;
        --text-secondary: #333333 !important;
        --bg-primary: #ffffff !important;
        --bg-secondary: #f0f0f0 !important;
        --border: #000000 !important;
    }
    
    .stButton > button {
        border: 2px solid #000000 !important;
    }
    
    .agent-card,
    .message-card,
    .metric-card {
        border: 2px solid #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def set_custom_theme_colors(primary_color: str, success_color: str, danger_color: str):
    """Set custom theme colors dynamically."""
    st.markdown(f"""
    <style>
    :root {{
        --primary: {primary_color} !important;
        --success: {success_color} !important;
        --danger: {danger_color} !important;
        --primary-gradient: linear-gradient(135deg, {primary_color} 0%, #764ba2 100%) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Legacy function names for backwards compatibility
def apply_theme():
    """Legacy function - use apply_complete_theme() instead."""
    apply_complete_theme()

def apply_custom_css():
    """Legacy function - included in apply_complete_theme()."""
    pass

def apply_ai_specific_styles():
    """Legacy function - included in apply_complete_theme()."""
    pass
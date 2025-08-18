# ui/tabs/ai_intelligence/styles/theme.py
import streamlit as st

def apply_theme():
    """Apply professional theme and styling to the application."""
    st.markdown("""
    <style>
    /* CSS Variables for consistent theming */
    :root {
        --primary: #1e88e5;
        --secondary: #424242;
        --success: #43a047;
        --danger: #e53935;
        --warning: #fb8c00;
        --info: #1e88e5;
        --light: #f5f5f5;
        --dark: #212121;
        --border-radius: 12px;
        --box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Professional Header */
    .header-container {
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid var(--light);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--dark);
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--box-shadow);
        transition: var(--transition);
        border: 1px solid #e0e0e0;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-header {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--dark);
        line-height: 1.2;
    }
    
    /* Signal Cards */
    .signal-card {
        background: white;
        border-radius: var(--border-radius);
        border: 2px solid;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: var(--transition);
    }
    
    .signal-card:hover {
        transform: translateX(4px);
    }
    
    .signal-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .agent-emoji {
        font-size: 1.5rem;
    }
    
    .agent-name {
        font-weight: 600;
        font-size: 1.1rem;
        flex-grow: 1;
    }
    
    .signal-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .confidence-bar {
        height: 24px;
        background: #e0e0e0;
        border-radius: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        transition: width 1s ease-out;
    }
    
    /* Loading Skeleton */
    .skeleton-loader {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: var(--border-radius);
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: var(--transition);
        box-shadow: var(--box-shadow);
    }
    
    .stButton > button:hover {
        background: #1976d2;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: white;
        padding: 0.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: var(--transition);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary);
        color: white;
    }
    
    /* Containers */
    .error-container, .success-container {
        padding: 1rem;
        border-radius: var(--border-radius);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .error-container {
        background: #ffebee;
        border: 1px solid #ffcdd2;
        color: var(--danger);
    }
    
    .success-container {
        background: #e8f5e9;
        border: 1px solid #c8e6c9;
        color: var(--success);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title { font-size: 2rem; }
        .metric-value { font-size: 1.5rem; }
        .signal-header { flex-direction: column; align-items: start; }
    }
    </style>
    """, unsafe_allow_html=True)
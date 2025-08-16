import streamlit as st

def apply_custom_css():
    """Applies custom CSS to the Streamlit application for a better UI."""
    st.markdown("""
    <style>
        /* Main Header Style */
        .main-header {
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #636efa, #ef553b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }

        /* Card Styles for Agents and Metrics */
        .agent-card, .metric-card {
            background: #FFFFFF;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
            border-left: 5px solid #636efa;
            transition: transform 0.2s;
        }
        .agent-card:hover, .metric-card:hover {
            transform: translateY(-4px);
        }

        /* Status Indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active {
            background-color: #00cc96; /* Green */
        }
        .status-inactive {
            background-color: #ef553b; /* Red */
        }

        /* General UI Tweaks */
        .stButton>button {
            border-radius: 10px;
            border: 2px solid #636efa;
            background-color: white;
            color: #636efa;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            background-color: #636efa;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

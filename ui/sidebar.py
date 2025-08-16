import streamlit as st
import pandas as pd
from services.data_fetcher import fetch_all_tickers

def render_sidebar():
    """Renders the sidebar for user inputs and configuration."""
    with st.sidebar:
        st.markdown("## 🎯 Agent Control Panel")
        
        # --- API Keys ---
        st.markdown("### 🔑 API Configuration")
        st.session_state.eodhd_api_key = st.text_input("EODHD API Key", type="password", value=st.session_state.get('eodhd_api_key', ''))
        st.session_state.azure_api_key = st.text_input("Azure OpenAI API Key", type="password", value=st.session_state.get('azure_api_key', ''))
        st.session_state.azure_endpoint = st.text_input("Azure OpenAI Endpoint", value=st.session_state.get('azure_endpoint', ''))
        st.session_state.azure_deployment = st.text_input("Azure Chat Deployment", value=st.session_state.get('azure_deployment', ''))
        st.session_state.azure_whisper_deployment = st.text_input("Azure Whisper Deployment", value=st.session_state.get('azure_whisper_deployment', ''))
        
        # --- Ticker Selection ---
        st.markdown("### 📈 Ticker Selection")
        
        if 'all_tickers_df' not in st.session_state:
            st.session_state.all_tickers_df = pd.DataFrame()

        if st.session_state.all_tickers_df.empty:
            if st.button("Fetch World Tickers", disabled=(not st.session_state.eodhd_api_key)):
                with st.spinner("Fetching tickers... this may take a minute."):
                    raw_tickers = fetch_all_tickers(st.session_state.eodhd_api_key)
                    if raw_tickers:
                        df = pd.DataFrame(raw_tickers)
                        df['display_name'] = df['Code'] + " - " + df['Name']
                        st.session_state.all_tickers_df = df
                        st.rerun()

        if not st.session_state.all_tickers_df.empty:
            search_term = st.text_input("Search Ticker", "").upper()
            filtered_df = st.session_state.all_tickers_df
            if search_term:
                filtered_df = filtered_df[
                    filtered_df['Code'].str.upper().str.contains(search_term) |
                    filtered_df['Name'].str.upper().str.contains(search_term)
                ]
            
            options_map = pd.Series(filtered_df.head(100)['Code'].values, index=filtered_df.head(100)['display_name']).to_dict()
            selected_display = st.selectbox("Select Symbol", list(options_map.keys()))
            selected_symbol = options_map.get(selected_display)
        else:
            st.info("Enter EODHD key and fetch tickers.")
            selected_symbol = st.selectbox("Select Symbol (Default)", ['AAPL.US', 'GOOGL.US', 'MSFT.US'])

        # Update session state if symbol changes, clear cached analysis
        if st.session_state.get('selected_symbol') != selected_symbol:
            st.session_state.selected_symbol = selected_symbol
            # Clear any cached data related to the old symbol
            keys_to_clear = ['chart_analysis_report', 'forecast_df', 'anomalies']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

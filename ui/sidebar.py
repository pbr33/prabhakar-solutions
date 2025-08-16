import streamlit as st
import pandas as pd

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
                    try:
                        # Import locally to avoid circular import
                        from services.data_fetcher import fetch_all_tickers
                        raw_tickers = fetch_all_tickers(st.session_state.eodhd_api_key)
                        if raw_tickers:
                            df = pd.DataFrame(raw_tickers)
                            # Ensure required columns exist
                            if 'Code' in df.columns and 'Name' in df.columns:
                                df['display_name'] = df['Code'] + " - " + df['Name'].fillna('')
                                st.session_state.all_tickers_df = df
                                st.rerun()
                            else:
                                st.error("Invalid ticker data format received.")
                        else:
                            st.error("No tickers received. Please check your API key.")
                    except Exception as e:
                        st.error(f"Error fetching tickers: {e}")

        if not st.session_state.all_tickers_df.empty:
            search_term = st.text_input("Search Ticker", "").upper()
            filtered_df = st.session_state.all_tickers_df
            if search_term:
                # Safe filtering with error handling
                try:
                    code_mask = filtered_df['Code'].str.upper().str.contains(search_term, na=False)
                    name_mask = filtered_df['Name'].str.upper().str.contains(search_term, na=False)
                    filtered_df = filtered_df[code_mask | name_mask]
                except Exception as e:
                    st.warning(f"Search error: {e}")
                    filtered_df = st.session_state.all_tickers_df
            
            # Limit results for performance
            display_df = filtered_df.head(100)
            
            if not display_df.empty and 'display_name' in display_df.columns:
                options_map = pd.Series(
                    display_df['Code'].values, 
                    index=display_df['display_name']
                ).to_dict()
                
                if options_map:
                    selected_display = st.selectbox("Select Symbol", list(options_map.keys()))
                    selected_symbol = options_map.get(selected_display, 'AAPL.US')
                else:
                    selected_symbol = st.selectbox("Select Symbol (Default)", ['AAPL.US', 'GOOGL.US', 'MSFT.US'])
            else:
                selected_symbol = st.selectbox("Select Symbol (Default)", ['AAPL.US', 'GOOGL.US', 'MSFT.US'])
        else:
            st.info("Enter EODHD key and fetch tickers for full symbol search.")
            selected_symbol = st.selectbox("Select Symbol (Default)", ['AAPL.US', 'GOOGL.US', 'MSFT.US', 'TSLA.US', 'AMZN.US'])

        # Update session state if symbol changes, clear cached analysis
        if st.session_state.get('selected_symbol') != selected_symbol:
            st.session_state.selected_symbol = selected_symbol
            # Clear any cached data related to the old symbol
            keys_to_clear = [
                'chart_analysis_report', 
                'forecast_df', 
                'anomalies', 
                'detected_patterns',
                'comprehensive_analysis',
                'forecast_plot_df'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Display current symbol info
        if selected_symbol:
            st.info(f"Selected: {selected_symbol}")
            
        # --- Additional Settings ---
        st.markdown("### ⚙️ Settings")
        st.session_state.use_cache = st.checkbox("Use Data Caching", value=True, help="Cache data for faster loading")
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False, help="Show additional debug information")
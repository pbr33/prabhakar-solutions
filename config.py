import streamlit as st

def get_config():
    """
    Returns a dictionary of configuration settings, primarily from Streamlit's session state.
    This centralizes access to API keys and other settings.
    """
    config = {
        # EODHD API Key for market data
        'eodhd_api_key': st.session_state.get('eodhd_api_key', ''),

        # Azure OpenAI Credentials for LLM and Whisper
        'azure_api_key': st.session_state.get('azure_api_key', ''),
        'azure_endpoint': st.session_state.get('azure_endpoint', ''),
        'azure_api_version': st.session_state.get('azure_api_version', '2024-02-01'),
        'azure_chat_deployment': st.session_state.get('azure_deployment', ''),
        'azure_whisper_deployment': st.session_state.get('azure_whisper_deployment', ''),
        
        # Selected symbol from the UI
        'selected_symbol': st.session_state.get('selected_symbol', 'AAPL.US'),
        
        # LLM instance
        'llm': st.session_state.get('llm')
    }
    return config


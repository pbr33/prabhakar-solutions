import streamlit as st

class VoiceAssistantTab:
    def __init__(self, symbol, market_data, ui_components):
        self.symbol = symbol
        self.market_data = market_data
        self.ui = ui_components
    
    def render(self):
        st.subheader("🎤 Voice Assistant")
        st.info("Voice Assistant features coming soon!")
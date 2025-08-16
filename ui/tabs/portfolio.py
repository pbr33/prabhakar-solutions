import streamlit as st
import pandas as pd
import random
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import get_config
from services.data_fetcher import pro_get_real_time_data
from analysis.reporting import generate_investment_memo, generate_board_pack_content

@st.cache_data
def generate_pe_portfolio_data():
    """Generates mock private equity portfolio data."""
    companies = ['InnovateTech', 'BioHealth Solutions', 'GreenEnergy Corp']
    portfolio_data = []
    
    for name in companies:
        invested_capital = random.uniform(50, 150)
        current_valuation = random.uniform(200, 600)
        moic = current_valuation / invested_capital  # Calculate MOIC directly instead of using lambda
        
        company_data = {
            'Company Name': name,
            'Industry': random.choice(['SaaS', 'Biotech', 'Renewables']),
            'Invested Capital (M)': invested_capital,
            'Current Valuation (M)': current_valuation,
            'MOIC': moic,
            'IRR (%)': random.uniform(18, 35),
            'KPI History': pd.DataFrame({
                'date': pd.to_datetime(['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4']),
                'revenue': [random.uniform(20, 30), random.uniform(30, 40), random.uniform(40, 50), random.uniform(50, 60)],
                'ebitda': [random.uniform(5, 10), random.uniform(10, 15), random.uniform(15, 20), random.uniform(20, 25)]
            })
        }
        portfolio_data.append(company_data)
    
    return portfolio_data

def render():
    """Renders the Institutional Portfolio Views tab."""
    st.markdown("## 💼 Institutional Portfolio Views")
    cfg = get_config()
    llm = cfg['llm']
    trading_engine = st.session_state.trading_engine

    # --- Hedge Fund View ---
    st.markdown("### Hedge Fund Portfolio Overview")
    hf_portfolio = trading_engine.positions
    if not hf_portfolio:
        st.info("No positions in the hedge fund portfolio.")
        hf_df = pd.DataFrame()  # Create empty DataFrame for later use
    else:
        live_prices = {s: pro_get_real_time_data(s, cfg['eodhd_api_key']).get('close', p['avg_price']) for s, p in hf_portfolio.items()}
        positions_data = [{
            'Symbol': symbol, 
            'Quantity': pos['quantity'], 
            'Avg Price': pos['avg_price'],
            'Current Price': live_prices.get(symbol, pos['avg_price']),
            'Market Value': pos['quantity'] * live_prices.get(symbol, pos['avg_price']),
            'Unrealized PNL': (live_prices.get(symbol, pos['avg_price']) - pos['avg_price']) * pos['quantity']
        } for symbol, pos in hf_portfolio.items()]
        hf_df = pd.DataFrame(positions_data)
        st.dataframe(hf_df)

    # --- Private Equity View ---
    st.markdown("---")
    st.markdown("### Private Equity Portfolio Monitoring")
    pe_portfolio = generate_pe_portfolio_data()
    pe_df = pd.DataFrame(pe_portfolio)
    
    # Remove the KPI History column for display as it contains DataFrames
    display_df = pe_df.drop(columns=['KPI History'])
    st.dataframe(display_df)
    
    selected_company_name = st.selectbox("Select Portfolio Company", pe_df['Company Name'].tolist())
    selected_company_data = pe_df[pe_df['Company Name'] == selected_company_name].iloc[0]

    # Display KPI history for selected company
    st.markdown(f"#### KPI History for {selected_company_name}")
    kpi_history = selected_company_data['KPI History']
    st.dataframe(kpi_history)

    # --- AI-Powered Reporting ---
    st.markdown("---")
    st.markdown("## 🧠 AI-Powered Document Automation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Investment Memo", disabled=(llm is None)):
            with st.spinner("Generating memo..."):
                memo = generate_investment_memo(llm, selected_company_data, hf_df)
                st.text_area("Generated Investment Memo", memo, height=300)
    
    with col2:
        if st.button("Generate Board Pack", disabled=(llm is None)):
            with st.spinner("Generating board pack..."):
                board_pack = generate_board_pack_content(llm, selected_company_data, kpi_history)
                st.text_area("Generated Board Pack", board_pack, height=300)
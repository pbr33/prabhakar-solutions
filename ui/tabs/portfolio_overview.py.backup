# ui/tabs/portfolio_overview.py
"""
Portfolio Overview Tab - Main dashboard with consolidated metrics
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

from config import get_config
from services.data_fetcher import pro_get_real_time_data

def calculate_portfolio_metrics(hf_portfolio, pe_portfolio):
    """Calculate consolidated portfolio metrics"""
    # Hedge Fund metrics
    hf_total_value = 0
    hf_unrealized_pnl = 0
    
    if hf_portfolio:
        cfg = get_config()
        live_prices = {s: pro_get_real_time_data(s, cfg['eodhd_api_key']).get('close', p['avg_price']) 
                      for s, p in hf_portfolio.items()}
        
        for symbol, pos in hf_portfolio.items():
            current_price = live_prices.get(symbol, pos['avg_price'])
            hf_total_value += pos['quantity'] * current_price
            hf_unrealized_pnl += (current_price - pos['avg_price']) * pos['quantity']
    
    # PE metrics
    pe_total_value = sum([company['Current Valuation (M)'] for company in pe_portfolio]) if pe_portfolio else 0
    pe_invested_capital = sum([company['Invested Capital (M)'] for company in pe_portfolio]) if pe_portfolio else 0
    
    return {
        'total_aum': hf_total_value + pe_total_value * 1000000,  # PE is in millions
        'hf_value': hf_total_value,
        'pe_value': pe_total_value * 1000000,
        'hf_pnl': hf_unrealized_pnl,
        'pe_unrealized_gain': (pe_total_value - pe_invested_capital) * 1000000 if pe_total_value > 0 else 0,
        'total_pnl': hf_unrealized_pnl + ((pe_total_value - pe_invested_capital) * 1000000 if pe_total_value > 0 else 0)
    }

def create_asset_allocation_chart(hf_value, pe_value):
    """Create asset allocation pie chart"""
    if hf_value == 0 and pe_value == 0:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=['Hedge Fund', 'Private Equity'],
        values=[hf_value, pe_value],
        hole=0.4,
        textinfo='label+percent',
        marker=dict(colors=['#1f77b4', '#ff7f0e'])
    )])
    
    fig.update_layout(
        title="Asset Allocation",
        height=400,
        showlegend=True
    )
    return fig

def create_geographic_exposure_chart():
    """Create mock geographic exposure chart"""
    regions = ['North America', 'Europe', 'Asia Pacific', 'Emerging Markets']
    exposure = [random.uniform(20, 40) for _ in regions]
    
    fig = px.bar(
        x=regions,
        y=exposure,
        title="Geographic Exposure (%)",
        labels={'x': 'Region', 'y': 'Exposure %'},
        color=exposure,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400)
    return fig

def render_portfolio_overview_tab(trading_engine, pe_portfolio):
    """Render the Portfolio Overview tab"""
    st.markdown("## 📊 Portfolio Overview Dashboard")
    
    # Get portfolio data
    hf_portfolio = trading_engine.positions if trading_engine else {}
    metrics = calculate_portfolio_metrics(hf_portfolio, pe_portfolio)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total AUM",
            f"${metrics['total_aum']:,.0f}",
            delta=f"${metrics['total_pnl']:,.0f}"
        )
    
    with col2:
        st.metric(
            "Hedge Fund Value",
            f"${metrics['hf_value']:,.0f}",
            delta=f"${metrics['hf_pnl']:,.0f}"
        )
    
    with col3:
        st.metric(
            "PE Portfolio Value",
            f"${metrics['pe_value']:,.0f}",
            delta=f"${metrics['pe_unrealized_gain']:,.0f}"
        )
    
    with col4:
        total_return = (metrics['total_pnl'] / (metrics['total_aum'] - metrics['total_pnl']) * 100) if (metrics['total_aum'] - metrics['total_pnl']) > 0 else 0
        st.metric(
            "Total Return %",
            f"{total_return:.2f}%"
        )
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        allocation_chart = create_asset_allocation_chart(metrics['hf_value'], metrics['pe_value'])
        if allocation_chart:
            st.plotly_chart(allocation_chart, use_container_width=True)
        else:
            st.info("No portfolio data available for asset allocation chart")
    
    with col2:
        geo_chart = create_geographic_exposure_chart()
        st.plotly_chart(geo_chart, use_container_width=True)
    
    # Recent Activity
    st.markdown("### 📈 Recent Portfolio Activity")
    
    # Mock recent activity data
    recent_activities = [
        {"Date": datetime.now() - timedelta(days=1), "Type": "Trade", "Description": "Purchased AAPL shares", "Impact": "+$15,000"},
        {"Date": datetime.now() - timedelta(days=3), "Type": "Valuation", "Description": "PE company valuation update", "Impact": "+$2.5M"},
        {"Date": datetime.now() - timedelta(days=5), "Type": "Dividend", "Description": "Dividend received from MSFT", "Impact": "+$850"},
        {"Date": datetime.now() - timedelta(days=7), "Type": "Trade", "Description": "Sold GOOGL position", "Impact": "+$8,500"}
    ]
    
    activity_df = pd.DataFrame(recent_activities)
    activity_df['Date'] = activity_df['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(activity_df, use_container_width=True)
    
    # Performance Summary
    st.markdown("### 📊 Performance Summary")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.markdown("**Hedge Fund Performance**")
        if hf_portfolio:
            hf_return = (metrics['hf_pnl'] / (metrics['hf_value'] - metrics['hf_pnl']) * 100) if (metrics['hf_value'] - metrics['hf_pnl']) > 0 else 0
            st.write(f"Return: {hf_return:.2f}%")
            st.write(f"Positions: {len(hf_portfolio)}")
        else:
            st.write("No HF positions")
    
    with perf_col2:
        st.markdown("**Private Equity Performance**")
        if pe_portfolio:
            avg_moic = sum([co['MOIC'] for co in pe_portfolio]) / len(pe_portfolio)
            avg_irr = sum([co['IRR (%)'] for co in pe_portfolio]) / len(pe_portfolio)
            st.write(f"Avg MOIC: {avg_moic:.2f}x")
            st.write(f"Avg IRR: {avg_irr:.1f}%")
        else:
            st.write("No PE companies")
    
    with perf_col3:
        st.markdown("**Risk Metrics**")
        # Mock risk metrics
        st.write("VaR (95%): $125,000")
        st.write("Max Drawdown: 8.5%")
        st.write("Sharpe Ratio: 1.85")
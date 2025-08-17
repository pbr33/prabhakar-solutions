# ui/tabs/hedge_fund_analytics.py
"""
Hedge Fund Analytics Tab - Detailed HF portfolio analytics with risk metrics
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

from config import get_config
from services.data_fetcher import pro_get_real_time_data

def calculate_portfolio_risk_metrics(portfolio_df):
    """Calculate risk metrics for the portfolio"""
    if portfolio_df.empty:
        return {}
    
    # Calculate concentration risk (Herfindahl-Hirschman Index)
    total_value = portfolio_df['Market Value'].sum()
    weights = portfolio_df['Market Value'] / total_value
    hhi = (weights ** 2).sum()
    
    # Risk metrics
    total_pnl = portfolio_df['Unrealized PNL'].sum()
    total_invested = (portfolio_df['Market Value'] - portfolio_df['Unrealized PNL']).sum()
    
    return {
        'concentration_risk': hhi,
        'total_return_pct': (total_pnl / total_invested * 100) if total_invested > 0 else 0,
        'largest_position_pct': (portfolio_df['Market Value'].max() / total_value * 100) if total_value > 0 else 0,
        'positions_count': len(portfolio_df),
        'winners': len(portfolio_df[portfolio_df['Unrealized PNL'] > 0]),
        'losers': len(portfolio_df[portfolio_df['Unrealized PNL'] < 0])
    }

def create_position_sizing_chart(portfolio_df):
    """Create position sizing visualization"""
    if portfolio_df.empty:
        return None
    
    fig = px.treemap(
        portfolio_df,
        values='Market Value',
        names='Symbol',
        color='Unrealized PNL',
        color_continuous_scale='RdYlGn',
        title="Position Sizing & P&L Heatmap"
    )
    
    fig.update_layout(height=500)
    return fig

def create_sector_exposure_chart(portfolio_df):
    """Create mock sector exposure chart"""
    if portfolio_df.empty:
        return None
    
    # Mock sector assignments
    sector_map = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
    }
    
    portfolio_df['Sector'] = portfolio_df['Symbol'].map(sector_map).fillna('Other')
    sector_exposure = portfolio_df.groupby('Sector')['Market Value'].sum().reset_index()
    
    fig = px.pie(
        sector_exposure,
        values='Market Value',
        names='Sector',
        title="Sector Exposure"
    )
    
    fig.update_layout(height=400)
    return fig

def create_pnl_distribution_chart(portfolio_df):
    """Create P&L distribution chart"""
    if portfolio_df.empty:
        return None
    
    fig = px.bar(
        portfolio_df.sort_values('Unrealized PNL'),
        x='Symbol',
        y='Unrealized PNL',
        color='Unrealized PNL',
        color_continuous_scale='RdYlGn',
        title="P&L by Position"
    )
    
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45
    )
    return fig

def render_hedge_fund_analytics_tab(trading_engine):
    """Render the Hedge Fund Analytics tab"""
    st.markdown("## 🏦 Hedge Fund Portfolio Analytics")
    
    if not trading_engine or not trading_engine.positions:
        st.info("No hedge fund positions available for analysis.")
        return
    
    # Get live portfolio data
    cfg = get_config()
    hf_portfolio = trading_engine.positions
    
    try:
        live_prices = {s: pro_get_real_time_data(s, cfg['eodhd_api_key']).get('close', p['avg_price']) 
                      for s, p in hf_portfolio.items()}
        
        positions_data = [{
            'Symbol': symbol, 
            'Quantity': pos['quantity'], 
            'Avg Price': pos['avg_price'],
            'Current Price': live_prices.get(symbol, pos['avg_price']),
            'Market Value': pos['quantity'] * live_prices.get(symbol, pos['avg_price']),
            'Unrealized PNL': (live_prices.get(symbol, pos['avg_price']) - pos['avg_price']) * pos['quantity'],
            'Return %': ((live_prices.get(symbol, pos['avg_price']) - pos['avg_price']) / pos['avg_price'] * 100) if pos['avg_price'] > 0 else 0
        } for symbol, pos in hf_portfolio.items()]
        
        portfolio_df = pd.DataFrame(positions_data)
        
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        # Fallback to static data
        positions_data = [{
            'Symbol': symbol, 
            'Quantity': pos['quantity'], 
            'Avg Price': pos['avg_price'],
            'Current Price': pos['avg_price'],
            'Market Value': pos['quantity'] * pos['avg_price'],
            'Unrealized PNL': 0,
            'Return %': 0
        } for symbol, pos in hf_portfolio.items()]
        portfolio_df = pd.DataFrame(positions_data)
    
    # Calculate risk metrics
    risk_metrics = calculate_portfolio_risk_metrics(portfolio_df)
    
    # Risk Metrics Dashboard
    st.markdown("### 📊 Risk & Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        concentration_color = "red" if risk_metrics.get('concentration_risk', 0) > 0.25 else "green"
        st.metric(
            "Concentration Risk (HHI)",
            f"{risk_metrics.get('concentration_risk', 0):.3f}",
            help="Higher values indicate more concentrated risk"
        )
        st.markdown(f"<small style='color:{concentration_color}'>{'High Risk' if risk_metrics.get('concentration_risk', 0) > 0.25 else 'Diversified'}</small>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Total Return",
            f"{risk_metrics.get('total_return_pct', 0):.2f}%"
        )
    
    with col3:
        st.metric(
            "Largest Position",
            f"{risk_metrics.get('largest_position_pct', 0):.1f}%"
        )
    
    with col4:
        win_rate = (risk_metrics.get('winners', 0) / max(risk_metrics.get('positions_count', 1), 1)) * 100
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{risk_metrics.get('winners', 0)}/{risk_metrics.get('positions_count', 0)}"
        )
    
    # Portfolio Composition
    st.markdown("### 📈 Portfolio Composition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        position_chart = create_position_sizing_chart(portfolio_df)
        if position_chart:
            st.plotly_chart(position_chart, use_container_width=True)
    
    with col2:
        sector_chart = create_sector_exposure_chart(portfolio_df)
        if sector_chart:
            st.plotly_chart(sector_chart, use_container_width=True)
    
    # P&L Analysis
    st.markdown("### 💰 P&L Analysis")
    
    pnl_chart = create_pnl_distribution_chart(portfolio_df)
    if pnl_chart:
        st.plotly_chart(pnl_chart, use_container_width=True)
    
    # Detailed Portfolio Table
    st.markdown("### 📋 Detailed Holdings")
    
    # Add color coding for P&L
    def color_pnl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'color: {color}'
    
    styled_df = portfolio_df.style.applymap(color_pnl, subset=['Unrealized PNL', 'Return %'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Risk Assessment
    st.markdown("### ⚠️ Risk Assessment")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("**Position Risk Analysis**")
        
        # Risk warnings
        if risk_metrics.get('concentration_risk', 0) > 0.25:
            st.warning("⚠️ High concentration risk detected")
        
        if risk_metrics.get('largest_position_pct', 0) > 20:
            st.warning("⚠️ Largest position exceeds 20% of portfolio")
        
        if win_rate < 50:
            st.warning("⚠️ Win rate below 50%")
        
        if not any([risk_metrics.get('concentration_risk', 0) > 0.25, 
                   risk_metrics.get('largest_position_pct', 0) > 20, 
                   win_rate < 50]):
            st.success("✅ Portfolio risk profile is healthy")
    
    with risk_col2:
        st.markdown("**Risk Recommendations**")
        
        recommendations = []
        
        if risk_metrics.get('concentration_risk', 0) > 0.25:
            recommendations.append("Consider diversifying positions to reduce concentration")
        
        if risk_metrics.get('largest_position_pct', 0) > 20:
            recommendations.append("Reduce size of largest position")
        
        if win_rate < 50:
            recommendations.append("Review loss-making positions for exit opportunities")
        
        if len(portfolio_df) < 10:
            recommendations.append("Consider adding more positions for better diversification")
        
        if not recommendations:
            recommendations.append("Portfolio composition looks balanced")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    
    # Export Options
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = portfolio_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "hedge_fund_portfolio.csv",
            "text/csv"
        )
    
    with col2:
        if st.button("Generate Risk Report"):
            st.info("Risk report generation feature coming soon...")
    
    with col3:
        if st.button("Schedule Risk Alert"):
            st.info("Risk alert scheduling feature coming soon...")
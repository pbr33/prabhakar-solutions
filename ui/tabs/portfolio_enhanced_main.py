# ui/tabs/portfolio_enhanced_main.py
"""
Enhanced Portfolio Tab - Main integration file
This replaces the existing portfolio.py tab with enhanced features
"""
import streamlit as st
import pandas as pd
import random
from datetime import datetime

# Import the new modular components
from ui.tabs.portfolio_overview import render_portfolio_overview_tab
from ui.tabs.hedge_fund_analytics import render_hedge_fund_analytics_tab
from ui.tabs.document_intelligence import render_document_intelligence_tab
from ui.tabs.ai_report_generator import render_ai_report_generator_tab
from ui.tabs.scenario_simulator import render_scenario_simulator_tab
from ui.tabs.market_intelligence import render_market_intelligence_tab
from ui.tabs.benchmarking import render_benchmarking_tab

from config import config

# Keep the original PE portfolio generation function
@st.cache_data
def generate_pe_portfolio_data():
    """Generates mock private equity portfolio data."""
    companies = ['InnovateTech', 'BioHealth Solutions', 'GreenEnergy Corp', 'FinTech Innovations', 'MedDevice Solutions']
    portfolio_data = []
    
    for name in companies:
        invested_capital = random.uniform(25, 150)
        current_valuation = random.uniform(invested_capital * 1.2, invested_capital * 4.5)
        moic = current_valuation / invested_capital
        
        company_data = {
            'Company Name': name,
            'Industry': random.choice(['SaaS', 'Biotech', 'Renewables', 'FinTech', 'MedTech']),
            'Invested Capital (M)': invested_capital,
            'Current Valuation (M)': current_valuation,
            'MOIC': moic,
            'IRR (%)': random.uniform(15, 45),
            'Ownership %': random.uniform(15, 85),
            'Investment Date': f"{random.randint(2019, 2023)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'KPI History': pd.DataFrame({
                'date': pd.to_datetime(['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4', '2024-Q1']),
                'revenue': [random.uniform(15, 25), random.uniform(25, 35), random.uniform(35, 45), random.uniform(45, 55), random.uniform(55, 65)],
                'ebitda': [random.uniform(3, 8), random.uniform(8, 12), random.uniform(12, 18), random.uniform(18, 22), random.uniform(22, 28)],
                'employees': [random.randint(50, 80), random.randint(80, 110), random.randint(110, 140), random.randint(140, 170), random.randint(170, 200)]
            })
        }
        portfolio_data.append(company_data)
    
    return portfolio_data

def render_pe_monitoring_tab(pe_portfolio, llm=None):
    """Render the Private Equity Monitoring sub-tab"""
    st.markdown("## 🏢 Private Equity Portfolio Monitoring")
    
    if not pe_portfolio:
        st.info("No private equity portfolio data available.")
        return
    
    # PE Portfolio Overview
    pe_df = pd.DataFrame([{k: v for k, v in company.items() if k != 'KPI History'} for company in pe_portfolio])
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_invested = pe_df['Invested Capital (M)'].sum()
        st.metric("Total Invested", f"${total_invested:.1f}M")
    
    with col2:
        total_value = pe_df['Current Valuation (M)'].sum()
        st.metric("Total Value", f"${total_value:.1f}M")
    
    with col3:
        total_gain = total_value - total_invested
        st.metric("Unrealized Gain", f"${total_gain:.1f}M", delta=f"{(total_gain/total_invested)*100:.1f}%")
    
    with col4:
        avg_moic = pe_df['MOIC'].mean()
        st.metric("Avg MOIC", f"{avg_moic:.2f}x")
    
    # Portfolio Table
    st.markdown("### 📊 Portfolio Companies")
    
    # Format the dataframe for better display
    display_df = pe_df.copy()
    display_df['Invested Capital (M)'] = display_df['Invested Capital (M)'].round(1)
    display_df['Current Valuation (M)'] = display_df['Current Valuation (M)'].round(1)
    display_df['MOIC'] = display_df['MOIC'].round(2)
    display_df['IRR (%)'] = display_df['IRR (%)'].round(1)
    display_df['Ownership %'] = display_df['Ownership %'].round(1)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Company Deep Dive
    st.markdown("### 🔍 Company Deep Dive")
    
    selected_company_name = st.selectbox("Select Portfolio Company", pe_df['Company Name'].tolist())
    selected_company_data = next(company for company in pe_portfolio if company['Company Name'] == selected_company_name)
    
    # Company Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{selected_company_name}**")
        st.write(f"Industry: {selected_company_data['Industry']}")
        st.write(f"Investment Date: {selected_company_data['Investment Date']}")
        st.write(f"Ownership: {selected_company_data['Ownership %']:.1f}%")
    
    with col2:
        st.write(f"Invested: ${selected_company_data['Invested Capital (M)']:.1f}M")
        st.write(f"Current Value: ${selected_company_data['Current Valuation (M)']:.1f}M")
        st.write(f"MOIC: {selected_company_data['MOIC']:.2f}x")
        st.write(f"IRR: {selected_company_data['IRR (%)']:.1f}%")
    
    # KPI History
    st.markdown(f"#### 📈 KPI History for {selected_company_name}")
    kpi_history = selected_company_data['KPI History']
    
    # Display KPI table
    st.dataframe(kpi_history, use_container_width=True)
    
    # KPI Charts
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Trend', 'EBITDA Trend', 'Employee Growth', 'Margin Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Revenue chart
    fig.add_trace(
        go.Scatter(x=kpi_history['date'], y=kpi_history['revenue'], 
                  mode='lines+markers', name='Revenue', line=dict(color='blue')),
        row=1, col=1
    )
    
    # EBITDA chart
    fig.add_trace(
        go.Scatter(x=kpi_history['date'], y=kpi_history['ebitda'], 
                  mode='lines+markers', name='EBITDA', line=dict(color='green')),
        row=1, col=2
    )
    
    # Employee chart
    fig.add_trace(
        go.Scatter(x=kpi_history['date'], y=kpi_history['employees'], 
                  mode='lines+markers', name='Employees', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Margin analysis
    margin = (kpi_history['ebitda'] / kpi_history['revenue'] * 100).round(1)
    fig.add_trace(
        go.Scatter(x=kpi_history['date'], y=margin, 
                  mode='lines+markers', name='EBITDA Margin %', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_compliance_audit_tab():
    """Render the Compliance & Audit sub-tab"""
    st.markdown("## 📋 Compliance & Audit Trail")
    
    # Compliance Dashboard
    st.markdown("### 🔍 Compliance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Compliance Score", "94%", delta="2%")
    
    with col2:
        st.metric("Open Issues", "3", delta="-2")
    
    with col3:
        st.metric("Last Audit", "Q2 2024")
    
    with col4:
        st.metric("Risk Level", "Low", delta="Improved")
    
    # Regulatory Frameworks
    st.markdown("### 📚 Regulatory Frameworks")
    
    frameworks = {
        "SOX Compliance": {"status": "✅ Compliant", "last_review": "2024-07-15", "next_review": "2024-10-15"},
        "GDPR": {"status": "✅ Compliant", "last_review": "2024-06-30", "next_review": "2024-12-30"},
        "ESG Standards": {"status": "⚠️ In Review", "last_review": "2024-05-20", "next_review": "2024-08-20"},
        "Investment Covenants": {"status": "✅ Compliant", "last_review": "2024-07-01", "next_review": "2024-10-01"}
    }
    
    for framework, details in frameworks.items():
        with st.expander(f"{framework} - {details['status']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Last Review: {details['last_review']}")
                st.write(f"Next Review: {details['next_review']}")
            with col2:
                if st.button(f"Run {framework} Check"):
                    st.info(f"Running compliance check for {framework}...")
    
    # Audit Trail
    st.markdown("### 📜 Audit Trail")
    
    # Mock audit events
    audit_events = [
        {"Timestamp": "2024-08-15 14:30", "User": "john.doe", "Action": "Document Upload", "Details": "Investment memo uploaded", "Status": "✅"},
        {"Timestamp": "2024-08-15 13:45", "User": "jane.smith", "Action": "Report Generated", "Details": "Quarterly investor report", "Status": "✅"},
        {"Timestamp": "2024-08-15 11:20", "User": "mike.wilson", "Action": "Portfolio Update", "Details": "PE valuations updated", "Status": "✅"},
        {"Timestamp": "2024-08-14 16:15", "User": "sarah.jones", "Action": "Compliance Check", "Details": "SOX compliance verification", "Status": "✅"},
        {"Timestamp": "2024-08-14 10:30", "User": "admin", "Action": "System Update", "Details": "Security patches applied", "Status": "✅"}
    ]
    
    audit_df = pd.DataFrame(audit_events)
    st.dataframe(audit_df, use_container_width=True)
    
    # Document Verification
    st.markdown("### 🔐 Document Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Integrity Checks**")
        if st.button("Verify All Documents"):
            with st.spinner("Verifying document integrity..."):
                import time
                time.sleep(2)
                st.success("✅ All documents verified successfully")
    
    with col2:
        st.markdown("**Backup Status**")
        st.write("Last Backup: 2024-08-17 02:00 UTC")
        st.write("Next Backup: 2024-08-18 02:00 UTC")
        if st.button("Manual Backup"):
            st.info("Manual backup initiated...")

def render():
    """Main render function for the enhanced portfolio tab"""
    st.markdown("# 💼 Enhanced Institutional Portfolio Management")
    
    # Get configuration and data
    api_key = config.get_eodhd_api_key()
    llm = st.session_state.get('llm', None)
    trading_engine = st.session_state.get('trading_engine')
    
    # Generate PE portfolio data
    pe_portfolio = generate_pe_portfolio_data()
    
    # Create sub-tabs for different views
    tab_names = [
        "📊 Portfolio Overview",
        "🏦 Hedge Fund Analytics",
        "🏢 PE Monitoring",
        "📄 Document Intelligence",
        "🤖 AI Reports",
        "📋 Compliance & Audit",
        "⚡ Scenario Simulator",
        "🌍 Market Intelligence",
        "🏆 Benchmarking",
    ]

    tabs = st.tabs(tab_names)

    # Portfolio Overview Tab
    with tabs[0]:
        render_portfolio_overview_tab(trading_engine, pe_portfolio)

    # Hedge Fund Analytics Tab
    with tabs[1]:
        render_hedge_fund_analytics_tab(trading_engine)

    # PE Monitoring Tab
    with tabs[2]:
        render_pe_monitoring_tab(pe_portfolio, llm)

    # Document Intelligence Tab
    with tabs[3]:
        render_document_intelligence_tab(llm)

    # AI Reports Tab
    with tabs[4]:
        render_ai_report_generator_tab(trading_engine, pe_portfolio, llm)

    # Compliance & Audit Tab
    with tabs[5]:
        render_compliance_audit_tab()

    # Scenario Simulator Tab
    with tabs[6]:
        render_scenario_simulator_tab(trading_engine)

    # Market Intelligence Tab
    with tabs[7]:
        render_market_intelligence_tab()

    # Benchmarking Tab
    with tabs[8]:
        render_benchmarking_tab()

# For backward compatibility, keep the original render function name
def render_enhanced_portfolio_tab(trading_engine=None, llm=None):
    """Backward compatibility function"""
    render()
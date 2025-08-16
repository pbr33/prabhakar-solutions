import json
from datetime import datetime
from typing import List, Dict
import pandas as pd

def generate_candlestick_analysis_report(llm, symbol: str, patterns: List[Dict], data: pd.DataFrame):
    """Generates a technical analysis report based on detected patterns using an LLM."""
    if not llm: return "LLM not initialized."
    
    patterns_str = "\\n".join([f"- **{p['name']}** ({p['type']}) on {p['date'].strftime('%Y-%m-%d')}" for p in patterns])
    latest_indicators = data.iloc[-1]
    
    prompt = f"""
    Act as an expert technical analyst.
    SECURITY: {symbol}
    
    Detected Candlestick Patterns:
    {patterns_str}

    Current Indicators:
    - RSI: {latest_indicators.get('RSI_14', 'N/A'):.2f}
    - MACD: {latest_indicators.get('MACD_12_26_9', 'N/A'):.2f}

    Task: Synthesize this into a cohesive insight. Then, provide a final signal and confidence in a single JSON object.
    Example: {{"signal": "Bullish", "confidence": 75}}
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"LLM call failed: {e}"

def generate_forecast_analysis_report(llm, symbol: str, forecast_data: pd.DataFrame):
    """Generates a qualitative analysis of an ARIMA forecast using an LLM."""
    if not llm: return "LLM not initialized."

    prompt = f"""
    Act as a quantitative analyst.
    SECURITY: {symbol}
    
    ARIMA Forecast Data (next 30 days):
    {forecast_data.to_string()}

    Task: Provide a professional analysis of this forecast, covering the trend, key levels, and volatility outlook.
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"LLM call failed: {e}"

def perform_rag_analysis(llm, document_text: str, user_query: str):
    """Analyzes a document to answer a user's query using RAG."""
    if not llm: return "LLM not initialized."
    if not user_query: return "Please enter a question."
    if not document_text: return "Please upload a document."

    prompt = f"""
    Context:
    ---
    {document_text}
    ---
    Based only on the context provided, answer the following question.
    Question: {user_query}
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"RAG analysis failed: {e}"

def generate_equity_research_report_content(llm, ticker, year, fundamentals, market_data):
    """Generates the full text content for the equity research report using an LLM."""
    if not llm: return "LLM not initialized."

    prompt = f"""
    Act as a Senior Equity Research Analyst.
    TASK: Write a comprehensive equity research report for {ticker} for fiscal year {year}.
    Base your analysis *only* on the provided data.
    
    PROVIDED DATA:
    - Fundamentals: {json.dumps(fundamentals, indent=2)}
    - Market Data Summary (Last Year):
        - 52-Week High: {market_data['adjusted_close'].max():.2f}
        - 52-Week Low: {market_data['adjusted_close'].min():.2f}
        - Average Volume: {market_data['volume'].mean():,.0f}

    REPORT STRUCTURE:
    1. Executive Summary
    2. Business Description & Industry Overview
    3. Financial Analysis
    4. Valuation
    5. Risk Analysis
    6. Final Recommendation & Price Target
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Report generation failed: {e}"

def generate_investment_memo(llm, company_data, portfolio_data: pd.DataFrame):
    """Generates an investment memo for private equity portfolio companies using an LLM."""
    if not llm: 
        return "LLM not initialized."
    
    # Extract company information
    company_name = company_data.get('Company Name', 'Unknown Company')
    industry = company_data.get('Industry', 'Unknown Industry')
    invested_capital = company_data.get('Invested Capital (M)', 0)
    current_valuation = company_data.get('Current Valuation (M)', 0)
    irr = company_data.get('IRR (%)', 0)
    
    # Calculate MOIC if not present
    moic = current_valuation / invested_capital if invested_capital > 0 else 0
    
    # Portfolio summary
    portfolio_summary = ""
    if not portfolio_data.empty:
        total_market_value = portfolio_data['Market Value'].sum()
        total_pnl = portfolio_data['Unrealized PNL'].sum()
        portfolio_summary = f"""
        Current Portfolio Overview:
        - Total Market Value: ${total_market_value:,.2f}
        - Total Unrealized PNL: ${total_pnl:,.2f}
        - Number of Positions: {len(portfolio_data)}
        """

    prompt = f"""
    Act as a Senior Investment Professional at a Private Equity firm.
    
    TASK: Generate a comprehensive investment memo for our portfolio company.
    
    COMPANY DETAILS:
    - Company Name: {company_name}
    - Industry: {industry}
    - Invested Capital: ${invested_capital:.1f}M
    - Current Valuation: ${current_valuation:.1f}M
    - Multiple on Invested Capital (MOIC): {moic:.1f}x
    - Internal Rate of Return (IRR): {irr:.1f}%
    
    {portfolio_summary}
    
    MEMO STRUCTURE:
    1. Executive Summary
    2. Investment Thesis
    3. Financial Performance
    4. Key Value Creation Initiatives
    5. Risk Assessment
    6. Exit Strategy Considerations
    7. Recommendations for Next Steps
    
    Please provide a professional, detailed memo suitable for presentation to the investment committee.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Investment memo generation failed: {e}"

def generate_board_pack_content(llm, company_data, kpi_history: pd.DataFrame):
    """Generates board pack content for portfolio companies using an LLM."""
    if not llm:
        return "LLM not initialized."
    
    company_name = company_data.get('Company Name', 'Unknown Company')
    industry = company_data.get('Industry', 'Unknown Industry')
    
    # Format KPI history for the prompt
    kpi_summary = ""
    if not kpi_history.empty:
        latest_revenue = kpi_history['revenue'].iloc[-1]
        latest_ebitda = kpi_history['ebitda'].iloc[-1]
        revenue_growth = ((kpi_history['revenue'].iloc[-1] / kpi_history['revenue'].iloc[0]) - 1) * 100
        kpi_summary = f"""
        Latest Financial Metrics:
        - Revenue: ${latest_revenue:.1f}M
        - EBITDA: ${latest_ebitda:.1f}M
        - Revenue Growth (YoY): {revenue_growth:.1f}%
        
        Quarterly Progression:
        {kpi_history.to_string(index=False)}
        """
    
    prompt = f"""
    Act as a Chief Operating Officer preparing a board presentation.
    
    TASK: Create executive-level board pack content for our portfolio company.
    
    COMPANY: {company_name}
    INDUSTRY: {industry}
    
    {kpi_summary}
    
    BOARD PACK SECTIONS:
    1. Executive Dashboard (Key Metrics & KPIs)
    2. Financial Performance Analysis
    3. Operational Highlights
    4. Market Position & Competitive Landscape
    5. Strategic Initiatives Progress
    6. Risk Management Update
    7. Forward-Looking Guidance
    8. Action Items for Board Consideration
    
    Please provide concise, data-driven content suitable for a board of directors meeting.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Board pack generation failed: {e}"
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

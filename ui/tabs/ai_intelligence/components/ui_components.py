# ui/tabs/ai_intelligence/components/ui_components.py
"""
Professional UI component library for AI Trading Intelligence.
Provides reusable, styled components for consistent UI.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import pandas as pd


class UIComponents:
    """Professional UI component library."""
    
    @staticmethod
    def render_header(
        title: str,
        subtitle: str,
        icon: Optional[str] = None,
        show_divider: bool = True
    ):
        """Render professional header with styling."""
        header_html = f"""
        <div class="header-container">
            <h1 class="main-title">
                {icon + ' ' if icon else ''}{title}
            </h1>
            <p class="subtitle">{subtitle}</p>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
        
        if show_divider:
            st.markdown("---")
    
    @staticmethod
    def render_metric_card(
        title: str,
        value: Any,
        delta: Optional[float] = None,
        delta_color: bool = True,
        icon: Optional[str] = None,
        subtitle: Optional[str] = None
    ):
        """Render professional metric card with optional delta."""
        delta_html = ""
        if delta is not None:
            color = "#10b981" if (delta >= 0 and delta_color) else "#ef4444"
            arrow = "↑" if delta >= 0 else "↓"
            delta_html = f"""
            <div class="metric-delta" style="color: {color};">
                {arrow} {abs(delta):.2f}%
            </div>
            """
        
        subtitle_html = f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''
        
        card_html = f"""
        <div class="metric-card">
            <div class="metric-header">
                {icon + ' ' if icon else ''}{title}
            </div>
            <div class="metric-value">{value}</div>
            {delta_html}
            {subtitle_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_signal_card(agent_data: Dict[str, Any]):
        """Render professional agent signal card."""
        signal_colors = {
            "BUY": "#10b981",
            "SELL": "#ef4444",
            "HOLD": "#f59e0b"
        }
        
        color = signal_colors.get(agent_data.get("signal", "HOLD"), "#6b7280")
        confidence = agent_data.get("confidence", 0)
        
        card_html = f"""
        <div class="signal-card" style="border-left: 4px solid {color};">
            <div class="signal-header">
                <span class="agent-emoji">{agent_data.get('emoji', '🤖')}</span>
                <span class="agent-name">{agent_data.get('agent', 'Agent')}</span>
                <span class="signal-badge" style="background: {color};">
                    {agent_data.get('signal', 'HOLD')}
                </span>
            </div>
            
            <div class="confidence-container">
                <div class="confidence-label">Confidence</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence}%; background: {color};">
                        {confidence}%
                    </div>
                </div>
            </div>
            
            <div class="signal-content">
                <div class="signal-reasoning">
                    <strong>Analysis:</strong> {agent_data.get('reasoning', 'No reasoning provided')}
                </div>
                <div class="key-levels">
                    <strong>Key Levels:</strong> {agent_data.get('key_levels', 'No levels provided')}
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_consensus_card(consensus_data: Dict[str, Any]):
        """Render consensus signal card with voting breakdown."""
        signal = consensus_data.get('signal', 'HOLD')
        confidence = consensus_data.get('confidence', 0)
        
        signal_colors = {
            "BUY": "#10b981",
            "SELL": "#ef4444",
            "HOLD": "#f59e0b"
        }
        color = signal_colors.get(signal, "#6b7280")
        
        consensus_html = f"""
        <div class="consensus-card" style="background: linear-gradient(135deg, {color}22, {color}11);">
            <div class="consensus-header">
                <h2 class="consensus-title">🎯 AI CONSENSUS</h2>
                <div class="consensus-signal" style="background: {color};">
                    {signal}
                </div>
            </div>
            
            <div class="consensus-metrics">
                <div class="consensus-confidence">
                    <span class="label">Average Confidence</span>
                    <span class="value">{confidence:.0f}%</span>
                </div>
                <div class="consensus-votes">
                    <span class="vote-item buy">
                        🟢 {consensus_data.get('buy_votes', 0)} BUY
                    </span>
                    <span class="vote-item hold">
                        🟡 {consensus_data.get('hold_votes', 0)} HOLD
                    </span>
                    <span class="vote-item sell">
                        🔴 {consensus_data.get('sell_votes', 0)} SELL
                    </span>
                </div>
            </div>
            
            <div class="consensus-reasoning">
                {consensus_data.get('reasoning', 'Consensus analysis complete.')}
            </div>
        </div>
        """
        st.markdown(consensus_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_loading_skeleton(
        height: int = 200,
        text: Optional[str] = "Loading..."
    ):
        """Render loading skeleton for better UX."""
        skeleton_html = f"""
        <div class="skeleton-container">
            <div class="skeleton-loader" style="height: {height}px;">
                <div class="skeleton-shimmer"></div>
            </div>
            {f'<div class="skeleton-text">{text}</div>' if text else ''}
        </div>
        """
        st.markdown(skeleton_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_error(
        message: str,
        icon: str = "⚠️",
        details: Optional[str] = None
    ):
        """Render professional error message."""
        details_html = f'<div class="error-details">{details}</div>' if details else ''
        
        error_html = f"""
        <div class="alert-container error-container">
            <div class="alert-header">
                <span class="alert-icon">{icon}</span>
                <span class="alert-message">{message}</span>
            </div>
            {details_html}
        </div>
        """
        st.markdown(error_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_success(
        message: str,
        icon: str = "✅",
        details: Optional[str] = None
    ):
        """Render professional success message."""
        details_html = f'<div class="success-details">{details}</div>' if details else ''
        
        success_html = f"""
        <div class="alert-container success-container">
            <div class="alert-header">
                <span class="alert-icon">{icon}</span>
                <span class="alert-message">{message}</span>
            </div>
            {details_html}
        </div>
        """
        st.markdown(success_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_info(
        message: str,
        icon: str = "ℹ️",
        details: Optional[str] = None
    ):
        """Render professional info message."""
        details_html = f'<div class="info-details">{details}</div>' if details else ''
        
        info_html = f"""
        <div class="alert-container info-container">
            <div class="alert-header">
                <span class="alert-icon">{icon}</span>
                <span class="alert-message">{message}</span>
            </div>
            {details_html}
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_warning(
        message: str,
        icon: str = "⚠️",
        details: Optional[str] = None
    ):
        """Render professional warning message."""
        details_html = f'<div class="warning-details">{details}</div>' if details else ''
        
        warning_html = f"""
        <div class="alert-container warning-container">
            <div class="alert-header">
                <span class="alert-icon">{icon}</span>
                <span class="alert-message">{message}</span>
            </div>
            {details_html}
        </div>
        """
        st.markdown(warning_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_progress_bar(
        progress: float,
        text: Optional[str] = None,
        color: str = "#3b82f6"
    ):
        """Render custom progress bar."""
        progress = max(0, min(100, progress))
        
        progress_html = f"""
        <div class="progress-container">
            {f'<div class="progress-text">{text}</div>' if text else ''}
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress}%; background: {color};">
                    {progress:.0f}%
                </div>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_scenario_card(
        scenario_name: str,
        scenario_data: Dict[str, Any],
        expanded: bool = False
    ):
        """Render scenario analysis card."""
        probability = scenario_data.get('probability', 0)
        target_price = scenario_data.get('target_price', 0)
        current_price = scenario_data.get('current_price', 100)
        expected_return = ((target_price / current_price) - 1) * 100
        
        # Color based on expected return
        if expected_return > 10:
            color = "#10b981"
        elif expected_return > 0:
            color = "#3b82f6"
        elif expected_return > -10:
            color = "#f59e0b"
        else:
            color = "#ef4444"
        
        card_html = f"""
        <div class="scenario-card {'expanded' if expanded else ''}">
            <div class="scenario-header">
                <h3 class="scenario-title">{scenario_name}</h3>
                <div class="scenario-probability" style="background: {color}22; color: {color};">
                    {probability}% Probability
                </div>
            </div>
            
            <div class="scenario-metrics">
                <div class="metric-item">
                    <span class="metric-label">Target Price</span>
                    <span class="metric-value">${target_price:.2f}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Expected Return</span>
                    <span class="metric-value" style="color: {color};">
                        {'+' if expected_return >= 0 else ''}{expected_return:.1f}%
                    </span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Timeframe</span>
                    <span class="metric-value">{scenario_data.get('timeframe', 'N/A')}</span>
                </div>
            </div>
            
            <div class="scenario-details">
                <div class="detail-section">
                    <strong>Key Catalysts:</strong>
                    <ul class="catalyst-list">
                        {''.join([f"<li>{catalyst}</li>" for catalyst in scenario_data.get('catalysts', [])])}
                    </ul>
                </div>
                <div class="detail-section">
                    <strong>Conditions:</strong>
                    <p>{scenario_data.get('conditions', 'No conditions specified')}</p>
                </div>
                <div class="detail-section">
                    <strong>Risk Factors:</strong>
                    <p>{scenario_data.get('risk_factors', 'No risk factors specified')}</p>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def create_gauge_chart(
        value: float,
        title: str,
        min_value: float = 0,
        max_value: float = 100,
        color: Optional[str] = None
    ) -> go.Figure:
        """Create a professional gauge chart."""
        if color is None:
            if value > 75:
                color = "#10b981"
            elif value > 50:
                color = "#3b82f6"
            elif value > 25:
                color = "#f59e0b"
            else:
                color = "#ef4444"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': color},
                'steps': [
                    {'range': [min_value, max_value * 0.25], 'color': "#fee2e2"},
                    {'range': [max_value * 0.25, max_value * 0.5], 'color': "#fed7aa"},
                    {'range': [max_value * 0.5, max_value * 0.75], 'color': "#dbeafe"},
                    {'range': [max_value * 0.75, max_value], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12)
        )
        
        return fig
    
    @staticmethod
    def create_distribution_chart(
        data: List[float],
        title: str,
        x_label: str = "Value",
        y_label: str = "Frequency"
    ) -> go.Figure:
        """Create a professional distribution chart."""
        fig = go.Figure(data=[
            go.Histogram(
                x=data,
                nbinsx=50,
                marker=dict(
                    color='#3b82f6',
                    line=dict(color='#1e40af', width=1)
                )
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=400,
            showlegend=False,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
        )
        
        return fig
# ui/tabs/market_intelligence.py
"""
Market Intelligence - Macro Command Center
Real-time (or richly mocked) macro signals, sector heatmaps, economic calendar
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA GENERATORS  (replace with live API feeds when keys available)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_macro_signals():
    """Return current snapshot of key macro indicators."""
    return {
        "Fear & Greed Index": {"value": 62, "prev": 55, "label": "Greed",
                               "range": (0, 100), "color_low": "#ef4444", "color_high": "#10b981"},
        "VIX (Volatility)": {"value": 18.4, "prev": 22.1, "label": "Calm",
                              "range": (10, 80), "color_low": "#10b981", "color_high": "#ef4444"},
        "10Y Treasury Yield": {"value": 4.28, "prev": 4.15, "label": "%",
                               "range": (0, 6), "color_low": "#6366f1", "color_high": "#f59e0b"},
        "USD Index (DXY)": {"value": 104.2, "prev": 103.5, "label": "pts",
                            "range": (80, 120), "color_low": "#10b981", "color_high": "#ef4444"},
        "Gold ($/oz)": {"value": 2345, "prev": 2290, "label": "$",
                        "range": (1500, 3000), "color_low": "#f59e0b", "color_high": "#f59e0b"},
        "WTI Oil ($/bbl)": {"value": 78.4, "prev": 81.2, "label": "$",
                            "range": (40, 120), "color_low": "#10b981", "color_high": "#ef4444"},
    }


def _mock_yield_curve():
    """Generate a yield curve dataset."""
    maturities = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    current   = [5.35, 5.30, 5.20, 5.05, 4.65, 4.38, 4.35, 4.28, 4.55, 4.48]
    one_yr_ago = [5.10, 5.15, 5.30, 5.25, 4.90, 4.50, 4.40, 4.30, 4.45, 4.35]
    return pd.DataFrame({"Maturity": maturities, "Current": current, "1Y Ago": one_yr_ago})


def _mock_sector_performance():
    """Generate sector YTD / MTD performance."""
    sectors = [
        "Technology", "Healthcare", "Financial", "Energy", "Consumer Disc.",
        "Industrials", "Materials", "Real Estate", "Utilities", "Comm. Services"
    ]
    ytd  = [28.4, -3.2, 14.8, 11.2, 9.6, 8.4, -5.1, -12.3, -8.7, 22.1]
    mtd  = [4.2,  1.1,  2.3,  -0.8,  1.9,  0.6, -1.2,  -2.1, -0.5,  3.4]
    qtr  = [12.1, -1.5,  5.9,  4.8,  3.2,  2.9, -2.8,  -6.4, -3.1,  8.7]
    return pd.DataFrame({"Sector": sectors, "YTD %": ytd, "MTD %": mtd, "QTD %": qtr})


def _mock_economic_calendar():
    """Generate upcoming economic events."""
    base = datetime.now()
    events = [
        {"Date": base + timedelta(days=1), "Event": "CPI Inflation (MoM)", "Forecast": "0.3%", "Previous": "0.4%", "Impact": "🔴 High"},
        {"Date": base + timedelta(days=2), "Event": "FOMC Meeting Minutes", "Forecast": "—", "Previous": "—", "Impact": "🔴 High"},
        {"Date": base + timedelta(days=3), "Event": "Initial Jobless Claims", "Forecast": "215K", "Previous": "208K", "Impact": "🟡 Medium"},
        {"Date": base + timedelta(days=5), "Event": "Nonfarm Payrolls", "Forecast": "+185K", "Previous": "+199K", "Impact": "🔴 High"},
        {"Date": base + timedelta(days=6), "Event": "Unemployment Rate", "Forecast": "3.9%", "Previous": "3.9%", "Impact": "🔴 High"},
        {"Date": base + timedelta(days=8), "Event": "PPI (YoY)", "Forecast": "2.1%", "Previous": "2.4%", "Impact": "🟡 Medium"},
        {"Date": base + timedelta(days=10), "Event": "Retail Sales (MoM)", "Forecast": "0.2%", "Previous": "-0.1%", "Impact": "🟡 Medium"},
        {"Date": base + timedelta(days=12), "Event": "Michigan Consumer Sentiment", "Forecast": "76.5", "Previous": "74.0", "Impact": "🟢 Low"},
        {"Date": base + timedelta(days=14), "Event": "GDP Growth (QoQ)", "Forecast": "2.8%", "Previous": "3.1%", "Impact": "🔴 High"},
        {"Date": base + timedelta(days=16), "Event": "Core PCE (Fed's Preferred)", "Forecast": "2.6%", "Previous": "2.8%", "Impact": "🔴 High"},
    ]
    for e in events:
        e["Date"] = e["Date"].strftime("%b %d, %Y")
    return pd.DataFrame(events)


def _mock_historical_vix(days=180):
    """Generate fake VIX history."""
    dates = pd.date_range(end=datetime.now(), periods=days)
    vix = [20]
    for _ in range(days - 1):
        vix.append(max(10, min(80, vix[-1] + random.gauss(0, 1.2))))
    return pd.DataFrame({"Date": dates, "VIX": vix})


def _mock_fear_greed_history(days=90):
    """Generate fake Fear & Greed history."""
    dates = pd.date_range(end=datetime.now(), periods=days)
    fg = [50]
    for _ in range(days - 1):
        fg.append(max(0, min(100, fg[-1] + random.gauss(0.3, 3))))
    return pd.DataFrame({"Date": dates, "Score": [round(x) for x in fg]})


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _gauge_card(label, value, prev, min_v, max_v, high_is_good=True):
    """Render a Plotly gauge card for a single macro indicator."""
    delta = value - prev
    range_pct = (value - min_v) / (max_v - min_v)

    if high_is_good:
        bar_color = "#10b981" if range_pct > 0.5 else "#ef4444"
    else:
        bar_color = "#ef4444" if range_pct > 0.5 else "#10b981"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": prev, "valueformat": ".2f",
               "increasing": {"color": "#10b981" if high_is_good else "#ef4444"},
               "decreasing": {"color": "#ef4444" if high_is_good else "#10b981"}},
        gauge={
            "axis": {"range": [min_v, max_v], "tickfont": {"color": "#64748b", "size": 9}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [min_v, min_v + (max_v - min_v) * 0.33], "color": "rgba(239,68,68,0.12)"},
                {"range": [min_v + (max_v - min_v) * 0.33, min_v + (max_v - min_v) * 0.66], "color": "rgba(245,158,11,0.12)"},
                {"range": [min_v + (max_v - min_v) * 0.66, max_v], "color": "rgba(16,185,129,0.12)"},
            ],
        },
        title={"text": label, "font": {"color": "#cbd5e1", "size": 11}},
        number={"font": {"color": "#f1f5f9", "size": 26}, "valueformat": ".1f"},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(15,15,30,0.9)",
        height=200,
        margin=dict(t=40, b=0, l=20, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_market_intelligence_tab():
    """Main render entry for the Market Intelligence tab."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a0a1a 0%, #0f2027 50%, #203a43 100%);
                border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
                border: 1px solid rgba(52,211,153,0.25);">
        <h2 style="margin:0;font-size:2rem;font-weight:800;
                   background:linear-gradient(90deg,#34d399,#60a5fa,#a78bfa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🌍 Market Intelligence
        </h2>
        <p style="margin:8px 0 0;color:#94a3b8;font-size:1rem;">
            Macro command center — live economic signals, yield curve, sector heatmap, and event calendar
            synthesised into a single intelligence dashboard.
        </p>
        <div style="display:inline-flex;align-items:center;gap:8px;margin-top:10px;
                    background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);
                    border-radius:20px;padding:4px 14px;">
            <span style="width:8px;height:8px;border-radius:50%;background:#34d399;
                         display:inline-block;animation:pulse 2s infinite;"></span>
            <span style="color:#34d399;font-size:0.78rem;font-weight:600;">LIVE MACRO FEED</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Macro Gauges ────────────────────────────────────────────────
    st.markdown("### 📡 Macro Signal Dashboard")

    signals = _mock_macro_signals()
    g_cols = st.columns(3)
    items = list(signals.items())

    for i, (label, info) in enumerate(items):
        col = g_cols[i % 3]
        high_good = label not in ("VIX (Volatility)", "10Y Treasury Yield",
                                  "USD Index (DXY)", "WTI Oil ($/bbl)")
        with col:
            fig = _gauge_card(label, info["value"], info["prev"],
                              info["range"][0], info["range"][1], high_good)
            st.plotly_chart(fig, use_container_width=True)

    # ── Section 2: Fear & Greed + VIX History ─────────────────────────────────
    st.markdown("### 📈 Sentiment & Volatility Trends")
    fg_hist = _mock_fear_greed_history()
    vix_hist = _mock_historical_vix()

    fig_trends = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fear & Greed Index (90d)", "VIX — Implied Volatility (180d)"),
    )

    # Fear & Greed line with gradient fill
    fig_trends.add_trace(go.Scatter(
        x=fg_hist["Date"], y=fg_hist["Score"],
        mode="lines", name="F&G Index",
        line=dict(color="#f59e0b", width=2),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.12)",
    ), row=1, col=1)

    fig_trends.add_hline(y=25, line_dash="dot", line_color="#ef4444",
                         annotation_text="Extreme Fear", annotation_font_color="#ef4444",
                         row=1, col=1)
    fig_trends.add_hline(y=75, line_dash="dot", line_color="#10b981",
                         annotation_text="Extreme Greed", annotation_font_color="#10b981",
                         row=1, col=1)

    # VIX line
    fig_trends.add_trace(go.Scatter(
        x=vix_hist["Date"], y=vix_hist["VIX"],
        mode="lines", name="VIX",
        line=dict(color="#6366f1", width=2),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.12)",
    ), row=1, col=2)

    fig_trends.add_hline(y=20, line_dash="dot", line_color="#f59e0b",
                         annotation_text="Elevated Volatility", annotation_font_color="#f59e0b",
                         row=1, col=2)

    fig_trends.update_layout(
        height=380,
        plot_bgcolor="rgba(10,10,25,0.95)",
        paper_bgcolor="rgba(10,10,25,0.95)",
        font=dict(color="#94a3b8"),
        showlegend=False,
        margin=dict(t=50, b=30),
        xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
        xaxis2=dict(gridcolor="rgba(99,102,241,0.1)"),
        yaxis2=dict(gridcolor="rgba(99,102,241,0.1)"),
    )
    fig_trends.update_annotations(font_size=10)
    st.plotly_chart(fig_trends, use_container_width=True)

    # ── Section 3: Yield Curve ─────────────────────────────────────────────────
    st.markdown("### 📉 US Treasury Yield Curve")
    yc = _mock_yield_curve()

    # Calculate 2y10y spread
    spread_2y10y = yc.loc[yc["Maturity"] == "10Y", "Current"].values[0] - \
                   yc.loc[yc["Maturity"] == "2Y", "Current"].values[0]
    spread_color = "#10b981" if spread_2y10y > 0 else "#ef4444"
    spread_label = "NORMAL" if spread_2y10y > 0 else "INVERTED ⚠️"

    col_yc, col_sp = st.columns([3, 1])

    with col_yc:
        fig_yc = go.Figure()
        fig_yc.add_trace(go.Scatter(
            x=yc["Maturity"], y=yc["Current"],
            mode="lines+markers", name="Current Curve",
            line=dict(color="#60a5fa", width=3),
            marker=dict(size=8, color="#60a5fa",
                        line=dict(width=2, color="#1e3a5f")),
        ))
        fig_yc.add_trace(go.Scatter(
            x=yc["Maturity"], y=yc["1Y Ago"],
            mode="lines+markers", name="1 Year Ago",
            line=dict(color="#94a3b8", width=2, dash="dot"),
            marker=dict(size=6, color="#94a3b8"),
        ))
        fig_yc.update_layout(
            height=320,
            plot_bgcolor="rgba(10,10,25,0.95)",
            paper_bgcolor="rgba(10,10,25,0.95)",
            font=dict(color="#94a3b8"),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(title="Maturity", gridcolor="rgba(99,102,241,0.1)"),
            yaxis=dict(title="Yield (%)", gridcolor="rgba(99,102,241,0.1)"),
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_yc, use_container_width=True)

    with col_sp:
        st.markdown(f"""
        <div style="background:rgba(20,20,40,0.9);border:2px solid {spread_color}55;
                    border-radius:14px;padding:24px;text-align:center;margin-top:10px;">
            <div style="color:#94a3b8;font-size:0.8rem;">2Y–10Y Spread</div>
            <div style="color:{spread_color};font-size:2.2rem;font-weight:800;margin:8px 0;">
                {spread_2y10y:+.2f}%
            </div>
            <div style="color:{spread_color};font-size:0.85rem;font-weight:600;">{spread_label}</div>
            <hr style="border-color:rgba(255,255,255,0.1);margin:14px 0;">
            <div style="color:#94a3b8;font-size:0.75rem;">An inverted curve (negative spread) is a historically reliable recession indicator.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Section 4: Sector Heatmap ──────────────────────────────────────────────
    st.markdown("### 🗺️ Sector Performance Heatmap")

    sector_df = _mock_sector_performance()
    period = st.radio("View Period", ["YTD %", "QTD %", "MTD %"], horizontal=True)

    heat_values = sector_df[period].tolist()
    abs_max = max(abs(v) for v in heat_values)

    fig_heat = go.Figure(go.Bar(
        x=sector_df["Sector"],
        y=sector_df[period],
        text=[f"{v:+.1f}%" for v in heat_values],
        textposition="outside",
        marker=dict(
            color=heat_values,
            colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#10b981"]],
            cmin=-abs_max,
            cmax=abs_max,
            line=dict(width=0),
        ),
    ))
    fig_heat.update_layout(
        height=380,
        plot_bgcolor="rgba(10,10,25,0.95)",
        paper_bgcolor="rgba(10,10,25,0.95)",
        font=dict(color="#94a3b8"),
        xaxis=dict(tickangle=-35, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title=f"Performance ({period})", gridcolor="rgba(99,102,241,0.1)",
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.2)", zerolinewidth=1),
        margin=dict(t=20, b=80),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Section 5: Sector spider / radar ──────────────────────────────────────
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        # Radar chart of YTD vs MTD
        top_sectors = sector_df.nlargest(6, "YTD %")
        fig_radar = go.Figure()
        for col_name, color in [("YTD %", "#60a5fa"), ("MTD %", "#34d399")]:
            fig_radar.add_trace(go.Scatterpolar(
                r=top_sectors[col_name].abs().tolist(),
                theta=top_sectors["Sector"].tolist(),
                fill="toself",
                name=col_name,
                line=dict(color=color, width=2),
                fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in color else
                           f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(10,10,25,0.8)",
                radialaxis=dict(visible=True, color="#64748b", gridcolor="rgba(99,102,241,0.2)"),
                angularaxis=dict(color="#94a3b8", gridcolor="rgba(99,102,241,0.2)"),
            ),
            paper_bgcolor="rgba(10,10,25,0.95)",
            font=dict(color="#94a3b8"),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            height=350,
            title=dict(text="Top Sector Performance Radar", font=dict(color="#e2e8f0", size=14)),
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r2:
        # Sortable sector table
        st.markdown("#### Sector Scorecard")
        sector_styled = sector_df.style.background_gradient(
            subset=["YTD %"], cmap="RdYlGn", vmin=-20, vmax=30
        ).background_gradient(
            subset=["MTD %"], cmap="RdYlGn", vmin=-5, vmax=5
        ).background_gradient(
            subset=["QTD %"], cmap="RdYlGn", vmin=-10, vmax=15
        ).format({"YTD %": "{:+.1f}%", "MTD %": "{:+.1f}%", "QTD %": "{:+.1f}%"})
        st.dataframe(sector_styled, use_container_width=True, height=320)

    # ── Section 6: Economic Calendar ───────────────────────────────────────────
    st.markdown("### 📅 Economic Event Calendar")

    cal_df = _mock_economic_calendar()

    impact_filter = st.multiselect(
        "Filter by Impact",
        ["🔴 High", "🟡 Medium", "🟢 Low"],
        default=["🔴 High", "🟡 Medium"],
    )
    filtered_cal = cal_df[cal_df["Impact"].isin(impact_filter)] if impact_filter else cal_df
    st.dataframe(filtered_cal, use_container_width=True, hide_index=True)

    # ── Section 7: Commodity Snapshot ─────────────────────────────────────────
    st.markdown("### 🪙 Commodity & Currency Snapshot")

    commodity_data = {
        "Asset": ["Gold", "Silver", "WTI Crude", "Brent Crude", "Natural Gas",
                  "Copper", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD"],
        "Price": [2345.0, 28.4, 78.4, 82.1, 2.35, 4.21, 1.085, 1.262, 149.8, 67420],
        "Day %": [0.8, -0.4, -1.2, -0.9, 2.1, 0.3, -0.2, 0.1, 0.4, 3.2],
        "Week %": [2.4, -1.8, -3.4, -2.8, 5.6, 1.2, -0.6, 0.4, 1.1, 8.7],
        "YTD %": [14.2, 22.1, 11.4, 9.8, -28.4, 8.9, -3.2, -1.8, 7.6, 42.8],
    }
    comm_df = pd.DataFrame(commodity_data)

    def _color_pct(v):
        if isinstance(v, float):
            return "color: #10b981; font-weight:600;" if v > 0 else "color: #ef4444; font-weight:600;"
        return ""

    styled_comm = comm_df.style.applymap(_color_pct, subset=["Day %", "Week %", "YTD %"])
    st.dataframe(styled_comm, use_container_width=True, hide_index=True)

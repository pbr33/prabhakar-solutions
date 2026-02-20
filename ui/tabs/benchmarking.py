# ui/tabs/benchmarking.py
"""
Benchmarking - Portfolio Alpha Proof
Rolling performance vs indices, alpha/beta/Sharpe attribution, peer comparison
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
# MOCK DATA
# ─────────────────────────────────────────────────────────────────────────────

def _mock_returns(days=365, base_drift=0.00055, volatility=0.012, seed=42):
    np.random.seed(seed)
    daily = np.random.normal(base_drift, volatility, days)
    prices = 100 * np.cumprod(1 + daily)
    return prices


def _build_performance_df(days=365):
    """Simulate portfolio + benchmark time-series."""
    dates = pd.date_range(end=datetime.now(), periods=days)
    portfolio  = _mock_returns(days, 0.00072, 0.011, seed=10)
    sp500      = _mock_returns(days, 0.00055, 0.012, seed=20)
    nasdaq     = _mock_returns(days, 0.00065, 0.016, seed=30)
    msci_world = _mock_returns(days, 0.00048, 0.010, seed=40)
    cambridge  = _mock_returns(days, 0.00040, 0.006, seed=50)  # PE benchmark

    df = pd.DataFrame({
        "Date": dates,
        "Our Portfolio": portfolio,
        "S&P 500": sp500,
        "NASDAQ 100": nasdaq,
        "MSCI World": msci_world,
        "Cambridge PE Benchmark": cambridge,
    })
    return df


def _rolling_sharpe(returns, window=63, rf=0.053 / 252):
    """Compute rolling Sharpe ratio."""
    excess = returns - rf
    roll_mean = excess.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).round(2)


def _rolling_beta(portfolio_ret, benchmark_ret, window=63):
    """Rolling beta vs benchmark."""
    def cov_beta(w_port, w_bench):
        cov = np.cov(w_port, w_bench)[0][1]
        var = np.var(w_bench)
        return cov / var if var > 0 else 1.0

    betas = [np.nan] * window
    for i in range(window, len(portfolio_ret)):
        wp = portfolio_ret[i - window:i]
        wb = benchmark_ret[i - window:i]
        betas.append(cov_beta(wp, wb))
    return pd.Series(betas, index=portfolio_ret.index)


def _compute_stats(port_prices, bench_prices):
    """Compute alpha, beta, Sharpe, max drawdown, etc."""
    p_ret = port_prices.pct_change().dropna()
    b_ret = bench_prices.pct_change().dropna()
    rf = 0.053 / 252  # risk-free daily

    beta_val = np.cov(p_ret, b_ret)[0][1] / np.var(b_ret)
    alpha_annualised = (p_ret.mean() - rf - beta_val * (b_ret.mean() - rf)) * 252 * 100
    sharpe = (p_ret.mean() - rf) / p_ret.std() * np.sqrt(252)
    sortino = (p_ret.mean() - rf) / p_ret[p_ret < 0].std() * np.sqrt(252)

    roll = port_prices / port_prices.cummax() - 1
    max_dd = roll.min() * 100

    total_ret = (port_prices.iloc[-1] / port_prices.iloc[0] - 1) * 100
    bench_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0] - 1) * 100

    return {
        "Total Return (%)": round(total_ret, 1),
        "Benchmark Return (%)": round(bench_ret, 1),
        "Alpha (ann., %)": round(alpha_annualised, 2),
        "Beta": round(beta_val, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Max Drawdown (%)": round(max_dd, 1),
        "Tracking Error (%)": round((p_ret - b_ret).std() * np.sqrt(252) * 100, 2),
        "Information Ratio": round(alpha_annualised / ((p_ret - b_ret).std() * np.sqrt(252) * 100 + 0.001), 2),
    }


def _peer_data():
    """Mock peer fund comparison."""
    return pd.DataFrame({
        "Fund": ["Our Portfolio", "Tiger Global", "Viking Global", "Coatue", "Lone Pine",
                 "D1 Capital", "Whale Rock", "Melvin Capital Successor", "Point72 Multi-Strat"],
        "AUM ($B)": [2.9, 14.0, 18.0, 21.0, 13.0, 12.0, 6.5, 3.2, 25.0],
        "YTD Return (%)": [28.4, 18.2, 22.4, 15.8, 17.6, 24.1, 12.3, 8.9, 19.5],
        "3Y Ann. Return (%)": [19.8, 12.4, 16.8, 11.2, 14.3, 17.9, 8.6, 5.1, 14.7],
        "Sharpe (3Y)": [1.42, 0.88, 1.15, 0.79, 1.02, 1.28, 0.65, 0.41, 1.08],
        "Max Drawdown (%)": [-18.2, -34.8, -28.1, -31.4, -26.7, -22.8, -41.2, -53.6, -15.4],
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render_benchmarking_tab():
    """Main render entry for the Benchmarking & Attribution tab."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0a1628 0%, #0f2847 50%, #1a3a5c 100%);
                border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
                border: 1px solid rgba(96,165,250,0.3);">
        <h2 style="margin:0;font-size:2rem;font-weight:800;
                   background:linear-gradient(90deg,#60a5fa,#a78bfa,#34d399);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🏆 Benchmarking &amp; Alpha Attribution
        </h2>
        <p style="margin:8px 0 0;color:#94a3b8;font-size:1rem;">
            Prove your alpha. Rolling performance vs. S&amp;P 500, NASDAQ, MSCI World and private
            equity peers — with full Sharpe, beta, drawdown and information ratio breakdown.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        benchmarks_selected = st.multiselect(
            "Select benchmarks to compare",
            ["S&P 500", "NASDAQ 100", "MSCI World", "Cambridge PE Benchmark"],
            default=["S&P 500", "NASDAQ 100"],
        )
    with col_ctrl2:
        period_days = st.selectbox(
            "Analysis period",
            {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "3 Years": 756},
            index=3,
        )
        days = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "3 Years": 756}[period_days]

    perf_df = _build_performance_df(days)
    perf_df = perf_df.iloc[-days:]

    # Normalise to 100
    for col in perf_df.columns[1:]:
        perf_df[col] = perf_df[col] / perf_df[col].iloc[0] * 100

    # ── Section 1: Performance Chart ───────────────────────────────────────────
    st.markdown("### 📈 Cumulative Performance")

    colors = {
        "Our Portfolio": "#a78bfa",
        "S&P 500": "#60a5fa",
        "NASDAQ 100": "#34d399",
        "MSCI World": "#f59e0b",
        "Cambridge PE Benchmark": "#fb923c",
    }

    fig_perf = go.Figure()
    cols_to_plot = ["Our Portfolio"] + [b for b in benchmarks_selected if b in perf_df.columns]

    for col in cols_to_plot:
        width = 3 if col == "Our Portfolio" else 2
        dash = "solid" if col == "Our Portfolio" else "dot"
        fig_perf.add_trace(go.Scatter(
            x=perf_df["Date"], y=perf_df[col],
            mode="lines", name=col,
            line=dict(color=colors.get(col, "#94a3b8"), width=width, dash=dash),
        ))

    fig_perf.update_layout(
        height=420,
        plot_bgcolor="rgba(10,10,25,0.95)",
        paper_bgcolor="rgba(10,10,25,0.95)",
        font=dict(color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(title="Indexed Return (Base=100)", gridcolor="rgba(99,102,241,0.1)"),
        margin=dict(t=20, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # ── Section 2: Stats Table ─────────────────────────────────────────────────
    st.markdown("### 📊 Risk-Adjusted Performance Statistics")

    primary_bench = benchmarks_selected[0] if benchmarks_selected else "S&P 500"
    stats = _compute_stats(perf_df["Our Portfolio"], perf_df[primary_bench])

    stat_cols = st.columns(4)
    stat_items = list(stats.items())

    stat_colors = {
        "Total Return (%)": lambda v: "#10b981" if v > 0 else "#ef4444",
        "Alpha (ann., %)": lambda v: "#10b981" if v > 0 else "#ef4444",
        "Beta": lambda v: "#f59e0b",
        "Sharpe Ratio": lambda v: "#10b981" if v > 1.2 else ("#f59e0b" if v > 0.8 else "#ef4444"),
        "Sortino Ratio": lambda v: "#10b981" if v > 1.5 else ("#f59e0b" if v > 1.0 else "#ef4444"),
        "Max Drawdown (%)": lambda v: "#10b981" if v > -15 else ("#f59e0b" if v > -25 else "#ef4444"),
        "Tracking Error (%)": lambda v: "#f59e0b",
        "Information Ratio": lambda v: "#10b981" if v > 0.5 else ("#f59e0b" if v > 0 else "#ef4444"),
        "Benchmark Return (%)": lambda v: "#60a5fa",
    }

    for i, (label, value) in enumerate(stat_items):
        col = stat_cols[i % 4]
        color_fn = stat_colors.get(label, lambda v: "#94a3b8")
        color = color_fn(value)
        val_str = f"{value:+.1f}%" if "%" in label else f"{value:+.2f}" if isinstance(value, float) else str(value)
        col.markdown(
            f"""<div style="background:rgba(20,20,40,0.9);border:1px solid {color}44;
                border-radius:10px;padding:14px;text-align:center;margin-bottom:10px;">
                <div style="color:#64748b;font-size:0.72rem;margin-bottom:6px;">{label}</div>
                <div style="color:{color};font-size:1.4rem;font-weight:800;">{val_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Section 3: Rolling Charts ──────────────────────────────────────────────
    st.markdown("### 🔄 Rolling Analytics")

    port_ret = perf_df["Our Portfolio"].pct_change().dropna()
    bench_ret = perf_df[primary_bench].pct_change().dropna()
    roll_window = min(63, max(21, days // 4))  # adaptive window

    rolling_sharpe = _rolling_sharpe(port_ret, roll_window)
    rolling_beta = _rolling_beta(port_ret, bench_ret, roll_window)
    rolling_alpha = (port_ret - bench_ret).rolling(roll_window).mean() * 252 * 100

    fig_rolling = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"Rolling Sharpe Ratio ({roll_window}d window)",
            f"Rolling Beta vs {primary_bench} ({roll_window}d window)",
            f"Rolling Annualised Alpha (%) vs {primary_bench}",
        ),
        vertical_spacing=0.10,
    )

    # Sharpe
    fig_rolling.add_trace(go.Scatter(
        x=perf_df["Date"].iloc[1:], y=rolling_sharpe,
        mode="lines", name="Sharpe",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
    ), row=1, col=1)
    fig_rolling.add_hline(y=1.0, line_dash="dot", line_color="#10b981",
                          annotation_text="Good (1.0)", annotation_font_color="#10b981",
                          row=1, col=1)

    # Beta
    fig_rolling.add_trace(go.Scatter(
        x=perf_df["Date"].iloc[1:], y=rolling_beta,
        mode="lines", name="Beta",
        line=dict(color="#60a5fa", width=2),
    ), row=2, col=1)
    fig_rolling.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8",
                          annotation_text="Market (β=1)", annotation_font_color="#94a3b8",
                          row=2, col=1)

    # Alpha
    alpha_colors = ["#10b981" if v > 0 else "#ef4444" for v in rolling_alpha.fillna(0)]
    fig_rolling.add_trace(go.Bar(
        x=perf_df["Date"].iloc[1:], y=rolling_alpha,
        name="Alpha %",
        marker_color=alpha_colors,
    ), row=3, col=1)

    for r in range(1, 4):
        fig_rolling.update_xaxes(gridcolor="rgba(99,102,241,0.1)", row=r, col=1)
        fig_rolling.update_yaxes(gridcolor="rgba(99,102,241,0.1)", row=r, col=1)

    fig_rolling.update_layout(
        height=720,
        plot_bgcolor="rgba(10,10,25,0.95)",
        paper_bgcolor="rgba(10,10,25,0.95)",
        font=dict(color="#94a3b8"),
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    fig_rolling.update_annotations(font_size=11, font_color="#94a3b8")
    st.plotly_chart(fig_rolling, use_container_width=True)

    # ── Section 4: Drawdown ────────────────────────────────────────────────────
    st.markdown("### 📉 Drawdown Analysis")

    fig_dd = go.Figure()
    for col in cols_to_plot:
        prices = perf_df[col]
        drawdown = (prices / prices.cummax() - 1) * 100
        width = 2 if col == "Our Portfolio" else 1
        fig_dd.add_trace(go.Scatter(
            x=perf_df["Date"], y=drawdown,
            mode="lines", name=col,
            line=dict(color=colors.get(col, "#94a3b8"), width=width),
            fill="tozeroy",
            fillcolor=colors.get(col, "#94a3b8").replace("#", "rgba(") + ",0.06)".replace("(#", "(")
            if False else "rgba(0,0,0,0)",
        ))

    fig_dd.update_layout(
        height=320,
        plot_bgcolor="rgba(10,10,25,0.95)",
        paper_bgcolor="rgba(10,10,25,0.95)",
        font=dict(color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(title="Drawdown (%)", gridcolor="rgba(99,102,241,0.1)"),
        margin=dict(t=20, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Section 5: Peer Comparison ─────────────────────────────────────────────
    st.markdown("### 🥊 Peer Fund Comparison")

    peer_df = _peer_data()

    # Bubble chart: YTD Return vs Sharpe, bubble = AUM
    fig_bubble = px.scatter(
        peer_df, x="YTD Return (%)", y="Sharpe (3Y)",
        size="AUM ($B)", text="Fund",
        color="YTD Return (%)",
        color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
        size_max=60,
    )
    fig_bubble.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)")),
        textfont=dict(color="#e2e8f0", size=10),
    )
    # Highlight our portfolio
    our_idx = peer_df[peer_df["Fund"] == "Our Portfolio"].index[0]
    fig_bubble.add_trace(go.Scatter(
        x=[peer_df.loc[our_idx, "YTD Return (%)"]],
        y=[peer_df.loc[our_idx, "Sharpe (3Y)"]],
        mode="markers",
        marker=dict(size=22, color="rgba(167,139,250,0.4)",
                    line=dict(width=3, color="#a78bfa")),
        name="Our Portfolio",
        showlegend=False,
    ))

    fig_bubble.update_layout(
        height=420,
        plot_bgcolor="rgba(10,10,25,0.95)",
        paper_bgcolor="rgba(10,10,25,0.95)",
        font=dict(color="#94a3b8"),
        coloraxis_showscale=False,
        xaxis=dict(title="YTD Return (%)", gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(title="3Y Sharpe Ratio", gridcolor="rgba(99,102,241,0.1)"),
        margin=dict(t=20, b=40),
        title=dict(text="YTD Return vs Risk-Adjusted Performance (Bubble = AUM)",
                   font=dict(color="#e2e8f0", size=14)),
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Peer table
    def _color_peer(v):
        if isinstance(v, float):
            if "Return" in str(type(v)) or v > 0:
                return "color:#10b981;font-weight:600;" if v > 0 else "color:#ef4444;font-weight:600;"
        return ""

    peer_styled = peer_df.style\
        .background_gradient(subset=["YTD Return (%)"], cmap="RdYlGn", vmin=5, vmax=30)\
        .background_gradient(subset=["Sharpe (3Y)"], cmap="RdYlGn", vmin=0.4, vmax=1.5)\
        .background_gradient(subset=["Max Drawdown (%)"], cmap="RdYlGn_r", vmin=-55, vmax=-10)\
        .format({
            "AUM ($B)": "${:.1f}B",
            "YTD Return (%)": "{:+.1f}%",
            "3Y Ann. Return (%)": "{:+.1f}%",
            "Sharpe (3Y)": "{:.2f}",
            "Max Drawdown (%)": "{:.1f}%",
        })

    st.dataframe(peer_styled, use_container_width=True, hide_index=True)

    # ── Section 6: Outperformance Summary Bar ──────────────────────────────────
    st.markdown("### ✅ Outperformance Summary")

    our_ytd = peer_df.loc[peer_df["Fund"] == "Our Portfolio", "YTD Return (%)"].values[0]
    outperf = peer_df[peer_df["Fund"] != "Our Portfolio"].copy()
    outperf["Outperformance vs Ours (%)"] = our_ytd - outperf["YTD Return (%)"]

    fig_out = go.Figure(go.Bar(
        x=outperf["Fund"],
        y=outperf["Outperformance vs Ours (%)"],
        text=[f"{v:+.1f}%" for v in outperf["Outperformance vs Ours (%)"]],
        textposition="outside",
        marker_color=["#10b981" if v > 0 else "#ef4444"
                      for v in outperf["Outperformance vs Ours (%)"]],
    ))
    fig_out.update_layout(
        height=340,
        title=dict(text="Our Portfolio Outperformance vs Each Peer (YTD)",
                   font=dict(color="#e2e8f0", size=14)),
        plot_bgcolor="rgba(10,10,25,0.95)",
        paper_bgcolor="rgba(10,10,25,0.95)",
        font=dict(color="#94a3b8"),
        xaxis=dict(tickangle=-35, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Outperformance (%)", gridcolor="rgba(99,102,241,0.1)",
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.25)", zerolinewidth=1),
        margin=dict(t=50, b=100),
    )
    st.plotly_chart(fig_out, use_container_width=True)

import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import config
from services.data_fetcher import (
    pro_get_real_time_data,
    pro_get_historical_data,
    pro_get_fundamental_data,
    pro_get_news,
)

WATCHLIST_FILE = "watchlist.json"

# ── Watchlist persistence ────────────────────────────────────────────────────

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return ["AAPL.US", "MSFT.US", "TSLA.US", "GOOGL.US", "AMZN.US"]


def save_watchlist(watchlist):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist, f)
    except Exception:
        pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def _float(val):
    try:
        return float(val or 0)
    except (TypeError, ValueError):
        return 0.0


def _fmt_large(v):
    try:
        v = float(v or 0)
        if v >= 1e12:
            return f"${v / 1e12:.2f}T"
        if v >= 1e9:
            return f"${v / 1e9:.2f}B"
        if v >= 1e6:
            return f"${v / 1e6:.2f}M"
        if v > 0:
            return f"${v:,.2f}"
        return "N/A"
    except (TypeError, ValueError):
        return "N/A"


# ── Technical indicator computation ──────────────────────────────────────────

def compute_indicators(df):
    """Compute MAs, Bollinger Bands, RSI, MACD on a lowercase-column OHLCV df."""
    close = df["close"]

    df["ma20"]  = close.rolling(20).mean()
    df["ma50"]  = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()

    # Bollinger Bands
    df["bb_mid"]   = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # RSI (14)
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12           = close.ewm(span=12, adjust=False).mean()
    ema26           = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    return df


# ── Chart builder ────────────────────────────────────────────────────────────

def build_chart(df, symbol):
    """3-panel chart: candlestick with MAs/BBands, Volume, RSI."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.58, 0.18, 0.24],
        subplot_titles=[f"{symbol} — Price & Indicators", "Volume", "RSI (14)"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # Bollinger Bands (filled)
    if df["bb_upper"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], name="BB Upper",
            line=dict(color="rgba(180,180,180,0.4)", width=0.8, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], name="BB Lower",
            line=dict(color="rgba(180,180,180,0.4)", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(180,180,180,0.05)",
            showlegend=False,
        ), row=1, col=1)

    # Moving averages
    ma_styles = [
        ("ma20",  "#FFA726", "MA 20"),
        ("ma50",  "#42A5F5", "MA 50"),
        ("ma200", "#AB47BC", "MA 200"),
    ]
    for col_name, color, label in ma_styles:
        if df[col_name].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=label,
                line=dict(color=color, width=1.4),
            ), row=1, col=1)

    # Volume bars coloured by direction
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        name="Volume", marker_color=vol_colors, showlegend=False,
    ), row=2, col=1)

    # RSI
    if df["rsi"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#FF7043", width=1.5),
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)",
                      line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.08)",
                      line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_dash="dash",
                      line=dict(color="#ef5350", width=0.8), row=3, col=1)
        fig.add_hline(y=30, line_dash="dash",
                      line=dict(color="#26a69a", width=0.8), row=3, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#FAFAFA", size=11),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1E2130", showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor="#1E2130", showgrid=True, row=i, col=1)

    return fig


# ── Fundamentals renderer ─────────────────────────────────────────────────────

def render_fundamentals(fund_data):
    if not fund_data:
        st.info("Fundamental data not available for this symbol.")
        return

    highlights = fund_data.get("Highlights", {})
    valuation  = fund_data.get("Valuation",  {})
    technicals = fund_data.get("Technicals", {})
    general    = fund_data.get("General",    {})

    # Revenue / net income from most-recent yearly income statement
    revenue = net_income = 0.0
    yearly_is = fund_data.get("Financials", {}).get("Income_Statement", {}).get("yearly", {})
    if yearly_is:
        try:
            latest     = max(yearly_is.keys())
            revenue    = _float(yearly_is[latest].get("totalRevenue"))
            net_income = _float(yearly_is[latest].get("netIncome"))
        except Exception:
            pass

    # Debt-to-equity from balance sheet
    debt_to_equity = 0.0
    yearly_bs = fund_data.get("Financials", {}).get("Balance_Sheet", {}).get("yearly", {})
    if yearly_bs:
        try:
            latest = max(yearly_bs.keys())
            liab   = _float(yearly_bs[latest].get("totalLiabilities"))
            equity = _float(yearly_bs[latest].get("totalStockholderEquity")) or 1
            debt_to_equity = liab / equity
        except Exception:
            pass

    sector   = general.get("Sector",   "N/A") or "N/A"
    industry = general.get("Industry", "N/A") or "N/A"
    name     = general.get("Name",     "")    or ""

    mc     = _float(highlights.get("MarketCapitalization"))
    ebitda = _float(highlights.get("EBITDA"))
    pe     = _float(highlights.get("PERatio"))
    peg    = _float(highlights.get("PEGRatio"))
    eps    = _float(highlights.get("EarningsShare"))
    dy     = _float(highlights.get("DividendYield")) * 100
    pm     = _float(highlights.get("ProfitMargin"))  * 100
    roe    = _float(highlights.get("ReturnOnEquityTTM")) * 100
    roa    = _float(highlights.get("ReturnOnAssetsTTM")) * 100
    beta   = _float(technicals.get("Beta"))
    hi52   = _float(technicals.get("52WeekHigh"))
    lo52   = _float(technicals.get("52WeekLow"))
    pb     = _float(valuation.get("PriceBookMRQ"))
    ps     = _float(valuation.get("PriceSalesTTM"))
    ev     = _float(valuation.get("EnterpriseValue"))

    if name:
        st.markdown(f"**{name}** · {sector} · {industry}")

    st.markdown("##### Valuation & Size")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Cap",      _fmt_large(mc))
    c2.metric("Enterprise Value", _fmt_large(ev))
    c3.metric("Revenue (TTM)",   _fmt_large(revenue))
    c4.metric("Net Income",      _fmt_large(net_income))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("EBITDA",   _fmt_large(ebitda))
    c6.metric("P/E",      f"{pe:.2f}"  if pe  else "N/A")
    c7.metric("PEG",      f"{peg:.2f}" if peg else "N/A")
    c8.metric("P/B",      f"{pb:.2f}"  if pb  else "N/A")

    st.markdown("##### Profitability & Returns")
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("P/S",           f"{ps:.2f}"  if ps  else "N/A")
    c10.metric("EPS",          f"${eps:.2f}" if eps else "N/A")
    c11.metric("Div. Yield",   f"{dy:.2f}%" if dy  else "N/A")
    c12.metric("Profit Margin", f"{pm:.1f}%")

    c13, c14, c15, c16 = st.columns(4)
    c13.metric("ROE",         f"{roe:.1f}%")
    c14.metric("ROA",         f"{roa:.1f}%")
    c15.metric("Debt/Equity", f"{debt_to_equity:.2f}")
    c16.metric("Beta",        f"{beta:.2f}" if beta else "N/A")

    st.markdown("##### Price Range")
    c17, c18 = st.columns(2)
    c17.metric("52-Week High", f"${hi52:.2f}" if hi52 else "N/A")
    c18.metric("52-Week Low",  f"${lo52:.2f}" if lo52 else "N/A")


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    """Renders the Pro Trading Dashboard."""
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
                padding:1.4rem 2rem; border-radius:12px; margin-bottom:1.5rem;">
        <h1 style="color:#fff; margin:0; font-size:1.9rem; font-weight:700;">
            📊 Pro Trading Dashboard
        </h1>
        <p style="color:#9CA3AF; margin:0.4rem 0 0; font-size:0.9rem;">
            Real-time prices · Live fundamentals · Technical analysis · Market news
        </p>
    </div>
    """, unsafe_allow_html=True)

    api_key = config.get_eodhd_api_key()
    if not api_key:
        st.error("⚠️ EODHD API key required. Please configure it in Settings.")
        return

    symbol = st.session_state.get("selected_symbol", "AAPL.US")

    left_col, right_col = st.columns([1, 3], gap="medium")

    # ── WATCHLIST ──────────────────────────────────────────────────────────
    with left_col:
        st.markdown("### 📋 Watchlist")

        watchlist = load_watchlist()

        # Add ticker via form (clears input after submit)
        with st.form("add_ticker_form", clear_on_submit=True):
            new_ticker = st.text_input(
                "Add ticker", placeholder="e.g. NVDA.US",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("➕ Add to Watchlist", use_container_width=True)

        if submitted and new_ticker:
            t = new_ticker.strip().upper()
            if t not in watchlist:
                watchlist.append(t)
                save_watchlist(watchlist)
                st.success(f"✅ {t} added")
                st.rerun()
            else:
                st.warning(f"{t} already in watchlist")

        st.markdown("---")

        to_remove = None
        for ticker in watchlist:
            rt = pro_get_real_time_data(ticker, api_key)
            price    = change_p = 0.0
            if rt and isinstance(rt, dict):
                price    = _float(rt.get("close") or rt.get("last"))
                change_p = _float(rt.get("change_p"))

            is_selected = ticker == symbol
            border_color = "#3B82F6" if is_selected else (
                "#26a69a" if change_p >= 0 else "#ef5350"
            )
            val_color = "#26a69a" if change_p >= 0 else "#ef5350"
            arrow = "▲" if change_p >= 0 else "▼"

            st.markdown(f"""
            <div style="background:#1E2130; border-radius:8px; padding:0.55rem 0.8rem;
                        margin-bottom:0.35rem; border-left:3px solid {border_color};">
                <div style="font-weight:600; color:#FAFAFA; font-size:0.8rem;">
                    {"🔵 " if is_selected else ""}{ticker}
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:3px;">
                    <span style="color:#FAFAFA; font-size:0.95rem; font-weight:700;">
                        ${price:,.2f}
                    </span>
                    <span style="color:{val_color}; font-size:0.78rem;">
                        {arrow} {abs(change_p):.2f}%
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"✕ {ticker}", key=f"rm_{ticker}", use_container_width=True,
                         help=f"Remove {ticker} from watchlist"):
                to_remove = ticker

        if to_remove:
            watchlist.remove(to_remove)
            save_watchlist(watchlist)
            st.rerun()

    # ── DETAILED VIEW ──────────────────────────────────────────────────────
    with right_col:

        # Live price header
        rt = pro_get_real_time_data(symbol, api_key)
        if rt and isinstance(rt, dict):
            price    = _float(rt.get("close") or rt.get("last"))
            change   = _float(rt.get("change"))
            change_p = _float(rt.get("change_p"))
            volume   = _float(rt.get("volume"))
            open_p   = _float(rt.get("open"))
            high_p   = _float(rt.get("high"))
            low_p    = _float(rt.get("low"))

            price_color = "#26a69a" if change_p >= 0 else "#ef5350"
            arrow       = "▲" if change_p >= 0 else "▼"
            sign        = "+" if change >= 0 else ""

            st.markdown(f"""
            <div style="background:#1E2130; border-radius:12px; padding:1rem 1.5rem; margin-bottom:1rem;">
                <div style="display:flex; align-items:baseline; gap:1.2rem; flex-wrap:wrap;">
                    <span style="font-size:1.5rem; font-weight:700; color:#FAFAFA;">{symbol}</span>
                    <span style="font-size:2.2rem; font-weight:800; color:#FAFAFA;">${price:,.2f}</span>
                    <span style="font-size:1.1rem; color:{price_color}; font-weight:600;">
                        {arrow} {sign}{change:.2f} ({sign}{change_p:.2f}%)
                    </span>
                    <span style="font-size:0.75rem; color:#6B7280; margin-left:auto;">
                        As of {datetime.now().strftime('%b %d, %H:%M')}
                    </span>
                </div>
                <div style="display:flex; gap:2.5rem; margin-top:0.6rem; flex-wrap:wrap;">
                    <span style="color:#9CA3AF; font-size:0.82rem;">
                        Open&nbsp;<strong style="color:#FAFAFA">${open_p:,.2f}</strong>
                    </span>
                    <span style="color:#9CA3AF; font-size:0.82rem;">
                        High&nbsp;<strong style="color:#26a69a">${high_p:,.2f}</strong>
                    </span>
                    <span style="color:#9CA3AF; font-size:0.82rem;">
                        Low&nbsp;<strong style="color:#ef5350">${low_p:,.2f}</strong>
                    </span>
                    <span style="color:#9CA3AF; font-size:0.82rem;">
                        Volume&nbsp;<strong style="color:#FAFAFA">{volume:,.0f}</strong>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.subheader(symbol)

        # Tabs
        tab_chart, tab_fundamentals, tab_news = st.tabs([
            "📈 Chart & Technicals", "💼 Fundamentals", "📰 News",
        ])

        # ── Chart Tab ───────────────────────────────────────────────────
        with tab_chart:
            period_map = {
                "1 Month": 30, "3 Months": 90, "6 Months": 180,
                "1 Year": 365, "2 Years": 730, "5 Years": 1825,
            }
            period_label = st.radio(
                "Period", list(period_map.keys()), index=3, horizontal=True,
                label_visibility="collapsed",
            )
            cutoff_days = period_map[period_label]

            with st.spinner("Loading chart data…"):
                df = pro_get_historical_data(symbol, api_key)

            if df.empty:
                st.error("Could not load historical price data for this symbol.")
            else:
                # Normalise column names (EODHD returns lowercase but guard anyway)
                df.columns = [c.lower() for c in df.columns]

                cutoff = datetime.now() - timedelta(days=cutoff_days)
                df = df[df.index >= pd.Timestamp(cutoff)]

                if df.empty:
                    st.warning("No data available for the selected period.")
                else:
                    df = compute_indicators(df)
                    st.plotly_chart(build_chart(df, symbol), use_container_width=True)

                    # Technical summary row
                    st.markdown("##### 📐 Technical Summary")
                    latest    = df.iloc[-1]
                    close_val = _float(latest.get("close"))

                    ts1, ts2, ts3, ts4, ts5 = st.columns(5)

                    rsi_val   = _float(latest.get("rsi"))
                    rsi_lbl   = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                    rsi_dc    = "inverse" if rsi_val > 70 else "normal" if rsi_val < 30 else "off"
                    ts1.metric("RSI (14)", f"{rsi_val:.1f}", rsi_lbl, delta_color=rsi_dc)

                    macd_val  = _float(latest.get("macd"))
                    macd_sig  = _float(latest.get("macd_signal"))
                    ts2.metric("MACD", f"{macd_val:.3f}",
                               "Bullish" if macd_val > macd_sig else "Bearish",
                               delta_color="normal" if macd_val > macd_sig else "inverse")

                    ma20 = _float(latest.get("ma20"))
                    ts3.metric("MA 20", f"${ma20:.2f}",
                               "Price above" if close_val > ma20 else "Price below",
                               delta_color="normal" if close_val > ma20 else "inverse")

                    ma50 = _float(latest.get("ma50"))
                    ts4.metric("MA 50", f"${ma50:.2f}",
                               "Price above" if close_val > ma50 else "Price below",
                               delta_color="normal" if close_val > ma50 else "inverse")

                    ma200 = _float(latest.get("ma200"))
                    ts5.metric("MA 200", f"${ma200:.2f}" if ma200 else "N/A",
                               "Price above" if ma200 and close_val > ma200 else "Price below" if ma200 else "",
                               delta_color="normal" if ma200 and close_val > ma200 else "inverse")

        # ── Fundamentals Tab ─────────────────────────────────────────────
        with tab_fundamentals:
            with st.spinner("Loading fundamental data…"):
                fund_data = pro_get_fundamental_data(symbol, api_key)
            render_fundamentals(fund_data)

        # ── News Tab ─────────────────────────────────────────────────────
        with tab_news:
            with st.spinner("Loading news…"):
                news = pro_get_news(symbol, api_key)

            if news and isinstance(news, list):
                for article in news[:10]:
                    if not isinstance(article, dict):
                        continue
                    title   = article.get("title", "No title")
                    date    = article.get("date", "")
                    content = (article.get("content") or "")[:250]
                    sent    = article.get("sentiment", "neutral")

                    border = {"positive": "#26a69a", "negative": "#ef5350"}.get(sent, "#FFA726")
                    icon   = {"positive": "🟢",       "negative": "🔴"      }.get(sent, "🟡")

                    st.markdown(f"""
                    <div style="background:#1E2130; border-radius:8px; padding:0.8rem 1rem;
                                margin-bottom:0.6rem; border-left:3px solid {border};">
                        <div style="font-weight:600; color:#FAFAFA; margin-bottom:0.25rem; line-height:1.4;">
                            {icon} {title}
                        </div>
                        <div style="color:#6B7280; font-size:0.73rem; margin-bottom:0.3rem;">{date}</div>
                        <div style="color:#D1D5DB; font-size:0.83rem; line-height:1.5;">
                            {content}{"…" if len(article.get("content") or "") > 250 else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent news available for this symbol.")

# ui/tabs/scenario_simulator.py
"""
Scenario Simulator - Portfolio Stress Testing & What-If Analysis
Interactive tool for simulating market scenarios and measuring portfolio impact
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PRESET SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
PRESET_SCENARIOS = {
    "🔴 2008 Global Financial Crisis": {
        "description": "Lehman Brothers collapse, credit freeze, equity markets -50%",
        "equity_shock": -52,
        "rate_change": -3.5,
        "credit_spread": +4.5,
        "usd_strength": +12,
        "vix_level": 80,
        "tech_multiplier": 0.85,
        "energy_multiplier": 0.60,
        "financial_multiplier": 0.38,
        "healthcare_multiplier": 0.72,
    },
    "🟠 2020 COVID-19 Crash": {
        "description": "Global pandemic shock, fastest 30% drawdown in history",
        "equity_shock": -34,
        "rate_change": -1.5,
        "credit_spread": +3.2,
        "usd_strength": +8,
        "vix_level": 66,
        "tech_multiplier": 1.10,
        "energy_multiplier": 0.45,
        "financial_multiplier": 0.65,
        "healthcare_multiplier": 0.95,
    },
    "🟡 2022 Rate Hike Cycle": {
        "description": "Fed hikes +500bps to combat 40-year high inflation",
        "equity_shock": -20,
        "rate_change": +5.0,
        "credit_spread": +1.8,
        "usd_strength": +15,
        "vix_level": 35,
        "tech_multiplier": 0.65,
        "energy_multiplier": 1.40,
        "financial_multiplier": 0.80,
        "healthcare_multiplier": 0.88,
    },
    "🟢 Bull Market Melt-Up": {
        "description": "AI-driven tech boom, low rates, strong earnings growth",
        "equity_shock": +35,
        "rate_change": -1.0,
        "credit_spread": -0.5,
        "usd_strength": -5,
        "vix_level": 12,
        "tech_multiplier": 1.55,
        "energy_multiplier": 0.90,
        "financial_multiplier": 1.20,
        "healthcare_multiplier": 1.10,
    },
    "⚫ Stagflation (1970s Redux)": {
        "description": "High inflation + economic stagnation + oil shock",
        "equity_shock": -28,
        "rate_change": +4.0,
        "credit_spread": +2.5,
        "usd_strength": -8,
        "vix_level": 45,
        "tech_multiplier": 0.55,
        "energy_multiplier": 1.80,
        "financial_multiplier": 0.70,
        "healthcare_multiplier": 0.85,
    },
    "🔵 China Hard Landing": {
        "description": "Chinese property crisis spills into global markets",
        "equity_shock": -18,
        "rate_change": -0.5,
        "credit_spread": +2.0,
        "usd_strength": +6,
        "vix_level": 42,
        "tech_multiplier": 0.72,
        "energy_multiplier": 0.55,
        "financial_multiplier": 0.68,
        "healthcare_multiplier": 0.90,
    },
}

# Mock portfolio positions with sector tags
MOCK_POSITIONS = {
    "NVDA": {"value": 8_200_000, "sector": "Technology", "beta": 1.75, "asset_type": "HF"},
    "MSFT": {"value": 6_500_000, "sector": "Technology", "beta": 1.10, "asset_type": "HF"},
    "GOOGL": {"value": 5_800_000, "sector": "Technology", "beta": 1.20, "asset_type": "HF"},
    "AAPL": {"value": 5_200_000, "sector": "Technology", "beta": 1.05, "asset_type": "HF"},
    "AMZN": {"value": 4_600_000, "sector": "Technology", "beta": 1.35, "asset_type": "HF"},
    "META": {"value": 3_900_000, "sector": "Technology", "beta": 1.45, "asset_type": "HF"},
    "TSLA": {"value": 3_100_000, "sector": "Technology", "beta": 2.10, "asset_type": "HF"},
    "JPM": {"value": 2_800_000, "sector": "Financial", "beta": 1.15, "asset_type": "HF"},
    "XOM": {"value": 2_200_000, "sector": "Energy", "beta": 0.95, "asset_type": "HF"},
    "JNJ": {"value": 1_800_000, "sector": "Healthcare", "beta": 0.70, "asset_type": "HF"},
    "InnovateTech PE": {"value": 45_000_000, "sector": "Technology", "beta": 0.80, "asset_type": "PE"},
    "BioHealth PE": {"value": 38_000_000, "sector": "Healthcare", "beta": 0.65, "asset_type": "PE"},
    "GreenEnergy PE": {"value": 52_000_000, "sector": "Energy", "beta": 0.70, "asset_type": "PE"},
    "FinTech PE": {"value": 67_000_000, "sector": "Financial", "beta": 0.75, "asset_type": "PE"},
    "MedDevice PE": {"value": 41_000_000, "sector": "Healthcare", "beta": 0.60, "asset_type": "PE"},
}

SECTOR_MULTIPLIER_MAP = {
    "Technology": "tech_multiplier",
    "Energy": "energy_multiplier",
    "Financial": "financial_multiplier",
    "Healthcare": "healthcare_multiplier",
}


def _compute_impact(params: dict) -> pd.DataFrame:
    """
    Calculate position-level P&L impact for a given scenario parameter set.
    Returns a DataFrame with before/after values and impact.
    """
    equity_shock = params["equity_shock"] / 100
    rows = []

    for name, pos in MOCK_POSITIONS.items():
        sector = pos["sector"]
        beta = pos["beta"]
        value = pos["value"]
        asset_type = pos["asset_type"]

        # Sector multiplier from scenario
        sector_key = SECTOR_MULTIPLIER_MAP.get(sector, "tech_multiplier")
        sector_mult = params.get(sector_key, 1.0)

        # PE assets are less liquid → damped beta, longer J-curve lag
        if asset_type == "PE":
            effective_shock = equity_shock * beta * 0.55 * sector_mult
        else:
            effective_shock = equity_shock * beta * sector_mult

        # Rate sensitivity (bonds/PE valuations hurt by rate rises)
        rate_adj = -params["rate_change"] * 0.015 * (0.8 if asset_type == "PE" else 0.3)
        total_shock = effective_shock + rate_adj

        impact_dollars = value * total_shock
        new_value = value + impact_dollars

        rows.append({
            "Position": name,
            "Type": asset_type,
            "Sector": sector,
            "Current Value ($M)": round(value / 1e6, 2),
            "Scenario Value ($M)": round(new_value / 1e6, 2),
            "Impact ($M)": round(impact_dollars / 1e6, 2),
            "Impact (%)": round(total_shock * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("Impact ($M)")


def render_scenario_simulator_tab(trading_engine=None):
    """Main render entry for the Scenario Simulator tab"""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
                border: 1px solid rgba(99,102,241,0.3); position:relative; overflow:hidden;">
        <div style="position:absolute;top:-20px;right:-20px;width:120px;height:120px;
                    background:radial-gradient(circle,rgba(99,102,241,0.25),transparent 70%);border-radius:50%;"></div>
        <h2 style="margin:0;font-size:2rem;font-weight:800;
                   background:linear-gradient(90deg,#a78bfa,#60a5fa,#34d399);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ⚡ Scenario Simulator
        </h2>
        <p style="margin:8px 0 0;color:#94a3b8;font-size:1rem;">
            Stress-test your portfolio against historical crises or build custom macro shocks.
            See real-time P&amp;L impact, VaR shifts, and risk attribution — before markets move.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Total portfolio baseline ───────────────────────────────────────────────
    total_baseline = sum(p["value"] for p in MOCK_POSITIONS.values())
    hf_baseline = sum(p["value"] for p in MOCK_POSITIONS.values() if p["asset_type"] == "HF")
    pe_baseline = sum(p["value"] for p in MOCK_POSITIONS.values() if p["asset_type"] == "PE")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total AUM", f"${total_baseline/1e6:.0f}M")
    with c2:
        st.metric("HF Book", f"${hf_baseline/1e6:.0f}M",
                  delta=f"{hf_baseline/total_baseline*100:.0f}% of AUM")
    with c3:
        st.metric("PE Book", f"${pe_baseline/1e6:.0f}M",
                  delta=f"{pe_baseline/total_baseline*100:.0f}% of AUM")
    with c4:
        st.metric("Positions", str(len(MOCK_POSITIONS)))

    st.markdown("---")

    # ── Scenario selection ────────────────────────────────────────────────────
    mode = st.radio(
        "Choose simulation mode",
        ["🎯 Historical Crisis Scenarios", "🛠️ Custom Macro Builder"],
        horizontal=True,
    )

    params = {}

    if mode == "🎯 Historical Crisis Scenarios":
        selected = st.selectbox(
            "Select a historical scenario",
            list(PRESET_SCENARIOS.keys()),
            index=0,
        )
        params = PRESET_SCENARIOS[selected].copy()

        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08);border-left:4px solid #6366f1;
                    border-radius:8px;padding:14px 18px;margin:12px 0;">
            <strong style="color:#a5b4fc;">Scenario Context</strong><br>
            <span style="color:#cbd5e1;">{params['description']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Show scenario parameters as read-only pills
        p_cols = st.columns(5)
        pill_data = [
            ("Equity Shock", f"{params['equity_shock']:+.0f}%",
             "#ef4444" if params["equity_shock"] < 0 else "#10b981"),
            ("Rate Change", f"{params['rate_change']:+.1f}%",
             "#f59e0b"),
            ("Credit Spread", f"{params['credit_spread']:+.1f}%",
             "#f59e0b"),
            ("USD Strength", f"{params['usd_strength']:+.0f}%",
             "#3b82f6"),
            ("VIX Level", str(params["vix_level"]),
             "#8b5cf6"),
        ]
        for col, (label, val, color) in zip(p_cols, pill_data):
            col.markdown(
                f"""<div style="background:rgba(30,30,50,0.8);border:1px solid {color}55;
                    border-radius:10px;padding:10px;text-align:center;">
                    <div style="color:#94a3b8;font-size:0.72rem;margin-bottom:4px;">{label}</div>
                    <div style="color:{color};font-size:1.25rem;font-weight:700;">{val}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    else:  # Custom builder
        st.markdown("#### Configure your macro shock")
        col_a, col_b = st.columns(2)
        with col_a:
            eq_shock = st.slider("Equity Market Move (%)", -60, 40, -25, 1,
                                 help="Broad market drawdown or rally")
            rate_chg = st.slider("Interest Rate Change (bps → %)", -300, 500, 200, 25,
                                 format="%d bps") / 100
            credit_sp = st.slider("Credit Spread Widening (%)", -1.0, 6.0, 1.5, 0.1)
        with col_b:
            usd_str = st.slider("USD Strength (%)", -20, 20, 5, 1)
            vix_lvl = st.slider("VIX Level", 10, 90, 30, 1)
        st.markdown("**Sector Multipliers** (how hard each sector gets hit relative to market)")
        sm_cols = st.columns(4)
        with sm_cols[0]:
            tech_m = st.slider("Technology", 0.3, 2.0, 0.8, 0.05)
        with sm_cols[1]:
            energy_m = st.slider("Energy", 0.3, 2.0, 1.1, 0.05)
        with sm_cols[2]:
            fin_m = st.slider("Financial", 0.3, 2.0, 0.85, 0.05)
        with sm_cols[3]:
            hc_m = st.slider("Healthcare", 0.3, 2.0, 0.90, 0.05)
        params = {
            "equity_shock": eq_shock,
            "rate_change": rate_chg,
            "credit_spread": credit_sp,
            "usd_strength": usd_str,
            "vix_level": vix_lvl,
            "tech_multiplier": tech_m,
            "energy_multiplier": energy_m,
            "financial_multiplier": fin_m,
            "healthcare_multiplier": hc_m,
            "description": "Custom user-defined scenario",
        }

    # ── Run simulation button ─────────────────────────────────────────────────
    st.markdown("---")
    run_sim = st.button("⚡ Run Simulation", type="primary", use_container_width=True)

    if run_sim or st.session_state.get("sim_ran", False):
        st.session_state["sim_ran"] = True
        st.session_state["sim_params"] = params
    else:
        st.info("👆 Configure a scenario above and click **Run Simulation** to see the impact.")
        return

    params = st.session_state.get("sim_params", params)

    with st.spinner("Running Monte Carlo stress test..."):
        impact_df = _compute_impact(params)

    total_impact = impact_df["Impact ($M)"].sum()
    total_impact_pct = total_impact / (total_baseline / 1e6) * 100
    hf_impact = impact_df[impact_df["Type"] == "HF"]["Impact ($M)"].sum()
    pe_impact = impact_df[impact_df["Type"] == "PE"]["Impact ($M)"].sum()

    # ── KPI cards ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 Simulation Results")
    kpi_cols = st.columns(4)
    kpi_data = [
        ("Portfolio Impact", f"${total_impact:+.1f}M", total_impact),
        ("Impact %", f"{total_impact_pct:+.1f}%", total_impact_pct),
        ("HF Book Impact", f"${hf_impact:+.1f}M", hf_impact),
        ("PE Book Impact", f"${pe_impact:+.1f}M", pe_impact),
    ]
    for col, (label, val, raw) in zip(kpi_cols, kpi_data):
        color = "#10b981" if raw >= 0 else "#ef4444"
        col.markdown(
            f"""<div style="background:linear-gradient(135deg,rgba(30,30,50,0.9),rgba(20,20,40,0.9));
                border:1px solid {color}55;border-radius:12px;padding:18px;text-align:center;">
                <div style="color:#94a3b8;font-size:0.78rem;margin-bottom:6px;">{label}</div>
                <div style="color:{color};font-size:1.6rem;font-weight:800;">{val}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Waterfall chart ────────────────────────────────────────────────────────
    waterfall_df = impact_df.sort_values("Impact ($M)")
    colors = ["#ef4444" if x < 0 else "#10b981" for x in waterfall_df["Impact ($M)"]]

    fig_waterfall = go.Figure(go.Waterfall(
        name="P&L Impact",
        orientation="v",
        measure=["relative"] * len(waterfall_df) + ["total"],
        x=list(waterfall_df["Position"]) + ["NET IMPACT"],
        y=list(waterfall_df["Impact ($M)"]) + [total_impact],
        connector={"line": {"color": "rgba(99,102,241,0.4)", "width": 1}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#6366f1"}},
        text=[f"${v:+.1f}M" for v in list(waterfall_df["Impact ($M)"]) + [total_impact]],
        textposition="outside",
    ))
    fig_waterfall.update_layout(
        title=dict(text="Position-Level Impact Waterfall", font=dict(color="#e2e8f0", size=16)),
        plot_bgcolor="rgba(15,15,30,0.95)",
        paper_bgcolor="rgba(15,15,30,0.95)",
        font=dict(color="#94a3b8"),
        height=500,
        xaxis=dict(tickangle=-45, gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(title="Impact ($M)", gridcolor="rgba(99,102,241,0.1)"),
        margin=dict(t=60, b=120),
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # ── Sector breakdown + Before/After ───────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        # Sector pie of impact
        sector_impact = impact_df.groupby("Sector")["Impact ($M)"].sum().reset_index()
        sector_impact["abs"] = sector_impact["Impact ($M)"].abs()
        fig_sector = px.pie(
            sector_impact, values="abs", names="Sector",
            color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#3b82f6"],
            hole=0.45,
        )
        fig_sector.update_layout(
            title=dict(text="Impact by Sector", font=dict(color="#e2e8f0", size=14)),
            plot_bgcolor="rgba(15,15,30,0.95)",
            paper_bgcolor="rgba(15,15,30,0.95)",
            font=dict(color="#94a3b8"),
            height=360,
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with col_right:
        # Before/After bar
        top_positions = impact_df.tail(8).copy()
        fig_ba = go.Figure()
        fig_ba.add_trace(go.Bar(
            name="Current Value",
            x=top_positions["Position"],
            y=top_positions["Current Value ($M)"],
            marker_color="rgba(99,102,241,0.7)",
        ))
        fig_ba.add_trace(go.Bar(
            name="Scenario Value",
            x=top_positions["Position"],
            y=top_positions["Scenario Value ($M)"],
            marker_color="rgba(239,68,68,0.7)",
        ))
        fig_ba.update_layout(
            title=dict(text="Before vs After (Top 8 Positions)", font=dict(color="#e2e8f0", size=14)),
            barmode="group",
            plot_bgcolor="rgba(15,15,30,0.95)",
            paper_bgcolor="rgba(15,15,30,0.95)",
            font=dict(color="#94a3b8"),
            height=360,
            xaxis=dict(tickangle=-40, gridcolor="rgba(99,102,241,0.1)"),
            yaxis=dict(title="Value ($M)", gridcolor="rgba(99,102,241,0.1)"),
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=50, b=80, l=20, r=20),
        )
        st.plotly_chart(fig_ba, use_container_width=True)

    # ── Risk Gauge ─────────────────────────────────────────────────────────────
    vix = params.get("vix_level", 25)
    risk_score = min(100, max(0, int(vix * 1.1 + abs(total_impact_pct) * 0.8)))

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={"text": "Portfolio Stress Score", "font": {"color": "#e2e8f0", "size": 16}},
        delta={"reference": 30, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": "#6366f1"},
            "steps": [
                {"range": [0, 30], "color": "rgba(16,185,129,0.2)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.2)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.2)"},
            ],
            "threshold": {
                "line": {"color": "#f59e0b", "width": 3},
                "thickness": 0.8,
                "value": 60,
            },
        },
        number={"font": {"color": "#a78bfa", "size": 48}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(15,15,30,0.95)",
        font=dict(color="#94a3b8"),
        height=300,
        margin=dict(t=50, b=20, l=60, r=60),
    )

    col_g, col_t = st.columns([1, 2])
    with col_g:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_t:
        label = "LOW" if risk_score < 30 else ("ELEVATED" if risk_score < 60 else "CRITICAL")
        color = "#10b981" if risk_score < 30 else ("#f59e0b" if risk_score < 60 else "#ef4444")
        st.markdown(f"""
        <div style="background:rgba(30,30,50,0.9);border-left:4px solid {color};
                    border-radius:12px;padding:20px;margin-top:20px;">
            <div style="color:{color};font-size:1.5rem;font-weight:800;">Risk Level: {label}</div>
            <div style="color:#94a3b8;margin-top:12px;line-height:1.7;">
                <b style="color:#cbd5e1;">Stress Score: {risk_score}/100</b><br>
                Total AUM at Risk: <b style="color:#ef4444;">${abs(total_impact):.1f}M</b><br>
                Portfolio Impact: <b style="color:#ef4444;">{total_impact_pct:+.1f}%</b><br>
                HF Drawdown: <b style="color:#f59e0b;">${hf_impact:.1f}M</b><br>
                PE NAV Change: <b style="color:#f59e0b;">${pe_impact:.1f}M</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Detailed table ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Full Position Impact Table")

    def color_impact(val):
        if isinstance(val, float):
            if val < 0:
                return "color: #ef4444; font-weight: 600;"
            elif val > 0:
                return "color: #10b981; font-weight: 600;"
        return ""

    styled = impact_df.style.applymap(color_impact, subset=["Impact ($M)", "Impact (%)"])
    st.dataframe(styled, use_container_width=True, height=400)

    # ── Download ───────────────────────────────────────────────────────────────
    csv = impact_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Stress Test Report (CSV)",
        csv,
        file_name="scenario_stress_test.csv",
        mime="text/csv",
    )

import streamlit as st
import pandas as pd

from config import get_config
from core.trading_bot import TradingBot, MockBrokerAPI

def render():
    """Renders the Auto-Trading Control Center tab."""
    st.markdown("## 🚀 Auto-Trading Control Center")
    cfg = get_config()
    trading_engine = st.session_state.trading_engine
    
    # --- Deploy New Agent Form ---
    with st.expander("🛠️ Deploy a New Trading Agent", expanded=True):
        with st.form("deploy_agent_form"):
            st.markdown("#### Agent Configuration")
            c1, c2, c3 = st.columns(3)
            bot_id = c1.text_input("Agent Name", f"Agent-{len(st.session_state.trading_bots) + 1}")
            bot_symbol = c2.text_input("Symbol", cfg['selected_symbol'])
            bot_trade_qty = c3.number_input("Trade Quantity", min_value=1, value=10)
            
            st.markdown("#### Strategy: MA Crossover")
            c1, c2, c3 = st.columns(3)
            short_window = c1.number_input("Short MA", 1, 100, 20)
            long_window = c2.number_input("Long MA", 2, 200, 50)
            interval = c3.selectbox("Interval (s)", [60, 300, 900], index=1)

            if st.form_submit_button("🚀 Deploy Agent"):
                if bot_id in st.session_state.trading_bots:
                    st.error(f"Agent '{bot_id}' already exists.")
                else:
                    broker = MockBrokerAPI("demo-key", "Simulated Broker")
                    strategy = {'type': 'ma_crossover', 'short_window': short_window, 'long_window': long_window}
                    new_bot = TradingBot(
                        bot_id, bot_symbol, strategy, bot_trade_qty, interval,
                        broker, trading_engine, cfg['eodhd_api_key']
                    )
                    st.session_state.trading_bots[bot_id] = new_bot
                    st.success(f"Agent '{bot_id}' deployed for {bot_symbol}.")
                    st.rerun()

    # --- Live Agent Dashboard ---
    st.markdown("---")
    st.markdown("### 🛰️ Live Agent Dashboard")
    if not st.session_state.trading_bots:
        st.info("No agents deployed.")
    else:
        cols = st.columns(3)
        bot_list = list(st.session_state.trading_bots.values())
        for i, bot in enumerate(bot_list):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"<h5>{bot.bot_id} ({bot.symbol})</h5>", unsafe_allow_html=True)
                    st.write(f"**Status:** {bot.status}")
                    st.caption(f"**Log:** {bot.last_log}")
                    
                    c1, c2 = st.columns(2)
                    if c1.button("▶️ Start", key=f"start_{bot.bot_id}", disabled=bot.is_running):
                        bot.start()
                        st.rerun()
                    if c2.button("⏹️ Stop", key=f"stop_{bot.bot_id}", disabled=not bot.is_running):
                        bot.stop()
                        st.rerun()

    # --- Central Trade Log ---
    st.markdown("---")
    st.markdown("### 📈 Central Portfolio & Trade Log")
    if trading_engine.trade_history:
        trades_df = pd.DataFrame(trading_engine.trade_history).sort_values('timestamp', ascending=False)
        st.dataframe(trades_df)

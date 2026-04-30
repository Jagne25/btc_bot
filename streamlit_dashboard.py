import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Crypto Trading Bot", page_icon="📈", layout="wide")

st.title("📈 Crypto Pullback Trading Bot")
st.caption("Rule-based trend-following system. Monitors BTC, ETH, SOL, BNB every 4 hours automatically.")

# ── LOAD DATA ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_csv(path):
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        df["utc_time"] = pd.to_datetime(df["utc_time"])
        df = df.drop_duplicates(subset=["bar_time"]).sort_values("utc_time").reset_index(drop=True)
        return df
    except:
        return None

btc = load_csv("signals_BTC_vps.csv")
eth = load_csv("signals_ETH_vps.csv")

# ── BACKTEST RESULTS (hardcoded z backttestu) ─────────────────────────────────

backtest = pd.DataFrame([
    {"Coin": "BTC", "EV/trade": "+1.49%", "Return": "+58.1%", "Max DD": "-22.5%", "Trades": 40, "Winrate": "45%"},
    {"Coin": "ETH", "EV/trade": "+0.83%", "Return": "+39.1%", "Max DD": "-32.1%", "Trades": 47, "Winrate": "45%"},
    {"Coin": "SOL", "EV/trade": "+2.91%", "Return": "+136.6%", "Max DD": "-48.4%", "Trades": 47, "Winrate": "38%"},
    {"Coin": "BNB", "EV/trade": "-0.25%", "Return": "-12.6%",  "Max DD": "-39.8%", "Trades": 50, "Winrate": "36%"},
])

# ── CURRENT STATUS ────────────────────────────────────────────────────────────

st.subheader("Current Signal Status")

cols = st.columns(4)
coins_data = [
    ("BTC", btc, "🟡"),
    ("ETH", eth, "🔵"),
    ("SOL", None, "🟣"),
    ("BNB", None, "🟠"),
]

for col, (coin, df_coin, emoji) in zip(cols, coins_data):
    with col:
        if df_coin is not None and len(df_coin) > 0:
            last = df_coin.iloc[-1]
            signal = int(last.get("side", 0))
            close  = float(last.get("close", 0))
            updated = str(last.get("utc_time", ""))[:16]
            if signal == 1:
                st.success(f"{emoji} **{coin}**\n\n🟢 LONG\n\n${close:,.2f}\n\n_{updated}_")
            else:
                st.info(f"{emoji} **{coin}**\n\n⚪ FLAT\n\n${close:,.2f}\n\n_{updated}_")
        else:
            st.info(f"{emoji} **{coin}**\n\n⚪ No data")

st.divider()

# ── STRATEGY + BACKTEST ───────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Strategy Rules")
    st.markdown("""
    **Entry conditions (all must be true):**
    - ✅ Price above MA200 (uptrend)
    - ✅ ADX > 25 (strong trend)
    - ✅ Price 1–8% below MA20 (pullback)
    - ✅ Close above previous high (breakout)

    **Exit:**
    - Trailing stop at 3× ATR
    - Updates every 4 hours automatically
    """)

with col_right:
    st.subheader("Backtest Results (4h, 5 years)")
    st.dataframe(backtest, use_container_width=True, hide_index=True)
    st.caption("EV = average return per trade. Tested on 2021–2026 data.")

st.divider()

# ── BTC PRICE CHART ───────────────────────────────────────────────────────────

if btc is not None and len(btc) > 0:
    st.subheader("BTC Price + LONG Signals")

    longs = btc[btc["side"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=btc["utc_time"], y=btc["close"],
        mode="lines", name="BTC Price",
        line=dict(color="#4895ef", width=1.5)
    ))
    if len(longs) > 0:
        fig.add_trace(go.Scatter(
            x=longs["utc_time"], y=longs["close"],
            mode="markers", name="LONG Entry",
            marker=dict(color="#06d6a0", size=10, symbol="triangle-up")
        ))
    fig.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── RECENT SIGNALS TABLE ──────────────────────────────────────────────────────

st.subheader("Recent Signals (BTC)")
if btc is not None and len(btc) > 0:
    show_cols = ["utc_time", "close", "side", "sl_price", "trail_sl", "atr"]
    available = [c for c in show_cols if c in btc.columns]
    recent = btc[available].tail(15).sort_values("utc_time", ascending=False).reset_index(drop=True)
    recent["side"] = recent["side"].map({1: "🟢 LONG", 0: "⚪ FLAT"})
    st.dataframe(recent, use_container_width=True)

st.divider()
st.caption("Bot runs on Hetzner VPS (Helsinki) · Binance Futures · Python + pandas + PyTorch")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="BTC Bot Dashboard", page_icon="📈", layout="wide")

st.title("📈 BTC Trading Bot — Live Dashboard")
st.caption("An ML-powered trading bot that analyzes BTC/USDT every 4 hours and automatically decides when to enter or exit the market — no manual intervention needed.")

# --- load CSV ---
CSV_PATH = "signals_BTC_vps.csv"

@st.cache_data(ttl=300)
def load_data(path):
    df = pd.read_csv(path, on_bad_lines="skip")
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    df = df.drop_duplicates(subset=["bar_time"]).sort_values("utc_time").reset_index(drop=True)
    return df

try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# --- metrics ---
last = df.iloc[-1]
long_signals = (df["signal"] == 1).sum()
flat_signals  = (df["signal"] == 0).sum()
last_total    = last["test_total_pct"]
last_maxdd    = last["test_maxdd_pct"]
last_trades   = last["trades_test"]
flat_streak   = int((df["signal"] == 0).iloc[::-1].cumprod().sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest Signal", "LONG 🟢" if last["signal"] == 1 else "FLAT ⚪")
col2.metric("ML Probability", f"{last['proba_up']:.3f}")
col3.metric("BTC Price", f"${last['close']:,.0f}")
col4.metric("Backtest Return", f"{last_total:.1f}%")
col5.metric("Backtest Max DD", f"{last_maxdd:.1f}%")

st.divider()

# --- equity curve ---
st.subheader("📊 Simulated Equity Curve (Backtest)")
st.caption("How $1,000 would have grown if the bot traded on historical data.")

equity = [1000.0]
for _, row in df.iterrows():
    prev = equity[-1]
    if row["signal"] == 1:
        ret = (row["close"] - df.loc[max(0, _ - 1), "close"]) / df.loc[max(0, _ - 1), "close"]
        equity.append(prev * (1 + ret * 0.5))
    else:
        equity.append(prev)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=df["utc_time"], y=equity[1:],
    mode="lines", name="Equity",
    line=dict(color="#06d6a0", width=2),
    fill="tozeroy", fillcolor="rgba(6,214,160,0.1)"
))
fig_eq.add_hline(y=1000, line_dash="dash", line_color="#888", annotation_text="Start $1,000")
fig_eq.update_layout(
    template="plotly_dark", height=280,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis_tickprefix="$"
)
st.plotly_chart(fig_eq, use_container_width=True)

st.divider()

# --- BTC price + signals ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("BTC Price + LONG Signals")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["utc_time"], y=df["close"],
        mode="lines", name="BTC close",
        line=dict(color="#4895ef", width=1.5)
    ))
    longs = df[df["signal"] == 1]
    fig.add_trace(go.Scatter(
        x=longs["utc_time"], y=longs["close"],
        mode="markers", name="LONG signal",
        marker=dict(color="#06d6a0", size=8, symbol="triangle-up")
    ))
    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Statistics")
    st.markdown(f"""
    | Metric | Value |
    |---|---|
    | Total signals | {len(df)} |
    | LONG signals | {long_signals} |
    | FLAT signals | {flat_signals} |
    | Trades (backtest) | {int(last_trades)} |
    | Flat streak | {flat_streak} |
    | Model | {last['model']} |
    | Interval | {last['interval']} |
    | Threshold | {last['threshold_long']} |
    """)

st.divider()

# --- ML probability ---
st.subheader("ML Probability Over Time")
st.caption("When probability crosses the threshold (0.50), the bot enters a LONG position.")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df["utc_time"], y=df["proba_up"],
    mode="lines", name="LONG probability",
    line=dict(color="#c77dff", width=1.5)
))
fig2.add_hline(y=float(last["threshold_long"]), line_dash="dash",
               line_color="#ffd166", annotation_text="Entry threshold")
fig2.update_layout(
    template="plotly_dark", height=250,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(range=[0.3, 0.7])
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- last signals table ---
st.subheader("Last 20 Signals")
cols_show = ["utc_time", "close", "proba_up", "signal", "sl_price", "partial_tp", "final_tp"]
st.dataframe(
    df[cols_show].tail(20).sort_values("utc_time", ascending=False).reset_index(drop=True),
    use_container_width=True
)

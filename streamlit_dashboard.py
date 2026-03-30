import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="BTC Bot Dashboard", page_icon="📈", layout="wide")

st.title("📈 BTC Trading Bot — Live Dashboard")
st.caption("ML-based signal bot | BTC/USDT 4h | Logistic Regression")

# --- načítaj CSV ---
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
    st.error(f"Nepodarilo sa načítať CSV: {e}")
    st.stop()

# --- metriky ---
last = df.iloc[-1]
long_signals = (df["signal"] == 1).sum()
flat_signals  = (df["signal"] == 0).sum()
last_total    = last["test_total_pct"]
last_maxdd    = last["test_maxdd_pct"]
last_trades   = last["trades_test"]
flat_streak   = int((df["signal"] == 0).iloc[::-1].cumprod().sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Posledný signál", "LONG 🟢" if last["signal"] == 1 else "FLAT ⚪")
col2.metric("Proba", f"{last['proba_up']:.3f}")
col3.metric("BTC close", f"${last['close']:,.0f}")
col4.metric("Backtest výnos (historický)", f"{last_total:.1f}%")
col5.metric("Backtest Max DD (historický)", f"{last_maxdd:.1f}%")

st.divider()

# --- graf: BTC cena + signály ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("BTC cena + LONG signály")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["utc_time"], y=df["close"],
        mode="lines", name="BTC close",
        line=dict(color="#4895ef", width=1.5)
    ))

    longs = df[df["signal"] == 1]
    fig.add_trace(go.Scatter(
        x=longs["utc_time"], y=longs["close"],
        mode="markers", name="LONG signál",
        marker=dict(color="#06d6a0", size=8, symbol="triangle-up")
    ))

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Štatistiky")
    st.markdown(f"""
    | Metrika | Hodnota |
    |---|---|
    | Celkom signálov | {len(df)} |
    | LONG signálov | {long_signals} |
    | FLAT signálov | {flat_signals} |
    | Obchodov (test) | {int(last_trades)} |
    | Flat streak | {flat_streak} |
    | Model | {last['model']} |
    | Interval | {last['interval']} |
    | Threshold | {last['threshold_long']} |
    """)

st.divider()

# --- graf: proba cez čas ---
st.subheader("ML pravdepodobnosť (proba) cez čas")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df["utc_time"], y=df["proba_up"],
    mode="lines", name="Proba LONG",
    line=dict(color="#c77dff", width=1.5)
))
fig2.add_hline(y=float(last["threshold_long"]), line_dash="dash",
               line_color="#ffd166", annotation_text="Threshold")
fig2.update_layout(
    template="plotly_dark", height=250,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(range=[0.3, 0.7])
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- tabuľka posledných signálov ---
st.subheader("Posledných 20 signálov")
cols_show = ["utc_time", "close", "proba_up", "signal", "sl_price", "partial_tp", "final_tp"]
st.dataframe(
    df[cols_show].tail(20).sort_values("utc_time", ascending=False).reset_index(drop=True),
    use_container_width=True
)

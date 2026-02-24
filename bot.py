import os, datetime as dt, time, requests
import pandas as pd, numpy as np
from dotenv import load_dotenv

# ============ .env ============
load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# FEATURES (NEW)
from features import build_features

# ============ Dáta ============
def get_klines(symbol="BTCUSDT", interval="1d", lookback=1000):
    url = "https://api.binance.com/api/v3/klines"
    rows, end_time = [], None
    while len(rows) < lookback:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        oldest_open = chunk[0][0]
        end_time = oldest_open - 1
        time.sleep(0.03)
        if len(rows) >= lookback:
            break

    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms",  utc=True)
    for c in ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df = df.sort_values("open_time").drop_duplicates(subset="open_time", keep="last").reset_index(drop=True)
    if len(df) > lookback:
        df = df.iloc[-lookback:].reset_index(drop=True)
    return df

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def make_first_touch_label(df: pd.DataFrame, atr_mult_sl: float, horizon: int) -> pd.Series:
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    atr = df["atr14"].values
    n = len(df)
    out = np.full(n, np.nan, dtype=float)

    for t in range(n - 2):
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue

        entry = o[t + 1]
        R = atr_mult_sl * atr[t]
        tp = entry + R
        sl = entry - R

        win_tp_idx = None
        lose_sl_idx = None

        start = t + 2
        end   = min(n, start + int(horizon))
        for j in range(start, end):
            if win_tp_idx is None and h[j] >= tp:
                win_tp_idx = j
            if lose_sl_idx is None and l[j] <= sl:
                lose_sl_idx = j
            if (win_tp_idx is not None) and (lose_sl_idx is not None):
                break

        if win_tp_idx is None and lose_sl_idx is None:
            out[t] = np.nan
        elif lose_sl_idx is None:
            out[t] = 1.0
        elif win_tp_idx is None:
            out[t] = 0.0
        else:
            out[t] = 1.0 if win_tp_idx < lose_sl_idx else 0.0

    return pd.Series(out, index=df.index, dtype="float64")

def fmt(x, nd=2, none_txt="None"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return none_txt
    return f"{x:.{nd}f}"

def perf(eq: pd.Series):
    total = (eq.iloc[-1]-1)*100
    dd = (eq/eq.cummax()-1).min()*100
    return float(total), float(dd)

def main():
    load_dotenv()

    SYMBOL    = os.getenv("SYMBOL", "BTCUSDT")
    INTERVAL  = os.getenv("INTERVAL", "4h")
    MODEL     = os.getenv("MODEL", "LR").upper()

    ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "true").lower() == "true"
    TH_LONG  = float(os.getenv("THRESHOLD_LONG", os.getenv("THRESHOLD", "0.55")))
    TH_SHORT = float(os.getenv("THRESHOLD_SHORT", "0.40"))

    USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "true").lower() == "true"
    SMA_LEN = int(os.getenv("SMA_LEN", "100"))

    LOOKBACK = int(os.getenv("LOOKBACK", "5000"))
    FEE      = float(os.getenv("FEE", "0.0005"))

    USE_ATR     = os.getenv("USE_ATR", "true").lower() == "true"
    ATR_PERIOD  = int(os.getenv("ATR_PERIOD", "14"))
    ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.5"))
    TIME_STOP_BARS = int(os.getenv("TIME_STOP_BARS", "20"))

    TP_R_raw = os.getenv("TP_R", "1.8").strip()
    TP_R = None if TP_R_raw.lower() == "none" else float(TP_R_raw)
    TRAIL_K     = float(os.getenv("TRAIL_K", "2.5"))
    PARTIAL_PCT = float(os.getenv("PARTIAL_PCT", "0.5"))

    EQUITY_USDT = float(os.getenv("EQUITY_USDT", "1000"))
    RISK_PCT    = float(os.getenv("RISK_PCT", "0.01"))

    LOG_CSV = os.getenv("LOG_CSV", f"logs/signals_{SYMBOL.replace('USDT','')}.csv")
    os.makedirs("logs", exist_ok=True)

    now_utc = dt.datetime.now(dt.timezone.utc)
    print(f"[{now_utc:%Y-%m-%d %H:%M} UTC] Fetch {SYMBOL} {INTERVAL}...")

    df = get_klines(SYMBOL, INTERVAL, LOOKBACK)
    if len(df) < 300:
        raise RuntimeError("Malo dat.")

    d = build_features(df)

    if USE_TREND_FILTER:
        d["sma_trend"] = d["close"].rolling(SMA_LEN).mean()

    feats = ["ret1","ret5","vol10","rsi14","price_sma20_ratio","zscore20"]
    
    need_cols = feats + ["close", "atr14"]
    if "close_time" in d.columns:
        need_cols.append("close_time")
    if USE_TREND_FILTER:
        need_cols.append("sma_trend")
    d = d.dropna(subset=need_cols).reset_index(drop=True)
    
    d_live = d.copy()
    
    horizon = TIME_STOP_BARS
    split = int(len(d) * 0.7)
    
    if split - horizon < 50:
        raise RuntimeError(
            f"Príliš málo dát: train={split - horizon} barov (min 50). "
            f"Zníž TIME_STOP_BARS alebo zväčš LOOKBACK."
        )
    
    train = d.iloc[:split - horizon].copy()
    test = d.iloc[split:].copy()
    
    print(f"Buffer split: train={len(train)} | buffer={horizon} | test={len(test)}")
    
    train["label_ft"] = make_first_touch_label(train, atr_mult_sl=ATR_MULT_SL, horizon=TIME_STOP_BARS)
    test["label_ft"] = make_first_touch_label(test, atr_mult_sl=ATR_MULT_SL, horizon=TIME_STOP_BARS)
    
    train = train.dropna(subset=["label_ft"]).reset_index(drop=True)
    test = test.dropna(subset=["label_ft"]).reset_index(drop=True)
    
    Xtr, ytr = train[feats], train["label_ft"].astype(int)
    Xte, yte = test[feats], test["label_ft"].astype(int)

    if MODEL == "RF":
        model = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=10,
                                       random_state=42, n_jobs=-1)
        pipe = Pipeline([("rf", model)]).fit(Xtr, ytr)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("lr", model)]).fit(Xtr, ytr)

    test = test.copy()
    test["proba_up"] = pipe.predict_proba(Xte)[:,1]

    p = np.zeros(len(test), dtype=int)
    long_mask  = test["proba_up"] > TH_LONG
    short_mask = (test["proba_up"] < TH_SHORT) if ALLOW_SHORTS else np.zeros(len(test), dtype=bool)

    if USE_TREND_FILTER and "sma_trend" in test.columns:
        sma = test["sma_trend"]
        long_mask  = long_mask  & (test["close"] > sma)
        short_mask = short_mask & (test["close"] < sma)

    p[long_mask.values]  =  1
    p[np.array(short_mask, dtype=bool)] = -1

    pos_both = pd.Series(p, index=test.index)
    ret = test["close"].pct_change()
    
    # FEE OPRAVA: správne počítanie fee pri flipoch (1->-1 = 2 fees)
    turnover = pos_both.diff().abs()
    fee_events = (turnover > 0).astype(int) + (turnover > 1).astype(int)
    r_plain = pos_both.shift(1).fillna(0) * ret - FEE * fee_events
    
    eq_plain = (1 + r_plain.fillna(0)).cumprod()

    tot_ml, dd_ml = perf(eq_plain)
    trades = int(((pos_both.shift(1).fillna(0) == 0) & (pos_both != 0)).sum())

    last_row = d_live.iloc[-1]
    close = float(last_row["close"])
    proba_live = float(pipe.predict_proba(d_live[feats].iloc[[-1]])[:,1][0])

    if "close_time" in last_row and pd.notna(last_row["close_time"]):
        bar_time = pd.to_datetime(last_row["close_time"])
    else:
        bar_time = pd.to_datetime(df["close_time"].iloc[-1])
    
    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=dt.timezone.utc)
    else:
        bar_time = bar_time.astimezone(dt.timezone.utc)

    trend_ok_long = True
    trend_ok_short = True
    if USE_TREND_FILTER:
        sma_val = float(last_row.get("sma_trend", np.nan))
        if not np.isfinite(sma_val):
            trend_ok_long = trend_ok_short = False
        else:
            trend_ok_long  = close > sma_val
            trend_ok_short = close < sma_val

    side = 0
    if proba_live >= TH_LONG and trend_ok_long:
        side = 1
    elif ALLOW_SHORTS and proba_live <= TH_SHORT and trend_ok_short:
        side = -1

    atr_series = compute_atr(df, period=ATR_PERIOD) if USE_ATR else None
    atr_prev = atr_series.shift(1) if atr_series is not None else None
    atr_last = float(atr_prev.iloc[-1]) if (atr_prev is not None and pd.notna(atr_prev.iloc[-1])) else None

    sl_price = None
    partial_tp = None
    final_tp = None
    trail_sl = None
    qty_suggest = None
    r_dist = None

    if side != 0 and atr_last is not None and np.isfinite(atr_last):
        if side == 1:
            sl_price = close - ATR_MULT_SL*atr_last
            R = max(close - sl_price, 0.0)
            partial_tp = close + R if R > 0 else None
            final_tp   = (None if TP_R is None else (close + TP_R*R)) if R > 0 else None
            trail_sl   = close - TRAIL_K*atr_last
        else:
            sl_price = close + ATR_MULT_SL*atr_last
            R = max(sl_price - close, 0.0)
            partial_tp = close - R if R > 0 else None
            final_tp   = (None if TP_R is None else (close - TP_R*R)) if R > 0 else None
            trail_sl   = close + TRAIL_K*atr_last

        r_dist = R
        risk_usdt = EQUITY_USDT * RISK_PCT
        qty_suggest = (risk_usdt / R) if (R and R > 0) else None

    print(f"MODEL={MODEL} | thL={TH_LONG:.2f} thS={TH_SHORT:.2f} | ALLOW_SHORTS={ALLOW_SHORTS} | TrendFilter={USE_TREND_FILTER}")
    print(f"Test sanity check (plain, correct fees): total={tot_ml:.2f}% | MaxDD={dd_ml:.2f}% | entries={trades}")

    tp_info = (f"TP@{TP_R:.1f}R={fmt(final_tp)}" if TP_R is not None else "TP=None")

    if side == 1:
        print(f"SIGNAL: LONG | proba={proba_live:.3f} | close={fmt(close)} | ATR={fmt(atr_last)} | SL={fmt(sl_price)} | PartialTP={fmt(partial_tp)} | {tp_info} | qty={fmt(qty_suggest,6)}")
    elif side == -1:
        print(f"SIGNAL: SHORT | proba={proba_live:.3f} | close={fmt(close)} | ATR={fmt(atr_last)} | SL={fmt(sl_price)} | PartialTP={fmt(partial_tp)} | {tp_info} | qty={fmt(qty_suggest,6)}")
    else:
        print(f"SIGNAL: FLAT | proba={proba_live:.3f} | close={fmt(close)}")

    out_row = {
        "utc_time": now_utc.strftime("%Y-%m-%d %H:%M"),
        "bar_time": bar_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "model": MODEL,
        "threshold_long": TH_LONG,
        "threshold_short": TH_SHORT,
        "allow_shorts": ALLOW_SHORTS,
        "use_trend_filter": USE_TREND_FILTER,
        "sma_len": SMA_LEN,
        "close": close,
        "proba_up": proba_live,
        "side": side,
        "signal": 1 if side != 0 else 0,
        "atr_period": ATR_PERIOD,
        "atr": atr_last,
        "sl_price": sl_price,
        "r_dist": r_dist,
        "partial_pct": PARTIAL_PCT,
        "partial_tp": partial_tp,
        "final_tp": final_tp,
        "trail_k": TRAIL_K,
        "trail_sl": trail_sl,
        "qty_suggest": qty_suggest,
        "test_total_pct": tot_ml,
        "test_maxdd_pct": dd_ml,
        "trades_test": trades,
        "env_version": os.getenv("ENV_VERSION"),
        "atr_pct":   float(last_row.get("atr_pct"))    if "atr_pct" in d_live.columns else None,
        "vol_rel_20":float(last_row.get("vol_rel_20")) if "vol_rel_20" in d_live.columns else None,
        "dist_hh_50":float(last_row.get("dist_hh_50")) if "dist_hh_50" in d_live.columns else None,
        "dist_ll_50":float(last_row.get("dist_ll_50")) if "dist_ll_50" in d_live.columns else None,
    }

    write_header = not os.path.exists(LOG_CSV)
    pd.DataFrame([out_row]).to_csv(LOG_CSV, index=False, mode="a", header=write_header)
    print(f"Zapisane do {LOG_CSV}")

    try:
        from db_mirror import insert_signal
        insert_signal(out_row)
        print("[INFO] signal mirrored to PostgreSQL")
    except Exception as e:
        print(f"[WARN] mirror insert skipped: {e}")

if __name__ == "__main__":
    main()
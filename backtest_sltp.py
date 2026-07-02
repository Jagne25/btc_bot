# backtest_sltp.py — LONG aj SHORT, LR/RF/ENSEMBLE (OR/AND), risk-based sizing,
# partial @ +1R, TP (vratane None), trailing, trend filter, soft-exit.
# Fixy:
# 1) ATR_prev pri vstupe (ziadny look-ahead): SL/TP a sizing z ATR z predchadzajucej sviecky.
# 2) SIMPLE LABEL (OPRAVENÉ): Správne shift smery + NaN maska (žiadne fillna)
# 3) Rozsirene featury cez features.build_features + trening na rozsirenom sete.
# 4) Filter MIN_TRADES_VAL na vyber "best" konfiguracie na VAL, aby nevyhravali 1-2 trady.
# 5) VAL_MAX_DD filter + score * sqrt(trades) (jemná preferencia väčšej vzorky).
# 6) OBSERVE mód – logovanie social_count_total z public.social_buckets_3h pre FWD segment.
# 7) DEBUG proba distribúcie (VAL aj FWD) + DEBUG ret škály.
# 8) TIME_STOP alignment: H = int(TSTOP) (KRITICKÝ FIX!)
# 9) STARÁ AI DEBUG: time-in-market, baseline EXEC (aligned), Buy&Hold context.
# 10) TREND FILTER SHIFT(1) FIX: Lookahead-free (druhá AI)
# 11) RET EXTRÉMY CHECK: Hygienický test + duplicity check (druhá AI)

import os, datetime as dt, requests, math
import numpy as np, pandas as pd
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import DictCursor

# Reproducibilita
np.random.seed(42)

# nacitaj .env podla ENV_FILE
load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))
print("Pouzivam symbol z ENV:", os.getenv("SYMBOL"))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# FEATURES
from features import build_features

# -------------------- helpers pre .env --------------------

def _parse_float_list(env_key: str, default_csv: str):
    raw = os.getenv(env_key, default_csv)
    out = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(float(x))
        except Exception:
            pass
    return out

def _parse_tp_list(env_key: str, default_csv: str):
    raw = os.getenv(env_key, default_csv)
    out = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        if x.lower() == "none":
            out.append(None)
        else:
            try:
                out.append(float(x))
            except Exception:
                pass
    return out

def _get_float(env_key: str, default_val: float | None):
    val = os.getenv(env_key, None)
    if val is None:
        return default_val
    s = str(val).strip()
    if s == "":
        return default_val
    try:
        return float(s)
    except Exception:
        return default_val

def _get_int(env_key: str, default_val: int | None):
    val = os.getenv(env_key, None)
    if val is None:
        return default_val
    s = str(val).strip()
    if s == "":
        return default_val
    try:
        return int(s)
    except Exception:
        return default_val

def _get_bool(env_key: str, default_val: bool = False):
    val = os.getenv(env_key, None)
    if val is None:
        return default_val
    return str(val).strip().lower() == "true"

# -------------------- PostgreSQL config pre social OBSERVE --------------------

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "")
)

def _pg_conn():
    return psycopg2.connect(**PGCFG)

def fetch_social_counts_for_times(symbol: str, times: list[dt.datetime]) -> dict:
    """
    Vráti slovník: close_time -> count_total z public.social_buckets_3h
    Pre každý bar_time nájde bucket, kde: bucket_start_utc < t <= bucket_end_utc.
    """
    if not times:
        return {}

    # times su pandas Timestamp (UTC) -> premenime na Python datetime (tiez UTC)
    py_times = [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t for t in times]
    t_min = min(py_times)
    t_max = max(py_times)

    # Stiahneme si vsetky buckety pre dany symbol, ktore prekryvaju [t_min, t_max].
    # Podmienka: bucket_end_utc >= t_min A bucket_start_utc <= t_max
    buckets = []
    with _pg_conn() as con, con.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            """
            SELECT symbol, bucket_start_utc, bucket_end_utc, count_total
            FROM public.social_buckets_3h
            WHERE symbol = %s
              AND bucket_end_utc   >= %s
              AND bucket_start_utc <= %s
            ORDER BY bucket_start_utc
            """,
            (symbol, t_min, t_max),
        )
        buckets = cur.fetchall()

    out: dict[dt.datetime, int] = {}
    for t in py_times:
        val = 0
        for b in buckets:
            if b["symbol"] != symbol:
                continue
            bs = b["bucket_start_utc"]
            be = b["bucket_end_utc"]
            # logika: bucket_start < t <= bucket_end
            if t > bs and t <= be:
                val = b["count_total"] or 0
                break
        out[t] = val
    return out

def attach_social_to_segment(seg_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Ku segmentu (napr. FWD) prida stlpec social_count_total podla close_time.
    """
    if "close_time" not in seg_df.columns:
        # build_features by malo nechat close_time z get_klines; ak nie je, nebudeme nic pridavat
        return seg_df

    times = list(seg_df["close_time"])
    sc_map = fetch_social_counts_for_times(symbol, times)
    seg2 = seg_df.copy()
    # mapujeme cez python datetime kluc (pandas Timestamps sa convertnu v sc_map funkcii)
    seg2["social_count_total"] = [sc_map.get(t.to_pydatetime(), 0) for t in seg2["close_time"]]
    return seg2

# -------------------- data --------------------

def get_klines(symbol="BTCUSDT", interval="1d", lookback=1000, start_end_time=None):
    """
    start_end_time: ak je zadaný (unix ms), stiahne lookback sviečok končiacich na tomto čase.
    Používa sa v walk-forward na testovanie rôznych historických okien.
    """
    url = "https://api.binance.com/api/v3/klines"
    out, end_time = [], start_end_time  # začni od zadaného end_time alebo od teraz
    while len(out) < lookback:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        out.extend(chunk)
        oldest_open_ms = chunk[0][0]
        end_time = oldest_open_ms - 1
        if len(out) >= lookback:
            break

    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(out, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    if len(df) > lookback:
        df = df.iloc[-lookback:].reset_index(drop=True)
    return df

# -------------------- First-touch label (+1R vs -1R) --------------------
# POZNAMKA: Tato funkcia je zachovana pre kompatibilitu, ale uz sa nepouziva
# Pouziva sa SIMPLE LABEL namiesto toho

def make_first_touch_label(
    df: pd.DataFrame,
    atr_mult_sl: float,
    horizon: int
) -> pd.Series:
    openp = df["open"].values
    high  = df["high"].values
    low   = df["low"].values
    atr   = df["atr14"].values

    n = len(df)
    out = np.full(n, np.nan, dtype=float)

    for i in range(n-2):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        ent = openp[i+1]
        R   = atr_mult_sl * atr[i]
        tp_long = ent + R
        sl_long = ent - R

        win_tp_idx = None
        lose_sl_idx = None

        start = i + 2
        j_end = min(n, start + int(horizon))
        for j in range(start, j_end):
            if win_tp_idx is None and high[j] >= tp_long:
                win_tp_idx = j
            if lose_sl_idx is None and low[j] <= sl_long:
                lose_sl_idx = j
            if (win_tp_idx is not None) and (lose_sl_idx is not None):
                break

        if win_tp_idx is None and lose_sl_idx is None:
            out[i] = np.nan
        elif lose_sl_idx is None:
            out[i] = 1.0
        elif win_tp_idx is None:
            out[i] = 0.0
        else:
            out[i] = 1.0 if win_tp_idx < lose_sl_idx else 0.0

    return pd.Series(out, index=df.index, dtype="float64")

# -------------------- simulator obchodu (+-1,0) --------------------

def run_backtest(
    df: pd.DataFrame,
    pos: pd.Series,          # -1 | 0 | +1
    proba: pd.Series,
    TH_long: float,
    TH_short: float,
    equity_usdt=1000.0,
    risk_pct=0.01,
    fee=0.001,
    atr_mult_sl=2.0,
    tp_R=2.0,
    partial_pct=0.5,
    trail_k=2.0,
    time_stop_bars=10,
    soft_mode="OFF",
    soft_n=2,
    soft_min_bars=6,
    soft_bias=0.05,
):
    equity = float(equity_usdt)
    eq_curve = [equity]

    in_pos = 0
    qty = 0.0
    entry = None
    sl = None
    tp = None
    bars_in_trade = 0
    did_partial = False
    flat_streak = 0

    for i in range(1, len(df)):
        o = float(df["open"].iloc[i])
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])
        c = float(df["close"].iloc[i])
        atr_curr = float(df["atr14"].iloc[i])

        if pos.iloc[i] == 0:
            flat_streak += 1
        else:
            flat_streak = 0

        # VSTUP
        if in_pos == 0 and pos.iloc[i-1] != 0:
            in_pos = int(pos.iloc[i-1])
            did_partial = False
            bars_in_trade = 0
            entry = o

            atr_prev = float(df["atr14"].iloc[i-1])
            if not np.isfinite(atr_prev) or atr_prev <= 0:
                atr_prev = atr_curr if np.isfinite(atr_curr) and atr_curr > 0 else 1e-8

            if in_pos == 1:
                sl = entry - atr_mult_sl * atr_prev
                R  = max(entry - sl, 1e-8)
                tp = (entry + tp_R * R) if (tp_R is not None) else None
            else:
                sl = entry + atr_mult_sl * atr_prev
                R  = max(sl - entry, 1e-8)
                tp = (entry - tp_R * R) if (tp_R is not None) else None

            risk_usdt = equity * risk_pct
            qty = risk_usdt / R
            equity -= qty * entry * fee
            eq_curve.append(equity)
            continue

        # PRIEBEH
        if in_pos != 0:
            bars_in_trade += 1

            # 1) SL
            sl_hit = (in_pos == 1 and l <= sl) or (in_pos == -1 and h >= sl)
            if sl_hit:
                pnl = (sl - entry) * qty * in_pos
                equity += pnl
                equity -= qty * abs(sl) * fee
                in_pos = 0
                eq_curve.append(equity)
                continue

            R = max(abs(entry - sl), 1e-8)

            # 2) TP
            if tp is not None and qty > 0:
                tp_hit = (in_pos == 1 and h >= tp) or (in_pos == -1 and l <= tp)
                if tp_hit:
                    pnl = (tp - entry) * qty * in_pos
                    equity += pnl
                    equity -= qty * abs(tp) * fee
                    in_pos = 0
                    eq_curve.append(equity)
                    continue

            # 3) PARTIAL @ +1R
            if not did_partial and qty > 0:
                if in_pos == 1:
                    partial_price = entry + R
                    if h >= partial_price:
                        partial_qty = qty * max(0.0, min(1.0, partial_pct))
                        equity += (partial_price - entry) * partial_qty
                        equity -= partial_qty * partial_price * fee
                        qty -= partial_qty
                        did_partial = True
                        sl = max(sl, entry)
                else:
                    partial_price = entry - R
                    if l <= partial_price:
                        partial_qty = qty * max(0.0, min(1.0, partial_pct))
                        equity += (entry - partial_price) * partial_qty
                        equity -= partial_qty * abs(partial_price) * fee
                        qty -= partial_qty
                        did_partial = True
                        sl = min(sl, entry)

            # 4) TRAILING po partiale
            if did_partial and qty > 0 and np.isfinite(atr_curr):
                if in_pos == 1:
                    trail = c - trail_k * atr_curr
                    sl = max(sl, trail)
                else:
                    trail = c + trail_k * atr_curr
                    sl = min(sl, trail)

            # 5) SOFT-EXIT / TIME-STOP
            do_soft = False
            if soft_mode.upper() == "CONFIRM_N_BARS" and bars_in_trade >= int(soft_min_bars):
                if flat_streak >= int(soft_n):
                    do_soft = True
                else:
                    this_TH = TH_long if in_pos == 1 else TH_short
                    if np.isfinite(this_TH) and np.isfinite(soft_bias):
                        if (in_pos == 1 and (proba.iloc[i] <= (this_TH - soft_bias))) or \
                           (in_pos == -1 and (proba.iloc[i] >= (this_TH + soft_bias))):
                            do_soft = True

            do_time = (int(time_stop_bars) > 0) and (bars_in_trade >= int(time_stop_bars))
            if qty > 0 and (do_soft or do_time):
                pnl = (c - entry) * qty * in_pos
                equity += pnl
                equity -= qty * abs(c) * fee
                in_pos = 0
                eq_curve.append(equity)
                continue

        eq_curve.append(equity)

    eq = pd.Series(eq_curve, dtype=float)
    total_pct = (eq.iloc[-1] / eq.iloc[0] - 1) * 100.0
    dd = (eq / eq.cummax() - 1).min() * 100.0
    trades = int((((pos.shift(1).fillna(0)==0) & (pos!=0))).sum())
    return eq, float(total_pct), float(dd), trades

# -------------------- baseline/segment vyhodnotenie --------------------

def evaluate_segment_dual(pipe, seg_df, feats, th_long, th_short, fee, allow_shorts, use_trend):
    proba = pipe.predict_proba(seg_df[feats])[:,1]
    seg = seg_df.copy()
    seg["proba_up"] = proba

    pos = np.zeros(len(seg), dtype=int)
    long_mask = proba > th_long
    if allow_shorts:
        short_mask = proba < th_short
    else:
        short_mask = np.zeros_like(long_mask, dtype=bool)

    if use_trend and "sma_trend" in seg.columns:
        long_mask  = long_mask  & (seg["close"] > seg["sma_trend"]).values
        short_mask = short_mask & (seg["close"] < seg["sma_trend"]).values

    pos[long_mask]  =  1
    pos[short_mask] = -1

    seg["pos"] = pos
    p = pd.Series(pos, index=seg.index)
    r = p.shift(1).fillna(0)*seg["ret"] - fee*(p.diff()!=0).astype(int)
    eq_plain = (1 + r.fillna(0)).cumprod()
    plain_total = float((eq_plain.iloc[-1]-1)*100)
    plain_dd    = float(((eq_plain/eq_plain.cummax())-1).min()*100)
    return seg, p, plain_total, plain_dd

# Ensemble helper (pre pripad, ze zapnes ENSEMBLE_MODE)
def make_ensemble_pos(mode: str,
                      proba_lr: np.ndarray, th_lr_long: float, th_lr_short: float,
                      proba_rf: np.ndarray, th_rf_long: float, th_rf_short: float,
                      allow_shorts: bool):
    mode = (mode or "OR").upper()
    if mode == "AND":
        long_mask = (proba_lr > th_lr_long) & (proba_rf > th_rf_long)
    else:
        long_mask = (proba_lr > th_lr_long) | (proba_rf > th_rf_long)
    if allow_shorts:
        if mode == "AND":
            short_mask = (proba_lr < th_lr_short) & (proba_rf < th_rf_short)
        else:
            short_mask = (proba_lr < th_lr_short) | (proba_rf < th_rf_short)
    else:
        short_mask = np.zeros_like(long_mask, dtype=bool)
    pos = np.zeros_like(long_mask, dtype=int)
    pos[long_mask]  =  1
    pos[short_mask] = -1
    return pos

# -------------------- DEBUG HELPER --------------------

def debug_proba(name: str, tag: str, proba_up):
    """Debug helper pre proba distribúciu"""
    proba_up = np.asarray(proba_up, dtype=float)
    if len(proba_up) == 0:
        print(f"[{name}] {tag}: EMPTY")
        return

    print(f"\n{'='*60}")
    print(f"PROBA DISTRIBUTION ({name} | {tag})")
    print(f"{'='*60}")
    print(f"  Count: {len(proba_up)}")
    print(f"  Min:   {np.min(proba_up):.4f}")
    print(f"  p10:   {np.percentile(proba_up, 10):.4f}")
    print(f"  p25:   {np.percentile(proba_up, 25):.4f}")
    print(f"  p50:   {np.percentile(proba_up, 50):.4f}")
    print(f"  p75:   {np.percentile(proba_up, 75):.4f}")
    print(f"  p90:   {np.percentile(proba_up, 90):.4f}")
    print(f"  p95:   {np.percentile(proba_up, 95):.4f}")
    print(f"  p99:   {np.percentile(proba_up, 99):.4f}")
    print(f"  Max:   {np.max(proba_up):.4f}")
    print(f"\nThreshold analysis (signals = proba > th):")
    for th in [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58]:
        n_signals = int(np.sum(proba_up > th))
        pct = 100.0 * n_signals / len(proba_up)
        print(f"  TH={th:.2f}: {n_signals:4d} signals ({pct:5.1f}%)")
    print(f"{'='*60}\n")

# -------------------- MAIN --------------------

def main():
    load_dotenv()

    SYMBOL    = os.getenv("SYMBOL", "BTCUSDT")
    INTERVAL  = os.getenv("INTERVAL", "1d")
    LOOKBACK  = _get_int("LOOKBACK", 900)
    VAL       = _get_int("VAL", 200)
    FORWARD   = _get_int("FORWARD", 200)
    FEE       = _get_float("FEE", 0.001)

    EQUITY_USDT = _get_float("EQUITY_USDT", 1000.0)
    RISK_PCT    = _get_float("RISK_PCT", 0.01)

    USE_TREND_FILTER = _get_bool("USE_TREND_FILTER", False)
    SMA_LEN = _get_int("SMA_LEN", 100)

    # SHORTS
    ALLOW_SHORTS = _get_bool("ALLOW_SHORTS", False)
    TH_LONG   = _get_float("THRESHOLD_LONG",  0.50)
    TH_SHORT  = _get_float("THRESHOLD_SHORT", 0.50)
    THS_LONG  = _parse_float_list("TH_LIST", "0.45,0.50,0.55,0.60")
    THS_SHORT = _parse_float_list("TH_LIST_SHORT", ",".join(map(str, THS_LONG)))

    # grids
    ATRS   = _parse_float_list("ATR_SL_LIST",  "1.0,1.5,2.0")
    TPRS   = _parse_tp_list   ("TP_R_LIST",    "1.5,1.8,2.0,None")
    TRAILS = _parse_float_list("TRAIL_K_LIST", "2.0,2.5,3.0")
    PARTS  = _parse_float_list("PARTIAL_LIST", "0.3,0.5")
    TSTOP  = _get_int("TIME_STOP_BARS", 12)
    SCORE_ALPHA = _get_float("SCORE_ALPHA", 0.6)

    # labelovy R pre training (uz sa nepouziva, zachovane pre kompatibilitu)
    ATR_MULT_SL_LABEL = _get_float("ATR_LABEL_SL", (ATRS[0] if len(ATRS) else 1.5))

    # soft-exit
    SOFT_EXIT_MODE = os.getenv("SOFT_EXIT", "OFF").upper()
    SOFT_EXIT_N = _get_int("SOFT_EXIT_N", 2)
    SOFT_EXIT_MIN_BARS = _get_int("SOFT_EXIT_MIN_BARS", 6)
    SOFT_EXIT_PROBA_BIAS = _get_float("SOFT_EXIT_PROBA_BIAS", 0.05)

    # Ensemble
    ENSEMBLE_MODE = os.getenv("ENSEMBLE_MODE", "OFF").upper()

    # Robustnost vyberu
    MIN_TRADES_VAL = _get_int("MIN_TRADES_VAL", 20)
    VAL_MAX_DD = _get_float("VAL_MAX_DD", 20.0)

    now_utc = dt.datetime.now(dt.timezone.utc)
    print(f"[{now_utc:%Y-%m-%d %H:%M} UTC] Backtest {SYMBOL} {INTERVAL}")
    print(f"LOOKBACK={LOOKBACK} | VAL={VAL} | FORWARD={FORWARD} | RISK={RISK_PCT*100:.2f}% z {EQUITY_USDT} USDT")
    print(f"Trend filter: {'ON' if USE_TREND_FILTER else 'OFF'} (SMA_LEN={SMA_LEN})")
    print(f"Soft exit: {SOFT_EXIT_MODE} | N={SOFT_EXIT_N} | min_bars={SOFT_EXIT_MIN_BARS} | bias={SOFT_EXIT_PROBA_BIAS}")
    print(f"Shorts: {'ON' if ALLOW_SHORTS else 'OFF'} | TH_LONG={TH_LONG:.2f} | TH_SHORT={TH_SHORT:.2f}")
    print(f"Ensemble: {ENSEMBLE_MODE}")
    print(f"Label: SIMPLE SIGN (CLEAN) - label=exekúcia alignment (TIME_STOP={TSTOP})\n")

    # data — ak WF_END_IDX je zadaný, stiahni sviečky končiace pred daným časom (walk-forward)
    wf_end_ms = _get_int("WF_END_MS", 0) or None  # unix ms — end čas okna pre walk-forward
    df = get_klines(SYMBOL, INTERVAL, lookback=LOOKBACK, start_end_time=wf_end_ms)
    
    # === DUPLICITY CHECK (DRUHÁ AI) ===
    dup_count = df.duplicated("open_time").sum()
    print(f"Duplicates open_time: {dup_count}")
    if dup_count > 0:
        print("⚠️ WARNING: Našli sa duplicitné záznamy! Odstráňujem...")
        df = df.drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    
    dt_diff = df["open_time"].diff().dropna()
    if len(dt_diff) > 0:
        mode_delta = dt_diff.mode()
        if len(mode_delta) > 0:
            print(f"Most common bar delta: {mode_delta.iloc[0]}")
        print(f"Max gap: {dt_diff.max()}\n")
    
    d  = build_features(df)

    # === TREND FILTER SHIFT(1) FIX (DRUHÁ AI) ===
    # KRITICKÝ FIX: Lookahead-free trend filter!
    if USE_TREND_FILTER:
        d["sma_trend"] = d["close"].rolling(SMA_LEN).mean().shift(1)

    # === RET EXTRÉMY CHECK (DRUHÁ AI - HYGIENICKÝ TEST) ===
    print("\n" + "="*60)
    print("RET EXTRÉMY CHECK (hygienický test)")
    print("="*60)
    
    # VŽDY vytvor nový čistý return (explicitne, bez podmienky)
    d["ret_raw"] = d["close"].pct_change()
    
    # DROPNA pred nsmallest/nlargest (DRUHÁ AI oprava!)
    ret_s = d["ret_raw"].dropna()
    
    print("\nRET DISTRIBUTION SUMMARY:")
    print(ret_s.describe())
    
    print("\nTOP 5 CRASHES (najväčšie prepady):")
    if len(ret_s) >= 5:
        crash_idx = ret_s.nsmallest(5).index
        print(d.loc[crash_idx, ["close_time", "close", "ret_raw"]].to_string(index=False))
    else:
        print("Nedostatok dát pre top 5")
    
    print("\nTOP 5 PUMPS (najväčšie skoky):")
    if len(ret_s) >= 5:
        pump_idx = ret_s.nlargest(5).index
        print(d.loc[pump_idx, ["close_time", "close", "ret_raw"]].to_string(index=False))
    else:
        print("Nedostatok dát pre top 5")
    
    print("="*60 + "\n")

    # === KRITICKÝ FIX: H = int(TSTOP) nie natvrdo 6! ===
    H = int(TSTOP)  # ← OPRAVENÉ!
    
    entry  = d["open"].shift(-1)   # open[t+1] (entry point, budúcnosť)
    future = d["close"].shift(-H)  # close[t+H] (cieľ za H barov)
    
    # Explicitná maska pre NaN (žiadne tiché konverzie ako fillna(0))
    mask = entry.isna() | future.isna()
    d["label_simple"] = np.where(
        mask, 
        np.nan,                         # Ponechaj NaN (nie 0!)
        (future > entry).astype(int)    # int vnútri (True/False → 1/0)
    ).astype("float")                    # float vonku (celý result)
    
    # === DEBUG: ret škála kontrola (DRUHÁ AI: pridaj ret_raw porovnanie) ===
    print("\n" + "="*60)
    print("DEBUG: ret škála")
    print("="*60)
    print(f"ret min:  {d['ret'].min():.6f}")
    print(f"ret max:  {d['ret'].max():.6f}")
    print(f"ret mean: {d['ret'].mean():.6f}")
    print(f"ret std:  {d['ret'].std():.6f}")
    print(f"\nret_raw std: {d['ret_raw'].std():.6f}")
    print(f"ret std:     {d['ret'].std():.6f}")
    if abs(d['ret_raw'].std() - d['ret'].std()) > 0.001:
        print("⚠️ WARNING: ret_raw a ret sa líšia! Možný problém v features.py")
    print("\nOčakávané hodnoty (4h BTC):")
    print("  bežné: ±0.01 až ±0.03 (1-3%)")
    print("  extrémy: ±0.10 (10%, zriedka)")
    print("  mean: ~0.0001 (takmer 0)")
    if abs(d['ret'].max()) > 1.0 or abs(d['ret'].min()) > 1.0:
        print("\n⚠️ CRITICAL WARNING: ret je v PERCENTÁCH (nie decimals)!")
        print("   Musíš fixnúť v features.py: ret = ret / 100")
        print("   Baseline bude katastrofálny (-50%+) kvôli zlej škále!")
    elif abs(d['ret'].max()) > 0.15 or abs(d['ret'].min()) > 0.15:
        print("\n⚠️ WARNING: ret hodnoty sú nezvyčajne vysoké!")
        print("   Skontroluj dáta (možno outlier alebo chyba).")
    else:
        print("\n✅ ret škála vyzerá OK!")
    print("="*60 + "\n")
    
    # Label stats
    valid_labels = d["label_simple"].notna().sum()
    up_count = (d["label_simple"] == 1).sum()
    down_count = (d["label_simple"] == 0).sum()
    
    print("Simple label stats:")
    print(f"  Total rows (before dropna): {len(d)}")
    print(f"  Valid labels: {valid_labels}")
    print(f"  Will drop (NaN): {len(d) - valid_labels}")
    if valid_labels > 0:
        print(f"  UP (1): {up_count} ({100*up_count/valid_labels:.1f}%)")
        print(f"  DOWN (0): {down_count} ({100*down_count/valid_labels:.1f}%)")
    
    if len(d) - valid_labels != H:
        print(f"\n⚠️ WARNING: Očakával som Will drop = {H}, ale mám {len(d) - valid_labels}")
        print("   Label maska možno nefunguje správne!")
    else:
        print(f"\n✅ Label NaN maska funguje správne (dropped {H} barov)!")
    print()

    # Rozsireny zoznam featur pre trening/predikciu (poradie fixne)
    FEATS = [
        "ret1","ret5","vol10","rsi14",
        "price_sma20_ratio","zscore20",
        "atr14","atr_pct",
        "sma20","sma200","trend_slope","above_slow",
        "pos_in_range_50","dist_hh_50","dist_ll_50",
        "vol_ma_20","vol_rel_20","vol_zscore_20",
        "bar_body_pct","bar_range_vs_atr"
    ]

    # === label_simple v feats_needed ===
    feats_needed = [
        "ret1","ret5","vol10","rsi14","price_sma20_ratio","zscore20",
        "atr14","ret","label_simple",
        "atr_pct","sma20","sma200","trend_slope","above_slow",
        "pos_in_range_50","dist_hh_50","dist_ll_50",
        "vol_ma_20","vol_rel_20","vol_zscore_20",
        "bar_body_pct","bar_range_vs_atr"
    ]
    if USE_TREND_FILTER:
        feats_needed.append("sma_trend")

    d = d.dropna(subset=feats_needed).reset_index(drop=True)

    n = len(d)
    min_needed = 50
    if n < VAL + FORWARD + min_needed:
        print("WARNING: Malo dat pre zvolene segmenty - rozdelenie upravene.")
    train_end = max(min_needed, n - (VAL + FORWARD))
    val_end   = max(train_end, n - FORWARD)

    train = d.iloc[:train_end]
    val   = d.iloc[train_end:val_end]
    fwd   = d.iloc[val_end:]

    # === STARÁ AI DEBUG: Buy&Hold context ===
    bh = (fwd["close"].iloc[-1] / fwd["close"].iloc[0] - 1) * 100
    print(f"[FWD] Buy&Hold close->close: {bh:.2f}%")
    print(f"Rozsah: TRAIN={len(train)} | VAL={len(val)} | FORWARD={len(fwd)} (spolu {n})\n")

    Xtr = train[FEATS]
    ytr = train["label_simple"].astype(int)

    # fit LR a RF
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))]).fit(Xtr, ytr)
    pipe_rf = Pipeline([("rf", RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=10,
                                                      random_state=42, n_jobs=-1))]).fit(Xtr, ytr)
    val_df = val.copy()
    fwd_df = fwd.copy()

    # ---------- vyhodnocovaci blok pre LR/RF ----------

    def eval_model_block(name, pipe, val_df, fwd_df, THS_long, THS_short):
        best = None
        thL_for_base = THS_long[0] if len(THS_long) else TH_LONG
        thS_for_base = THS_short[0] if len(THS_short) else TH_SHORT
        seg_fwd_base, pos_fwd_base, plain_tot_fwd, plain_dd_fwd = evaluate_segment_dual(
            pipe, fwd_df, FEATS, thL_for_base, thS_for_base, FEE, ALLOW_SHORTS, USE_TREND_FILTER
        )
        
        # === STARÁ AI DEBUG: Time-in-market + Baseline EXEC (aligned) ===
        in_mkt_pct = 100.0 * (seg_fwd_base["pos"] != 0).mean()
        print(f"[{name} BASELINE] time-in-market: {in_mkt_pct:.1f}%")
        
        _, tot_exec, dd_exec, n_exec = run_backtest(
            seg_fwd_base[["open","high","low","close","atr14"]],
            seg_fwd_base["pos"],
            seg_fwd_base["proba_up"],
            thL_for_base, thS_for_base,
            equity_usdt=EQUITY_USDT, risk_pct=RISK_PCT, fee=FEE,
            atr_mult_sl=1.5, tp_R=None, partial_pct=0.0, trail_k=999.0,
            time_stop_bars=TSTOP,
            soft_mode="OFF"
        )
        print(f"[{name} BASELINE EXEC] vynos {tot_exec:.2f}% | MaxDD {dd_exec:.2f}% | obchody {n_exec}")
        
        # DEBUG: FWD baseline proba distribution
        debug_proba(name, "FWD baseline", seg_fwd_base["proba_up"].values)
        
        print(f"\n====================  MODEL {name}  ====================")
        print(f"\nFORWARD baseline (plain ret, NIE aligned): vynos {plain_tot_fwd:.2f}% | MaxDD {plain_dd_fwd:.2f}% | TH_L={thL_for_base:.2f}")

        if len(val_df) > 0:
            for thL in THS_long:
                for thS in THS_short:
                    val_seg, pos_val, _, _ = evaluate_segment_dual(
                        pipe, val_df, FEATS, thL, thS, FEE, ALLOW_SHORTS, USE_TREND_FILTER
                    )
                    
                    # DEBUG: VAL proba distribution (len pri prvom threshold páre)
                    if thL == THS_long[0] and thS == THS_short[0]:
                        debug_proba(name, f"VAL (thL={thL:.2f}, thS={thS:.2f})", val_seg["proba_up"].values)
                    
                    for k_sl in ATRS:
                        for tpR in TPRS:
                            for trK in TRAILS:
                                for part in PARTS:
                                    _, tot, dd, n_tr = run_backtest(
                                        val_seg[["open","high","low","close","atr14"]],
                                        val_seg["pos"],
                                        val_seg["proba_up"],
                                        thL, thS,
                                        equity_usdt=EQUITY_USDT, risk_pct=RISK_PCT,
                                        fee=FEE, atr_mult_sl=k_sl, tp_R=tpR,
                                        partial_pct=part, trail_k=trK, time_stop_bars=TSTOP,
                                        soft_mode=SOFT_EXIT_MODE, soft_n=SOFT_EXIT_N,
                                        soft_min_bars=SOFT_EXIT_MIN_BARS, soft_bias=SOFT_EXIT_PROBA_BIAS
                                    )
                                    # filtre na VAL
                                    if n_tr < MIN_TRADES_VAL:
                                        continue
                                    if dd < -abs(VAL_MAX_DD):
                                        continue

                                    # score s penalizaciou DD a vahou za vzorku
                                    score_raw = tot - SCORE_ALPHA * abs(dd)
                                    score = score_raw * math.sqrt(max(n_tr, 1))

                                    if (best is None) or (score > best[0]):
                                        best = (score, thL, thS, k_sl, tpR, trK, part, tot, dd, n_tr)
        if best:
            _, thL_b, thS_b, k_sl, tpR, trK, part, tot_v, dd_v, n_v = best
            fwd_seg2, pos_fwd2, _, _ = evaluate_segment_dual(
                pipe, fwd_df, FEATS, thL_b, thS_b, FEE, ALLOW_SHORTS, USE_TREND_FILTER
            )

            # >>> OBSERVE mód: nalepenie social_count_total na FWD segment a ulozenie logu <<<
            try:
                fwd_seg2_obs = attach_social_to_segment(fwd_seg2, SYMBOL)

                observe_cols = ["close_time", "open", "high", "low", "close",
                                "proba_up", "social_count_total"]
                observe_df = fwd_seg2_obs.copy()
                observe_df["pos"] = pos_fwd2.values

                use_cols = [c for c in observe_cols + ["pos"] if c in observe_df.columns]
                observe_df = observe_df[use_cols]

                _data_dir = os.getenv("DATA_DIR")
                if not _data_dir or not os.path.exists(_data_dir):
                    raise RuntimeError(f"DATA_DIR chýba alebo SSD nie je pripojené: {_data_dir}")
                os.makedirs(os.path.join(_data_dir, "logs"), exist_ok=True)
                observe_path = os.path.join(_data_dir, "logs", f"observe_{SYMBOL}_{INTERVAL}_{name}.csv")
                observe_df.to_csv(observe_path, index=False)
                print(f"  [OBSERVE] Social log ulozeny do: {observe_path}")
            except Exception as e:
                print(f"  [OBSERVE] Chyba pri logovani social dat: {e}")

            _, tot_f, dd_f, n_f = run_backtest(
                fwd_seg2[["open","high","low","close","atr14"]],
                pos_fwd2,
                fwd_seg2["proba_up"],
                thL_b, thS_b,
                equity_usdt=EQUITY_USDT, risk_pct=RISK_PCT, fee=FEE,
                atr_mult_sl=k_sl, tp_R=tpR, partial_pct=part, trail_k=trK, time_stop_bars=TSTOP,
                soft_mode=SOFT_EXIT_MODE, soft_n=SOFT_EXIT_N,
                soft_min_bars=SOFT_EXIT_MIN_BARS, soft_bias=SOFT_EXIT_PROBA_BIAS
            )
            tp_txt = f"{tpR:.1f}R" if tpR is not None else "None"
            print(f"  BEST na VAL: TH_L={thL_b:.2f} | TH_S={thS_b:.2f} | SL={k_sl}xATR | TP={tp_txt} | partial={part*100:.0f}%@1R | trail={trK}xATR | time-stop={TSTOP}")
            print(f"    -> VAL: vynos {tot_v:.2f}% | MaxDD {dd_v:.2f}% | obchody {n_v}")
            print(f"    -> FWD: vynos {tot_f:.2f}% | MaxDD {dd_f:.2f}% | obchody {n_f}")
        else:
            print("Ziadne best parametre (VAL prazdna alebo po filtroch nic nepreslo).")

    # LR
    THS_lr_long  = THS_LONG.copy()
    THS_lr_short = THS_SHORT.copy()
    if TH_LONG  is not None and TH_LONG  not in THS_lr_long:  THS_lr_long  = [TH_LONG]  + THS_lr_long
    if TH_SHORT is not None and TH_SHORT not in THS_lr_short: THS_lr_short = [TH_SHORT] + THS_lr_short
    eval_model_block("LR", pipe_lr, val_df, fwd_df, THS_lr_long, THS_lr_short)

    # RF
    THS_rf_long  = THS_LONG.copy()
    THS_rf_short = THS_SHORT.copy()
    if TH_LONG  is not None and TH_LONG  not in THS_rf_long:  THS_rf_long  = [TH_LONG]  + THS_rf_long
    if TH_SHORT is not None and TH_SHORT not in THS_rf_short: THS_rf_short = [TH_SHORT] + THS_rf_short
    eval_model_block("RF", pipe_rf, val_df, fwd_df, THS_rf_long, THS_rf_short)

    # Ensemble (OR/AND)
    if ENSEMBLE_MODE in ("OR", "AND"):
        print(f"\n====================  MODEL ENSEMBLE ({ENSEMBLE_MODE})  ====================")
        proba_lr_val = pipe_lr.predict_proba(val_df[FEATS])[:,1] if len(val_df) else np.array([])
        proba_rf_val = pipe_rf.predict_proba(val_df[FEATS])[:,1] if len(val_df) else np.array([])
        proba_lr_fwd = pipe_lr.predict_proba(fwd_df[FEATS])[:,1] if len(fwd_df) else np.array([])
        proba_rf_fwd = pipe_rf.predict_proba(fwd_df[FEATS])[:,1] if len(fwd_df) else np.array([])

        thL = THS_LONG[0]
        thS = TH_SHORT

        pos_fwd = make_ensemble_pos(ENSEMBLE_MODE, proba_lr_fwd, thL, thS,
                                    proba_rf_fwd, thL, thS, ALLOW_SHORTS)
        fwd_base = fwd_df.copy()
        fwd_base["pos"] = pos_fwd
        if USE_TREND_FILTER and "sma_trend" in fwd_base.columns:
            mask_long  = (fwd_base["pos"]==1) & (fwd_base["close"]>fwd_base["sma_trend"])
            mask_short = (fwd_base["pos"]==-1) & (fwd_base["close"]<fwd_base["sma_trend"])
            fwd_base["pos"] = 0
            fwd_base.loc[mask_long, "pos"] = 1
            fwd_base.loc[mask_short,"pos"] = -1

        best = None
        if len(val_df) > 0:
            pos_val = make_ensemble_pos(ENSEMBLE_MODE, proba_lr_val, thL, thS,
                                        proba_rf_val, thL, thS, ALLOW_SHORTS)
            val_base = val_df.copy()
            val_base["pos"] = pos_val
            if USE_TREND_FILTER and "sma_trend" in val_base.columns:
                mask_long  = (val_base["pos"]==1) & (val_base["close"]>val_base["sma_trend"])
                mask_short = (val_base["pos"]==-1) & (val_base["close"]<val_base["sma_trend"])
                val_base["pos"] = 0
                val_base.loc[mask_long,"pos"] = 1
                val_base.loc[mask_short,"pos"] = -1

            for k_sl in ATRS:
                for tpR in TPRS:
                    for trK in TRAILS:
                        for part in PARTS:
                            _, tot, dd, n_tr = run_backtest(
                                val_base[["open","high","low","close","atr14"]],
                                val_base["pos"],
                                pd.Series((proba_lr_val+proba_rf_val)/2, index=val_base.index),
                                thL, thS,
                                equity_usdt=EQUITY_USDT, risk_pct=RISK_PCT, fee=FEE,
                                atr_mult_sl=k_sl, tp_R=tpR, partial_pct=part, trail_k=trK, time_stop_bars=TSTOP,
                                soft_mode=SOFT_EXIT_MODE, soft_n=SOFT_EXIT_N,
                                soft_min_bars=SOFT_EXIT_MIN_BARS, soft_bias=SOFT_EXIT_PROBA_BIAS
                            )
                            if n_tr < MIN_TRADES_VAL:
                                continue
                            if dd < -abs(VAL_MAX_DD):
                                continue
                            score_raw = tot - SCORE_ALPHA * abs(dd)
                            score = score_raw * math.sqrt(max(n_tr, 1))
                            if (best is None) or (score > best[0]):
                                best = (score, k_sl, tpR, trK, part, tot, dd, n_tr)

        if best:
            _, k_sl, tpR, trK, part, tot_v, dd_v, n_v = best
            fwd_use = fwd_base.copy()
            _, tot_f, dd_f, n_f = run_backtest(
                fwd_use[["open","high","low","close","atr14"]],
                fwd_use["pos"],
                pd.Series((proba_lr_fwd+proba_rf_fwd)/2, index=fwd_use.index),
                thL, thS,
                equity_usdt=EQUITY_USDT, risk_pct=RISK_PCT, fee=FEE,
                atr_mult_sl=k_sl, tp_R=tpR, partial_pct=part, trail_k=trK, time_stop_bars=TSTOP,
                soft_mode=SOFT_EXIT_MODE, soft_n=SOFT_EXIT_N,
                soft_min_bars=SOFT_EXIT_MIN_BARS, soft_bias=SOFT_EXIT_PROBA_BIAS
            )
            tp_txt = f"{tpR:.1f}R" if tpR is not None else "None"
            print(f"  BEST na VAL: SL={k_sl}xATR | TP={tp_txt} | partial={part*100:.0f}%@1R | trail={trK}xATR | time-stop={TSTOP}")
            print(f"    -> VAL: vynos {tot_v:.2f}% | MaxDD {dd_v:.2f}% | obchody {n_v}")
            print(f"    -> FWD: vynos {tot_f:.2f}% | MaxDD {dd_f:.2f}% | obchody {n_f}")
        else:
            print("Ziadne best parametre (VAL prazdna alebo po filtroch nic nepreslo).")

if __name__ == "__main__":
    main()
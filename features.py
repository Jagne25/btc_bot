# C:\btc_bot\features.py
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df s open, high, low, close, volume, close_time (zoradené od najstaršieho).
    Output: df + featury bez lookahead (kontext do close[t], vstup na open[t+1]).
    """
    d = df.copy()

    # 0) Minule hodnoty
    close_prev = d["close"].shift(1)
    open_prev = d["open"].shift(1)
    high_prev = d["high"].shift(1)
    low_prev  = d["low"].shift(1)

    # 1) Moment a vola
    d["ret"]   = d["close"].pct_change()
    d["ret1"]  = d["ret"].shift(1)
    d["ret5"]  = d["close"].pct_change(5).shift(1)
    d["vol10"] = d["ret"].rolling(10).std().shift(1)

    # RSI(14) z close[t]
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    d["rsi14"] = 100 - 100 / (1 + rs)

    # 2) MA/STD z minulosti (shift(1))
    prev_ma20  = d["close"].rolling(20).mean().shift(1)
    prev_std20 = d["close"].rolling(20).std().shift(1)
    d["sma20"] = prev_ma20
    d["price_sma20_ratio"] = (d["close"] / prev_ma20) - 1
    d["zscore20"] = (d["close"] - prev_ma20) / prev_std20.replace(0, np.nan)

    # 3) ATR(14) a derivaty
    hl = (d["high"] - d["low"]).abs()
    hc = (d["high"] - d["close"].shift()).abs()
    lc = (d["low"]  - d["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr14"].shift(1) / close_prev

    # 4) Trend kontext (t-1)
    sma200 = d["close"].rolling(200).mean()
    sma200_prev = sma200.shift(1)
    d["sma200"] = sma200_prev
    d["trend_slope"] = (prev_ma20 - sma200_prev) / close_prev
    d["above_slow"] = (close_prev > sma200_prev).astype("Int8")

    # 5) Range poloha (50 barov), vsetko z t-1
    d["hh_50"] = d["high"].rolling(50).max()
    d["ll_50"] = d["low"].rolling(50).min()
    hh_prev = d["hh_50"].shift(1)
    ll_prev = d["ll_50"].shift(1)
    range_prev = (hh_prev - ll_prev).replace(0, np.nan)
    d["pos_in_range_50"] = (close_prev - ll_prev) / range_prev

    # 6) Volume aktivita
    d["vol_ma_20"] = d["volume"].rolling(20).mean()
    d["vol_rel_20"] = d["volume"] / d["vol_ma_20"].replace(0, np.nan)
    d["vol_zscore_20"] = (d["volume"] - d["vol_ma_20"]) / d["volume"].rolling(20).std().replace(0, np.nan)


    # 7) Vzdialenosti k hranam range (prakticke do SQL)
    d["dist_hh_50"] = (hh_prev - close_prev) / close_prev
    d["dist_ll_50"] = (close_prev - ll_prev) / close_prev

    # 8) Tvar sviecky a velkost voci ATR
    bar_range = (d["high"] - d["low"])
    body = (d["close"] - d["open"]).abs()
    d["bar_body_pct"] = body / bar_range.replace(0, np.nan)
    d["bar_range_vs_atr"] = bar_range / d["atr14"].replace(0, np.nan)

    # 9) MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    d["macd"] = macd_line.shift(1)
    d["macd_signal"] = macd_signal.shift(1)
    d["macd_hist"] = (macd_line - macd_signal).shift(1)

    # 10) ADX(14) — sila trendu (nie smer)
    # +DM = o koľko je dnešné high vyššie ako včerajšie
    # -DM = o koľko je dnešné low nižšie ako včerajšie
    high_diff = d["high"].diff()
    low_diff  = -d["low"].diff()
    plus_dm   = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm  = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr14_adx = tr.ewm(span=14, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14_adx.replace(0, np.nan)
    minus_di  = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14_adx.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx       = dx.ewm(span=14, adjust=False).mean()
    d["adx14"]    = adx.shift(1)      # sila trendu: >25 = trend, <20 = choppy
    d["plus_di"]  = plus_di.shift(1)  # bullish sila
    d["minus_di"] = minus_di.shift(1) # bearish sila

    return d
"""
Regime Analysis — analýza správania BTC v rôznych trhových režimoch
Otázka: Kde má trh prirodzený edge? Trend / Chop / Neutral?
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1. DÁTA ──────────────────────────────────────────────────────────────────

def fetch_data(symbol="BTCUSDT", interval="4h", limit=10950):
    url = "https://api.binance.com/api/v3/klines"
    rows, end_time = [], None
    while len(rows) < limit:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(url, params=params)
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        end_time = chunk[0][0] - 1
        if len(rows) >= limit:
            break
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbav","tbqv","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").drop_duplicates(subset="open_time").reset_index(drop=True)
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df[["open_time","close","high","low"]].copy()

# ── 2. INDIKÁTORY ─────────────────────────────────────────────────────────────

def add_indicators(df):
    # ATR — potrebujeme pre ADX
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # ADX(14)
    high_diff = df["high"].diff()
    low_diff  = -df["low"].diff()
    plus_dm   = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm  = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr_adx   = true_range.ewm(span=14, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_adx
    minus_di  = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_adx
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx14"] = dx.ewm(span=14, adjust=False).mean()

    return df

# ── 3. OZNAČENIE REŽIMOV ──────────────────────────────────────────────────────

def add_regime(df):
    # Každá sviečka dostane označenie podľa ADX
    # trend   = ADX > 25 → silný smer
    # chop    = ADX < 20 → trh nikam nejde
    # neutral = ADX 20-25 → prechodná zóna
    conditions = [
        df["adx14"] > 25,
        df["adx14"] < 20,
    ]
    choices = ["trend", "chop"]
    df["regime"] = np.select(conditions, choices, default="neutral")
    return df

# ── 4. VÝNOS PO N SVIEČKACH ───────────────────────────────────────────────────

def add_forward_return(df, horizon=20):
    # forward_return = o koľko percent sa cena zmení za nasledujúcich N sviečok
    # napr. horizon=20 → pozrieme sa čo sa stalo po 20 sviečkach (= 80 hodín)
    df[f"fwd_{horizon}"] = df["close"].shift(-horizon) / df["close"] - 1
    df[f"fwd_{horizon}"] *= 100  # prevedieme na percentá
    return df

# ── 5. ANALÝZA ────────────────────────────────────────────────────────────────

def analyze_regimes(df, horizon=20):
    col = f"fwd_{horizon}"

    # Vyhodíme sviečky kde nemáme budúci výnos (posledných N sviečok)
    data = df.dropna(subset=[col, "adx14"])

    print(f"\n{'='*55}")
    print(f"  ANALÝZA REŽIMOV — výnos po {horizon} sviečkach ({horizon*4}h)")
    print(f"{'='*55}")
    print(f"  Celkový počet sviečok: {len(data)}")
    print()

    for regime in ["trend", "chop", "neutral"]:
        subset = data[data["regime"] == regime]
        if len(subset) == 0:
            continue

        wins    = subset[subset[col] > 0]
        losses  = subset[subset[col] <= 0]
        winrate = len(wins) / len(subset) * 100
        avg_ret = subset[col].mean()
        avg_win = wins[col].mean() if len(wins) > 0 else 0
        avg_loss = losses[col].mean() if len(losses) > 0 else 0

        print(f"  Režim: {regime.upper()}")
        print(f"  Počet sviečok:  {len(subset)}")
        print(f"  Winrate:        {winrate:.1f}%")
        print(f"  Avg výnos:      {avg_ret:+.2f}%")
        print(f"  Avg win:        {avg_win:+.2f}%")
        print(f"  Avg loss:       {avg_loss:+.2f}%")
        print(f"  {'─'*40}")

# ── 6. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Sťahujem dáta BTC 4h...")
    df = fetch_data()
    df = add_indicators(df)
    df = add_regime(df)
    df = add_forward_return(df, horizon=20)
    df = add_forward_return(df, horizon=40)
    df = df.dropna(subset=["adx14"]).reset_index(drop=True)

    print(f"Dáta: {len(df)} sviečok | od {df['open_time'].iloc[0].date()} do {df['open_time'].iloc[-1].date()}")

    # Rozdelenie sviečok podľa režimu
    print(f"\n  Rozdelenie: trend={len(df[df['regime']=='trend'])} | "
          f"chop={len(df[df['regime']=='chop'])} | "
          f"neutral={len(df[df['regime']=='neutral'])}")

    # Analýza pre oba horizonty
    analyze_regimes(df, horizon=20)
    analyze_regimes(df, horizon=40)

"""
Trend + Pullback Backtest
Logika: ide s trhom, nie proti nemu

MARKET STATE:  close > MA200 AND MA200 rastie
SETUP:         cena sa stiahla k MA20/MA50 (pullback)
ENTRY:         bullish sviečka — close > high predchádzajúcej sviečky
EXIT:          trailing stop ALEBO close < MA50
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1. DÁTA ──────────────────────────────────────────────────────────────────

def fetch_data(symbol="BTCUSDT", interval="4h", limit=10950):
    url = "https://api.binance.com/api/v3/klines"
    rows, end_time = [], None

    # Sťahujeme v slučke — každý request max 1000 sviečok
    # posúvame endTime dozadu kým nemáme dosť
    while len(rows) < limit:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time          # "daj mi sviečky pred týmto časom"
        r = requests.get(url, params=params)
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        oldest_open = chunk[0][0]                 # čas najstaršej sviečky v batchi
        end_time = oldest_open - 1                # posuň okno dozadu o 1ms
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
    return df[["open_time","open","high","low","close"]].copy()

# ── 2. INDIKÁTORY ─────────────────────────────────────────────────────────────

def add_indicators(df):

    # MA20 — krátkodobý priemer (20 sviečok)
    df["ma20"] = df["close"].rolling(20).mean()

    # MA50 — strednodobý priemer (50 sviečok)
    df["ma50"] = df["close"].rolling(50).mean()

    # MA200 — dlhodobý priemer (200 sviečok) = hlavný trend filter
    df["ma200"] = df["close"].rolling(200).mean()

    # MA200 slope — o koľko percent sa MA200 zmenil za posledných 5 sviečok
    # kladný slope = MA200 rastie = trend žije
    # záporný slope = MA200 klesá = trend slabne
    df["ma200_slope"] = (df["ma200"] - df["ma200"].shift(5)) / df["ma200"] * 100

    # ATR — Average True Range = priemerná veľkosť pohybu za 14 sviečok
    # používame na meranie či pullback nie je príliš extrémny
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14).mean()

    # atr_pct — ATR ako percento ceny (normalizované)
    df["atr_pct"] = df["atr"] / df["close"] * 100

    # dist_ma20 — vzdialenosť ceny od MA20 v % (záporné = pod MA20)
    df["dist_ma20"] = (df["close"] - df["ma20"]) / df["ma20"] * 100

    # dist_ma50 — vzdialenosť ceny od MA50 v %
    df["dist_ma50"] = (df["close"] - df["ma50"]) / df["ma50"] * 100

    # ADX(14) — sila trendu
    # Krok 1: +DM a -DM — bullish a bearish sila každej sviečky
    high_diff = df["high"].diff()          # o koľko je high vyššie ako včera
    low_diff  = -df["low"].diff()          # o koľko je low nižšie ako včera
    plus_dm   = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm  = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    # Krok 2: normalizujeme voči ATR → plus_di a minus_di v %
    atr_adx  = true_range.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_adx
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_adx

    # Krok 3: DX = rozdiel / súčet → ADX = priemer DX za 14 sviečok
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx14"] = dx.ewm(span=14, adjust=False).mean()

    return df

# ── 3. BACKTEST ───────────────────────────────────────────────────────────────

def run_backtest(df,
                 pullback_ma="ma20",      # k akej MA čakáme pullback (ma20 alebo ma50)
                 pullback_dist=-1.0,      # cena musí byť aspoň X% pod MA (napr. -1%)
                 max_pullback=-8.0,       # ale nie viac ako X% (príliš extrémny = skip)
                 trail_atr=2.0,           # trailing stop = entry - N x ATR
                 use_ma50_exit=True,      # False = len trailing stop, žiadny MA50 break
                 max_duration=None):      # max počet sviečok v obchode (None = vypnutý)

    trades = []
    in_trade = False

    for i in range(2, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i-1]   # predchádzajúca sviečka

        if not in_trade:

            # ── MARKET STATE ──
            # 1. cena musí byť nad MA200 → sme v dlhodobom trende hore
            trend_up = row["close"] > row["ma200"]
            adx_ok = row["adx14"] > 25
            # 2. MA200 musí rásť → trend žije (nie sideways nad MA200)
            ma200_rising = row["ma200_slope"] > 0

            # ── SETUP ──
            # 3. cena sa stiahla k MA (pullback) — je pod MA o X%
            dist = row[f"dist_{pullback_ma}"]
            pullback_ok = (dist < pullback_dist) and (dist > max_pullback)
            # dist < -1%  → sme aspoň 1% pod MA → pullback nastal
            # dist > -8%  → nie sme príliš hlboko → nie je to crash

            # 4. ATR filter — pohyb nie je extrémny (trh nie je v panike)
            # ak ATR je viac ako 5% → príliš volatilné → skip
            volatility_ok = row["atr_pct"] < 5.0

            # ── ENTRY ──
            # 5. bullish reversal sviečka:
            # aktuálna sviečka zavrie VYŠŠIE ako high predchádzajúcej sviečky
            # to znamená: trh sa odrazil a prerazil posledný odpor
            breakout = row["close"] > prev["high"]

            # Vstúpime len keď VŠETKY podmienky zelené
            if trend_up and ma200_rising and pullback_ok and volatility_ok and breakout and adx_ok:
                in_trade = True
                entry_price = row["close"]
                entry_time  = row["open_time"]
                entry_i     = i
                # trailing stop = vstupná cena - 2x ATR
                # ak cena klesne sem → exit (chránime kapitál)
                trail_stop  = entry_price - trail_atr * row["atr"]
                max_adverse = 0.0

        else:

            # Sledujeme najhorší pohyb od vstupu
            adverse = (row["close"] - entry_price) / entry_price * 100
            if adverse < max_adverse:
                max_adverse = adverse

            duration = i - entry_i

            # Posúvame trailing stop nahor keď cena rastie
            # stop nikdy nejde dole — len hore (chránime zisk)
            new_stop = row["close"] - trail_atr * row["atr"]
            if new_stop > trail_stop:
                trail_stop = new_stop

            # ── EXIT 1: trailing stop zasiahnutý ──
            # cena klesla pod náš pohyblivý stop
            if row["close"] < trail_stop:
                pnl = (row["close"] - entry_price) / entry_price * 100
                trades.append({
                    "entry": entry_time, "exit": row["open_time"],
                    "entry_price": entry_price, "exit_price": row["close"],
                    "pnl": pnl, "type": "trail_stop",
                    "duration": duration, "max_adverse": max_adverse,
                })
                in_trade = False

            # ── EXIT 2: time stop — príliš dlho v obchode bez zisku ──
            elif max_duration is not None and duration >= max_duration:
                pnl = (row["close"] - entry_price) / entry_price * 100
                trades.append({
                    "entry": entry_time, "exit": row["open_time"],
                    "entry_price": entry_price, "exit_price": row["close"],
                    "pnl": pnl, "type": "time_stop",
                    "duration": duration, "max_adverse": max_adverse,
                })
                in_trade = False

            # ── EXIT 3: dve sviečky za sebou pod MA50 → trend sa zlomil ──
            # jedna sviečka = možno len test MA50 (normálny pohyb)
            # dve sviečky = trh naozaj otočil → exit
            # use_ma50_exit=False → tento exit vypnutý, len trailing stop
            elif use_ma50_exit and row["close"] < row["ma50"] and prev["close"] < prev["ma50"]:
                pnl = (row["close"] - entry_price) / entry_price * 100
                trades.append({
                    "entry": entry_time, "exit": row["open_time"],
                    "entry_price": entry_price, "exit_price": row["close"],
                    "pnl": pnl, "type": "ma50_break",
                    "duration": duration, "max_adverse": max_adverse,
                })
                in_trade = False

    return pd.DataFrame(trades)

# ── 4. ŠTATISTIKA ─────────────────────────────────────────────────────────────

def print_stats(trades, label=""):
    print(f"── {label} ──")
    if len(trades) == 0:
        print("  Žiadne obchody.\n")
        return

    wins   = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    winrate  = len(wins) / len(trades) * 100
    avg_win  = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    ev       = (len(wins)/len(trades)) * avg_win + (len(losses)/len(trades)) * avg_loss

    equity      = trades["pnl"].cumsum()
    rolling_max = equity.cummax()
    max_dd      = (equity - rolling_max).min()

    # Profit factor = súčet ziskov / súčet strát (> 1 = systém zarába)
    gross_win  = wins["pnl"].sum()   if len(wins)   > 0 else 0
    gross_loss = losses["pnl"].abs().sum() if len(losses) > 0 else 1
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    # Typy exitov
    trail = trades[trades["type"] == "trail_stop"]
    ma_break = trades[trades["type"] == "ma50_break"]

    print("=" * 50)
    print(f"  Počet obchodov:     {len(trades)}")
    print(f"  Winrate:            {winrate:.1f}%")
    print(f"  Avg win:            +{avg_win:.2f}%")
    print(f"  Avg loss:           {avg_loss:.2f}%")
    print(f"  Expectancy (EV):    {ev:.2f}% na obchod")
    print(f"  Profit factor:      {pf:.2f}  (>1 = dobré)")
    print(f"  Kumulatívny výnos:  {equity.iloc[-1]:.1f}%")
    print(f"  Max Drawdown:       {max_dd:.2f}%")
    print(f"  Trail stop exity:   {len(trail)}/{len(trades)}")
    print(f"  MA50 break exity:   {len(ma_break)}/{len(trades)}")
    print("=" * 50)

    if ev > 0 and len(trades) >= 20:
        print("  → Pozitívny EV. Edge existuje.")
    elif len(trades) < 20:
        print("  → Príliš málo obchodov — nedá sa vyvodiť záver.")
    else:
        print("  → Negatívny EV.")
    print()

    # Detail losing trades
    if len(losses) > 0:
        print("  LOSING TRADES:")
        print(f"  {'dátum':<20} {'pnl':>6} {'trvanie':>8} {'max_adv':>9} {'exit typ':<12}")
        print("  " + "-" * 60)
        for _, t in losses.iterrows():
            print(f"  {str(t['entry']):<20} "
                  f"{t['pnl']:>+6.2f}% "
                  f"{int(t['duration']):>7}sv "
                  f"{t['max_adverse']:>+8.2f}% "
                  f"  {t['type']}")
        print()

# ── 5. GRAF ───────────────────────────────────────────────────────────────────

def plot_trades(df, trades, title="Trend + Pullback — BTC 4h"):
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(df["open_time"], df["close"],  color="steelblue", linewidth=1,   label="BTC cena")
    ax.plot(df["open_time"], df["ma20"],   color="orange",    linewidth=1,   label="MA20")
    ax.plot(df["open_time"], df["ma50"],   color="purple",    linewidth=1.2, label="MA50")
    ax.plot(df["open_time"], df["ma200"],  color="red",       linewidth=1.5, label="MA200")

    if len(trades) > 0:
        for _, t in trades.iterrows():
            entry_row = df[df["open_time"] == t["entry"]]
            exit_row  = df[df["open_time"] == t["exit"]]
            if len(entry_row) == 0 or len(exit_row) == 0:
                continue
            color  = "green" if t["pnl"] > 0 else "red"
            ax.scatter(entry_row["open_time"], entry_row["close"],
                       color="lime", zorder=5, s=80, marker="^")
            ax.scatter(exit_row["open_time"], exit_row["close"],
                       color=color, zorder=5, s=80, marker="v")
            ax.plot([entry_row["open_time"].values[0], exit_row["open_time"].values[0]],
                    [entry_row["close"].values[0], exit_row["close"].values[0]],
                    color=color, alpha=0.3, linewidth=1)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Cena (USD)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("trend_pullback_chart.png", dpi=150)
    print("Graf uložený: trend_pullback_chart.png")
    plt.show()

# ── 6. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    all_trades = []

    for symbol in SYMBOLS:
        print(f"\nSťahujem dáta {symbol} 4h...")
        df = fetch_data(symbol=symbol, interval="4h", limit=10950)
        df = add_indicators(df)
        df = df.dropna().reset_index(drop=True)
        print(f"Dáta: {len(df)} sviečok | od {df['open_time'].iloc[0].date()} do {df['open_time'].iloc[-1].date()}")

        # Experiment 4 — najlepšie parametre z BTC testu
        trades = run_backtest(df, pullback_ma="ma20", pullback_dist=-1.0,
                              max_pullback=-8.0, trail_atr=3.0,
                              use_ma50_exit=False)
        trades["symbol"] = symbol
        print_stats(trades, f"{symbol} | trail=3xATR, bez MA50 exit, ADX>25")
        all_trades.append(trades)

    # Kombinované štatistiky — všetky coiny spolu
    combined = pd.concat(all_trades, ignore_index=True)
    print_stats(combined, "COMBINED — BTC + ETH + SOL + BNB")
    print(f"Celkový počet obchodov pre B1 label: {len(combined)}")

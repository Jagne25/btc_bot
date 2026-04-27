"""
Mean Reversion Backtest — jednoduchý experiment
Hypotéza: keď z-score < -2, cena sa vráti k MA20

Žiadny ML. Žiadne indikátory navyše.
Len: cena, MA20, štandardná odchýlka, z-score.
"""

# requests = knižnica na sťahovanie dát z internetu (ako prehliadač, len v kóde)
# pandas = knižnica na prácu s tabuľkami (ako Excel v Pythone)
# numpy = knižnica na matematiku (priemery, odmocniny, atď.)
import requests
import pandas as pd
import numpy as np
# matplotlib = knižnica na kreslenie grafov
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. DÁTA ──────────────────────────────────────────────────────────────────

# Definujeme funkciu ktorá stiahne sviečky z Binance
# symbol = ktorý coin (BTCUSDT)
# interval = veľkosť sviečky (4h = každé 4 hodiny jedna sviečka)
# limit = koľko sviečok chceme (1500 = asi 250 dní dozadu)
def fetch_data(symbol="BTCUSDT", interval="4h", limit=1500):

    # Adresa Binance API — odtiaľ berieme dáta, bez API kľúča
    url = "https://api.binance.com/api/v3/klines"

    # Parametre — hovoríme Binance: daj mi BTC, 4h sviečky, 1500 kusov
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    # Pošleme požiadavku na Binance a uložíme odpoveď
    r = requests.get(url, params=params)

    # Binance nám vráti zoznam čísel — prevedieme ho na tabuľku
    # columns = názvy stĺpcov (Binance posiela vždy v tomto poradí)
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbav","tbqv","ignore"
    ])

    # close = záverečná cena sviečky — prevedieme na číslo (Binance posiela ako text)
    df["close"] = df["close"].astype(float)

    # open_time = čas otvorenia sviečky — prevedieme z milisekúnd na normálny dátum
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    # Vrátime len stĺpce ktoré potrebujeme: čas a záverečná cena
    return df[["open_time","close"]].copy()

# ── 2. INDIKÁTORY ─────────────────────────────────────────────────────────────

# Funkcia ktorá vypočíta z-score + trend filtre
# window=20 = počítame priemer z posledných 20 sviečok
def add_zscore(df, window=20):

    # ma = moving average = priemer ceny za posledných 20 sviečok
    df["ma"] = df["close"].rolling(window).mean()

    # std = štandardná odchýlka = o koľko sa cena bežne odchyľuje od priemeru
    df["std"] = df["close"].rolling(window).std()

    # z-score = (aktuálna cena - priemer) / odchýlka
    # z = -2 → som 2x ďalej dole ako je normálne → možno prepredané
    df["zscore"] = (df["close"] - df["ma"]) / df["std"]

    # ── FILTER 1: Sklon MA20 ──
    # slope = rozdiel MA20 teraz vs MA20 pred 5 sviečkami
    # malý slope → MA je rovná → RANGE → obchoduj
    # veľký slope → MA ide prudko → TREND → preskočí
    df["slope"] = df["ma"] - df["ma"].shift(5)

    # Normalizujeme slope voči cene (aby sme porovnávali %, nie $)
    # napr. slope 500$ pri BTC 50000$ = 1% → malé
    # napr. slope 3000$ pri BTC 50000$ = 6% → veľké = trend
    df["slope_pct"] = df["slope"].abs() / df["close"] * 100

    # ── FILTER 2: Vzdialenosť od MA100 ──
    # ma100 = priemer posledných 100 sviečok = dlhodobý „stred"
    df["ma100"] = df["close"].rolling(100).mean()

    # dist_pct = koľko percent je cena vzdialená od MA100
    # blízko MA100 (napr. 3%) → range fáza → OK
    # ďaleko od MA100 (napr. 15%) → trend fáza → skip
    df["dist_pct"] = (df["close"] - df["ma100"]).abs() / df["ma100"] * 100

    return df

# ── 3. BACKTEST ───────────────────────────────────────────────────────────────

# Funkcia ktorá simuluje obchodovanie na historických dátach
# entry_z    = pri akom z-score vstúpime (-2.0)
# exit_z     = pri akom z-score vystúpime (0.0)
# stop_z     = stop loss z-score (-3.5)
# max_slope  = maximálny dovolený sklon MA20 v % (filter 1)
# max_dist   = maximálna dovolená vzdialenosť od MA100 v % (filter 2)
def run_backtest(df, entry_z=-2.0, exit_z=0.0, stop_z=-3.5,
                 max_slope=1.5, max_dist=8.0):

    # trades = zoznam všetkých obchodov ktoré spravíme
    trades = []

    # in_trade = sme práve v obchode? (True/False)
    in_trade = False

    # entry_price = za koľko sme nakúpili
    entry_price = None

    # Prejdeme každú sviečku od začiatku do konca
    for i in range(1, len(df)):

        # row = aktuálna sviečka (jej čas, cena, z-score)
        row = df.iloc[i]

        # Ak NIE sme v obchode — hľadáme vstup
        if not in_trade:

            slope_ok = row["slope_pct"] < max_slope
            dist_ok = row["dist_pct"] < max_dist

            if row["zscore"] < entry_z and slope_ok and dist_ok:
                in_trade = True
                entry_price = row["close"]
                entry_time = row["open_time"]
                entry_i = i                        # index vstupu — na meranie trvania
                entry_zscore = row["zscore"]       # z-score pri vstupe
                entry_slope = row["slope_pct"]     # sklon MA pri vstupe
                entry_dist = row["dist_pct"]       # vzdialenosť od MA100 pri vstupe
                max_adverse = 0.0                  # najhorší pohyb počas obchodu (reset)

        # Ak SME v obchode — hľadáme výstup
        else:

            # Sledujeme najhorší pohyb ceny od vstupu (max adverse excursion)
            # adverse = o koľko percent cena klesla od vstupnej ceny
            adverse = (row["close"] - entry_price) / entry_price * 100
            if adverse < max_adverse:
                max_adverse = adverse              # uložíme najhorší bod

            # duration = koľko sviečok sme už v obchode
            duration = i - entry_i

            # Normálny exit: z-score sa vrátil nad 0 → návrat k priemeru
            if row["zscore"] > exit_z:

                pnl = (row["close"] - entry_price) / entry_price * 100

                trades.append({
                    "entry": entry_time, "exit": row["open_time"],
                    "entry_price": entry_price, "exit_price": row["close"],
                    "pnl": pnl, "type": "exit",
                    "entry_zscore": entry_zscore,
                    "entry_slope": entry_slope,
                    "entry_dist": entry_dist,
                    "duration": duration,          # koľko sviečok trval obchod
                    "max_adverse": max_adverse,    # najhorší pokles počas obchodu
                })
                in_trade = False

            # Stop loss: z-score klesol pod -3.5
            elif row["zscore"] < stop_z:

                pnl = (row["close"] - entry_price) / entry_price * 100

                trades.append({
                    "entry": entry_time, "exit": row["open_time"],
                    "entry_price": entry_price, "exit_price": row["close"],
                    "pnl": pnl, "type": "stop",
                    "entry_zscore": entry_zscore,
                    "entry_slope": entry_slope,
                    "entry_dist": entry_dist,
                    "duration": duration,
                    "max_adverse": max_adverse,
                })
                in_trade = False

    # Vrátime tabuľku všetkých obchodov
    return pd.DataFrame(trades)

# ── 4. ŠTATISTIKA ─────────────────────────────────────────────────────────────

# Funkcia ktorá vypočíta a vytlačí výsledky
def print_stats(trades):

    # Ak nemáme žiadne obchody, skončíme
    if len(trades) == 0:
        print("Žiadne obchody.")
        return

    # Oddelíme výherné obchody (pnl > 0) od prehraných (pnl <= 0)
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    # Oddelíme obchody ktoré skončili stopom
    stops = trades[trades["type"] == "stop"]

    # winrate = koľko percent obchodov sme vyhrali
    winrate = len(wins) / len(trades) * 100

    # avg_win = priemerný zisk na výhernom obchode
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0

    # avg_loss = priemerná strata na prehratom obchode
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0

    # Expectancy = očakávaný výnos na jeden obchod
    # ak je kladné → systém dlhodobo zarába
    # ak je záporné → systém dlhodobo stráca
    ev = (len(wins)/len(trades)) * avg_win + (len(losses)/len(trades)) * avg_loss

    # Kumulatívny výnos = súčet všetkých obchodov dohromady
    cumulative = trades["pnl"].sum()

    # Max drawdown = najväčší pokles od vrcholu
    # napr. zarábali sme +20%, potom sme klesli na +5% → drawdown = -15%
    equity = trades["pnl"].cumsum()          # postupný súčet výnosov
    rolling_max = equity.cummax()            # najvyšší bod dovtedy
    drawdown = equity - rolling_max          # rozdiel = pokles
    max_dd = drawdown.min()                  # najhorší pokles

    # Vytlačíme všetky výsledky
    print("=" * 50)
    print(f"  Počet obchodov:     {len(trades)}")
    print(f"  Winrate:            {winrate:.1f}%")
    print(f"  Avg win:            +{avg_win:.2f}%")
    print(f"  Avg loss:           {avg_loss:.2f}%")
    print(f"  Expectancy (EV):    {ev:.2f}% na obchod")
    print(f"  Kumulatívny výnos:  {cumulative:.1f}%")
    print(f"  Max Drawdown:       {max_dd:.2f}%")
    print(f"  Stop lossy:         {len(stops)}/{len(trades)}")
    print("=" * 50)

    # Záver: má edge zmysel?
    if ev > 0 and len(trades) >= 30:
        print("  → Pozitívny EV + dostatok obchodov. Edge existuje.")
    elif len(trades) < 30:
        print("  → Príliš málo obchodov — nedá sa vyvodiť záver.")
    else:
        print("  → Negatívny EV. Edge neexistuje (na tomto nastavení).")
    print()

    # ── LOSS TAXONOMY ──
    # Vypíšeme detail každého prehraneho obchodu
    # cieľ: pochopiť KDE a PREČO vznikajú straty
    if len(losses) > 0:
        print("  LOSING TRADES — detail:")
        print(f"  {'dátum vstupu':<20} {'pnl':>6} {'trvanie':>8} {'max_adverse':>12} {'z pri vstupe':>13} {'slope':>7} {'dist':>6}")
        print("  " + "-" * 80)
        for _, t in losses.iterrows():
            print(f"  {str(t['entry']):<20} "
                  f"{t['pnl']:>+6.2f}% "
                  f"{int(t['duration']):>7}sv "
                  f"{t['max_adverse']:>+11.2f}% "
                  f"{t['entry_zscore']:>12.2f} "
                  f"{t['entry_slope']:>6.2f}% "
                  f"{t['entry_dist']:>5.2f}%")
        print()

# ── 5. GRAF ───────────────────────────────────────────────────────────────────

# Funkcia ktorá nakreslí graf s cenou, MA20 a obchodmi
def plot_trades(df, trades, title="Mean Reversion — BTC 4h"):

    # Vytvoríme okno s 2 grafmi pod sebou (price + z-score)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # ── Horný graf: cena a MA20 ──

    # Modrá čiara = záverečná cena každej sviečky
    ax1.plot(df["open_time"], df["close"], color="steelblue", linewidth=1, label="BTC cena")

    # Oranžová čiara = MA20 (priemer posledných 20 sviečok)
    ax1.plot(df["open_time"], df["ma"], color="orange", linewidth=1.5, label="MA20")

    # Zelené body = miesta kde sme VSTÚPILI do obchodu (z-score < -2)
    if len(trades) > 0:
        entry_rows = df[df["open_time"].isin(trades["entry"])]
        ax1.scatter(entry_rows["open_time"], entry_rows["close"],
                    color="lime", zorder=5, s=80, label="Vstup (z < -2)")

        # Červené/zelené body = miesta kde sme VYSTÚPILI
        for _, t in trades.iterrows():
            exit_row = df[df["open_time"] == t["exit"]]
            if len(exit_row) == 0:
                continue
            # zelený trojuholník = zisk, červený = strata
            color = "green" if t["pnl"] > 0 else "red"
            marker = "^" if t["pnl"] > 0 else "v"
            ax1.scatter(exit_row["open_time"], exit_row["close"],
                        color=color, zorder=5, s=80, marker=marker)

            # Čiara spájajúca vstup a výstup
            entry_row = df[df["open_time"] == t["entry"]]
            if len(entry_row) > 0:
                ax1.plot([entry_row["open_time"].values[0], exit_row["open_time"].values[0]],
                         [entry_row["close"].values[0], exit_row["close"].values[0]],
                         color=color, alpha=0.3, linewidth=1)

    ax1.set_title(title, fontsize=13)
    ax1.set_ylabel("Cena (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # ── Dolný graf: z-score ──

    # Sivá čiara = z-score v čase
    ax2.plot(df["open_time"], df["zscore"], color="gray", linewidth=1, label="z-score")

    # Červená prerušovaná čiara = hranica vstupu (-2.0)
    ax2.axhline(-2.0, color="red", linestyle="--", linewidth=1, label="Vstup (-2.0)")

    # Zelená prerušovaná čiara = hranica exitu (0)
    ax2.axhline(0.0, color="green", linestyle="--", linewidth=1, label="Exit (0)")

    # Modrá prerušovaná čiara = nulová línia referencia
    ax2.axhline(-3.5, color="darkred", linestyle=":", linewidth=1, label="Stop (-3.5)")

    # Podfarbíme oblasť kde z-score < -2 (kde hľadáme vstup)
    ax2.fill_between(df["open_time"], df["zscore"], -2.0,
                     where=(df["zscore"] < -2.0), color="red", alpha=0.2)

    ax2.set_ylabel("z-score")
    ax2.set_xlabel("Čas")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Uložíme graf ako PNG súbor
    plt.savefig("mean_reversion_chart.png", dpi=150)
    print("Graf uložený: mean_reversion_chart.png")
    plt.show()

# ── 6. MAIN ───────────────────────────────────────────────────────────────────

# Toto sa spustí keď napíšeš: python mean_reversion_test.py
if __name__ == "__main__":

    # Stiahni dáta z Binance
    print("Sťahujem dáta BTC 4h...")
    df = fetch_data(symbol="BTCUSDT", interval="4h", limit=1500)

    # Vypočítaj z-score pre každú sviečku
    df = add_zscore(df, window=20)

    # Vyhoď riadky kde z-score ešte nie je vypočítaný (prvých 20 sviečok)
    df = df.dropna().reset_index(drop=True)

    # Vypíš info o dátach
    print(f"Dáta: {len(df)} sviečok | od {df['open_time'].iloc[0].date()} do {df['open_time'].iloc[-1].date()}\n")

    # Experiment 1: BEZ filtrov (pôvodné, porovnanie)
    print("── BEZ filtrov (entry=-2.0, exit=0, stop=-3.5) ──")
    trades = run_backtest(df, entry_z=-2.0, exit_z=0.0, stop_z=-3.5,
                          max_slope=999, max_dist=999)  # 999 = filter vypnutý
    print_stats(trades)

    # Experiment 2: S filtrami — len range fáza
    # max_slope=1.5 → MA20 sa smie pohnúť max 1.5% za 5 sviečok
    # max_dist=8.0  → cena smie byť max 8% od MA100
    print("── S filtrami (slope<1.5%, dist<8%) ──")
    trades2 = run_backtest(df, entry_z=-2.0, exit_z=0.0, stop_z=-3.5,
                           max_slope=1.5, max_dist=8.0)
    print_stats(trades2)

    # Experiment 3: Prísnejšie filtre
    print("── Prísne filtre (slope<1.0%, dist<5%) ──")
    trades3 = run_backtest(df, entry_z=-2.0, exit_z=0.0, stop_z=-3.5,
                           max_slope=1.0, max_dist=5.0)
    print_stats(trades3)

    # Graf pre experiment 2 (s filtrami) — uloží mean_reversion_chart.png
    plot_trades(df, trades2, title="Mean Reversion + Trend Filter — BTC 4h")

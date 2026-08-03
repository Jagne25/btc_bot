"""
Týždeň 1 — prvý dotyk Order Flow dát.
1) aggTrades — delta/CVD po jednotlivých obchodoch (detail, max 1000)
2) klines — delta/CVD po minútach cez taker_buy_base (dlhšia história)
3) graf — cena + CVD za posledných N minút (ukladá na SSD)
4) divergencia — cena vs CVD za posledných N minút
5) live — WebSocket aggTrades, bežiaca delta/CVD + ALERT pri veľkom obchode

Spusti snapshot:  python order_flow_week1.py
Spusti live:      python order_flow_week1.py live
"""
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt

SYMBOL = "BTCUSDT"
MINUTES = 5       # aggTrades: koľko minút spätne
KLINE_INTERVAL = "1m"
KLINE_LOOKBACK = 60  # klines: koľko sviečok (60 × 1m = 60 min)
DIVERGENCE_LOOKBACK = 20  # minút na porovnanie cena vs CVD
CHART_NAME = "order_flow_chart.png"
# Live: vypíš každý N-tý obchod, alebo vždy ak qty >= LIVE_MIN_QTY
LIVE_PRINT_EVERY = 10
LIVE_MIN_QTY = 0.01
# ALERT: relatívny prah — ≥ N× rolling priemer posledných K obchodov
LIVE_ALERT_ROLLING = 100
LIVE_ALERT_MULT = 10.0
LIVE_ALERT_QTY_FALLBACK = 0.5  # kým nie je dosť histórie
LIVE_DIVERGENCE_SEC = 60  # check každú 1 min (neskôr môžeš dať 5 * 60)
_BASE_SSD = Path(os.getenv("BASE_SSD_DIR", "/Volumes/WORK_SSD/TradingData/btc_bot"))


def chart_path(symbol: str = SYMBOL) -> Path:
    """Cesta pre graf na SSD: .../btc_bot/BTC/exports/order_flow_chart.png"""
    if not _BASE_SSD.exists():
        raise RuntimeError(f"SSD nie je pripojené: {_BASE_SSD}")
    coin = symbol.replace("USDT", "")
    out_dir = _BASE_SSD / coin / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / CHART_NAME


def get_klines(symbol="BTCUSDT", interval="1d", lookback=1000):
    """Rovnaká logika ako bot.get_klines — tu priamo, bez importu bot.py (ML závislosti)."""
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

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df = df.sort_values("open_time").drop_duplicates(subset="open_time", keep="last").reset_index(drop=True)
    if len(df) > lookback:
        df = df.iloc[-lookback:].reset_index(drop=True)
    return df


def nacitaj_agg_trades(symbol: str, minutes: int = 5) -> pd.DataFrame:
    """Posledných N minút agregovaných obchodov z Binance (verejné API, bez kľúča)."""
    url = "https://api.binance.com/api/v3/aggTrades"
    start_ms = int((time.time() - minutes * 60) * 1000)

    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "limit": 1000,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    raw = r.json()

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df = df.rename(columns={
        "a": "trade_id",
        "p": "price",
        "q": "qty",
        "T": "time_ms",
        "m": "buyer_is_maker",
    })
    df["price"] = pd.to_numeric(df["price"])
    df["qty"] = pd.to_numeric(df["qty"])
    df["time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True)

    # m=True  → kupujúci bol maker → predávajúci bol agresor (sell)
    # m=False → predávajúci bol maker → kupujúci bol agresor (buy)
    df["side"] = df["buyer_is_maker"].map({True: "sell", False: "buy"})
    df["buy_vol"] = df["qty"].where(df["side"] == "buy", 0.0)
    df["sell_vol"] = df["qty"].where(df["side"] == "sell", 0.0)
    df["delta"] = df["buy_vol"] - df["sell_vol"]
    df["cvd"] = df["delta"].cumsum()

    return df


def cvd_z_klines(symbol: str, interval: str = "1m", lookback: int = 60) -> pd.DataFrame:
    """Delta + CVD z klines — Binance už dá taker_buy_base (agresívne kúpy) per sviečka."""
    df = get_klines(symbol, interval, lookback)
    df["taker_sell"] = df["volume"] - df["taker_buy_base"]
    df["delta"] = df["taker_buy_base"] - df["taker_sell"]
    df["cvd"] = df["delta"].cumsum()
    return df


def vykresli_graf(df: pd.DataFrame, symbol: str, interval: str, path: Path) -> None:
    """Week 2: cena hore, CVD dole — rovnaká časová os."""
    fig, (ax_price, ax_cvd) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax_price.plot(df["open_time"], df["close"], color="#2563eb", linewidth=1.2)
    ax_price.set_ylabel("Cena (USDT)")
    ax_price.set_title(f"{symbol} — cena + CVD ({len(df)}× {interval})")
    ax_price.grid(True, alpha=0.3)

    ax_cvd.plot(df["open_time"], df["cvd"], color="#16a34a", linewidth=1.2)
    ax_cvd.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax_cvd.set_ylabel("CVD (BTC)")
    ax_cvd.set_xlabel("Čas (UTC)")
    ax_cvd.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def detekuj_divergenciu(df: pd.DataFrame, lookback: int = DIVERGENCE_LOOKBACK) -> None:
    """Week 3: porovná smer ceny a CVD za posledných N minútových sviečok."""
    if len(df) < lookback:
        print(f"Málo sviečok na divergenciu (máme {len(df)}, treba {lookback}).")
        return

    okno = df.tail(lookback)
    price_start, price_end = float(okno["close"].iloc[0]), float(okno["close"].iloc[-1])
    cvd_start, cvd_end = float(okno["cvd"].iloc[0]), float(okno["cvd"].iloc[-1])

    if price_end > price_start:
        smer_cena = "rástla"
    elif price_end < price_start:
        smer_cena = "klesala"
    else:
        smer_cena = "bez zmeny"

    if cvd_end > cvd_start:
        smer_cvd = "rástol"
    elif cvd_end < cvd_start:
        smer_cvd = "klesal"
    else:
        smer_cvd = "bez zmeny"

    print(f"\n=== Divergencia (posledných {lookback} min) ===")
    print(f"Cena: {price_start:.2f} → {price_end:.2f} ({smer_cena})")
    print(f"CVD:  {cvd_start:+.4f} → {cvd_end:+.4f} ({smer_cvd})")

    cena_hore = price_end > price_start
    cena_dole = price_end < price_start
    cvd_hore = cvd_end > cvd_start
    cvd_dole = cvd_end < cvd_start

    if (cena_hore and cvd_dole) or (cena_dole and cvd_hore):
        print(f"DIVERGENCIA: cena {smer_cena}, CVD {smer_cvd} — možný obrat")
    else:
        print("Cena a CVD sedia, žiadna divergencia")


def live_agg_trades(symbol: str = SYMBOL) -> None:
    """Week 4: live aggTrades — CVD + ALERT + každých N min check divergencie."""
    import ssl

    try:
        import websocket
    except ImportError:
        print("Chýba balík websocket-client. Nainštaluj: pip3 install --user websocket-client")
        return

    stream = f"{symbol.lower()}@aggTrade"
    url = f"wss://stream.binance.com:9443/ws/{stream}"
    cvd = 0.0
    n = 0
    buy_vol = 0.0
    sell_vol = 0.0
    # Okno na live divergenciu: zapamätáme cenu+CVD na začiatku, po N sekundách porovnáme
    win_t0 = None
    win_price0 = None
    win_cvd0 = None
    # veľkosti predchádzajúcich obchodov (rolling priemer na ALERT)
    qty_hist: deque[float] = deque(maxlen=LIVE_ALERT_ROLLING)

    print(f"=== Order Flow Týždeň 4 LIVE: {symbol} ===")
    print("WebSocket aggTrades — rovnaká logika ako Week 1 (buyer_is_maker → side).")
    print(
        f"ALERT: ≥{LIVE_ALERT_MULT:.0f}× priemer posledných {LIVE_ALERT_ROLLING} obchodov "
        f"(záloha {LIVE_ALERT_QTY_FALLBACK} BTC kým <{LIVE_ALERT_ROLLING}) | "
        f"divergencia každých {LIVE_DIVERGENCE_SEC}s"
    )
    print("Ukonči: Ctrl+C\n")

    def on_message(_ws, message: str) -> None:
        nonlocal cvd, n, buy_vol, sell_vol, win_t0, win_price0, win_cvd0
        data = json.loads(message)
        price = float(data["p"])
        qty = float(data["q"])
        # m=True → kupujúci bol maker → agresor = sell
        side = "sell" if data["m"] else "buy"
        delta = qty if side == "buy" else -qty
        cvd += delta
        n += 1
        if side == "buy":
            buy_vol += qty
        else:
            sell_vol += qty

        now = time.time()
        if win_t0 is None:
            win_t0, win_price0, win_cvd0 = now, price, cvd

        # prah: relatívny až máme plné okno predchádzajúcich obchodov
        if len(qty_hist) >= LIVE_ALERT_ROLLING:
            avg_qty = sum(qty_hist) / len(qty_hist)
            threshold = LIVE_ALERT_MULT * avg_qty
            use_relative = True
        else:
            avg_qty = sum(qty_hist) / len(qty_hist) if qty_hist else 0.0
            threshold = LIVE_ALERT_QTY_FALLBACK
            use_relative = False

        if qty >= threshold and threshold > 0:
            if use_relative and avg_qty > 0:
                mult = qty / avg_qty
                print(
                    f"*** ALERT {side.upper():4s} {qty:.5f} @ {price:.2f} | "
                    f"priemer posledných {LIVE_ALERT_ROLLING} = {avg_qty:.5f} BTC, "
                    f"{mult:.1f}x nad priemerom | "
                    f"delta {delta:+.5f} | CVD {cvd:+.4f} ***"
                )
            else:
                print(
                    f"*** ALERT {side.upper():4s} {qty:.5f} @ {price:.2f} | "
                    f"záložný prah {LIVE_ALERT_QTY_FALLBACK} BTC "
                    f"(história {len(qty_hist)}/{LIVE_ALERT_ROLLING}, "
                    f"priemer zatiaľ {avg_qty:.5f}) | "
                    f"delta {delta:+.5f} | CVD {cvd:+.4f} ***"
                )
        elif qty >= LIVE_MIN_QTY or n % LIVE_PRINT_EVERY == 0:
            print(
                f"LIVE #{n:5d} | {side:4s} {qty:8.5f} @ {price:.2f} | "
                f"delta {delta:+.5f} | CVD {cvd:+.4f}"
            )

        qty_hist.append(qty)

        # Každých LIVE_DIVERGENCE_SEC: porovnaj smer ceny a CVD v okne
        if now - win_t0 >= LIVE_DIVERGENCE_SEC:
            if price > win_price0:
                smer_cena = "rástla"
            elif price < win_price0:
                smer_cena = "klesala"
            else:
                smer_cena = "bez zmeny"

            if cvd > win_cvd0:
                smer_cvd = "rástol"
            elif cvd < win_cvd0:
                smer_cvd = "klesal"
            else:
                smer_cvd = "bez zmeny"

            print(f"\n--- Divergencia check ({LIVE_DIVERGENCE_SEC}s) ---")
            print(f"Cena: {win_price0:.2f} → {price:.2f} ({smer_cena})")
            print(f"CVD:  {win_cvd0:+.4f} → {cvd:+.4f} ({smer_cvd})")
            if (price > win_price0 and cvd < win_cvd0) or (price < win_price0 and cvd > win_cvd0):
                print(f"DIVERGENCIA: cena {smer_cena}, CVD {smer_cvd} — možný obrat\n", flush=True)
            else:
                print("Cena a CVD sedia, žiadna divergencia\n", flush=True)

            # nové okno odteraz
            win_t0, win_price0, win_cvd0 = now, price, cvd

    def on_error(_ws, error) -> None:
        print(f"WebSocket chyba: {error}")

    def on_close(_ws, close_status_code, close_msg) -> None:
        print(f"\nSpojenie zatvorené ({close_status_code}).")
        print(f"Súhrn relácie: {n} obchodov | buy {buy_vol:.4f} | sell {sell_vol:.4f} | CVD {cvd:+.4f}")

    def on_open(_ws) -> None:
        print(f"Pripojené: {url}\n")

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    sslopt = {"cert_reqs": ssl.CERT_NONE}
    try:
        ws.run_forever(sslopt=sslopt)
    except KeyboardInterrupt:
        print("\nCtrl+C — ukončujem live…")
        ws.close()
        print(f"Súhrn relácie: {n} obchodov | buy {buy_vol:.4f} | sell {sell_vol:.4f} | CVD {cvd:+.4f}")


def main_snapshot():
    print(f"=== Order Flow Týždeň 1: {SYMBOL} (posledných {MINUTES} min) ===\n")

    df = nacitaj_agg_trades(SYMBOL, MINUTES)
    if df.empty:
        print("Žiadne obchody — skús neskôr alebo zväčši MINUTES.")
        return

    total_buy = df["buy_vol"].sum()
    total_sell = df["sell_vol"].sum()
    total_delta = total_buy - total_sell
    final_cvd = df["cvd"].iloc[-1]

    print(f"Počet obchodov: {len(df)}")
    print(f"Agresívne kúpy:  {total_buy:.4f} BTC")
    print(f"Agresívne predaje: {total_sell:.4f} BTC")
    print(f"Delta (kúpy - predaje): {total_delta:+.4f} BTC")
    print(f"CVD (kumulatívna delta): {final_cvd:+.4f} BTC")
    print()

    if total_delta > 0:
        print("→ Kupujúci dominujú (delta kladná)")
    elif total_delta < 0:
        print("→ Predávajúci dominujú (delta záporná)")
    else:
        print("→ Rovnováha")

    print("\n--- Posledných 10 obchodov ---")
    cols = ["time", "price", "qty", "side", "delta", "cvd"]
    print(df[cols].tail(10).to_string(index=False))

    # --- Week 1.5: CVD z klines (súhrn po minútach, celá história bez limitu 1000) ---
    print(f"\n\n=== Order Flow Týždeň 1.5: {SYMBOL} ({KLINE_LOOKBACK}× {KLINE_INTERVAL}) ===\n")

    df_k = cvd_z_klines(SYMBOL, KLINE_INTERVAL, KLINE_LOOKBACK)
    if df_k.empty:
        print("Žiadne klines — skús neskôr.")
        return

    total_delta_k = df_k["delta"].sum()
    final_cvd_k = df_k["cvd"].iloc[-1]

    print(f"Počet sviečok: {len(df_k)}")
    print(f"Celková delta ({KLINE_LOOKBACK} min): {total_delta_k:+.4f} BTC")
    print(f"CVD (kumulatívna):              {final_cvd_k:+.4f} BTC")
    print()

    if total_delta_k > 0:
        print("→ Kupujúci dominujú (delta kladná)")
    elif total_delta_k < 0:
        print("→ Predávajúci dominujú (delta záporná)")
    else:
        print("→ Rovnováha")

    print(f"\n--- Posledných 5 minút (klines) ---")
    cols_k = ["open_time", "close", "volume", "taker_buy_base", "delta", "cvd"]
    print(df_k[cols_k].tail(5).to_string(index=False))

    # --- Week 3: divergencia cena vs CVD ---
    detekuj_divergenciu(df_k, DIVERGENCE_LOOKBACK)

    # --- Week 2: graf cena + CVD ---
    out = chart_path(SYMBOL)
    vykresli_graf(df_k, SYMBOL, KLINE_INTERVAL, out)
    print(f"\n→ Graf uložený: {out}")


def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() == "live":
        live_agg_trades(SYMBOL)
    else:
        main_snapshot()


if __name__ == "__main__":
    main()

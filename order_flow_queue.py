"""
Order Flow — Queue / Order Book.
1) REST snapshot — top 10 + imbalance
2) live — WebSocket depth, priebežný imbalance + ALERT pri |imb| > 0.8

Spusti snapshot:  python3 order_flow_queue.py
Spusti live:      python3 order_flow_queue.py live
"""
import json
import ssl
import sys
import time

import requests

SYMBOL = "BTCUSDT"
LIMIT = 10  # top N úrovní (bids + asks)
# Live: vypíš max raz za N sekúnd (depth ide ~100ms, inak by terminal lietal)
LIVE_PRINT_EVERY_SEC = 0.5
LIVE_ALERT_IMB = 0.8  # |imb| nad týmto → ALERT (raz pri vstupe do zóny)


def nacitaj_depth(symbol: str = SYMBOL, limit: int = LIMIT) -> dict:
    """Jednorazový snapshot order book z Binance (verejné API)."""
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def spocitaj_imbalance(raw: dict) -> tuple[float, float, float]:
    """
    bid_vol / ask_vol / imbalance z top N úrovní.
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)  ∈ [-1, +1]
    """
    bids = raw.get("bids", [])
    asks = raw.get("asks", [])
    bid_vol = sum(float(qty) for _, qty in bids)
    ask_vol = sum(float(qty) for _, qty in asks)
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
    return bid_vol, ask_vol, imbalance


def vypis_knihu(raw: dict, symbol: str = SYMBOL) -> None:
    """Vypíše surové bids/asks — cena a množstvo."""
    print(f"=== Order Book Q1: {symbol} (top {LIMIT}) ===\n")
    print(f"lastUpdateId: {raw.get('lastUpdateId')}\n")

    bids = raw.get("bids", [])  # [[price, qty], ...] — od najvyššej kúpy
    asks = raw.get("asks", [])  # [[price, qty], ...] — od najnižšieho predaja

    print("--- BIDS (čakajúce KÚPY, maker buy) ---")
    print(f"{'#':>3}  {'cena':>12}  {'qty (BTC)':>14}")
    for i, (price, qty) in enumerate(bids, start=1):
        print(f"{i:>3}  {float(price):>12.2f}  {float(qty):>14.5f}")

    print()
    print("--- ASKS (čakajúce PREDAJE, maker sell) ---")
    print(f"{'#':>3}  {'cena':>12}  {'qty (BTC)':>14}")
    for i, (price, qty) in enumerate(asks, start=1):
        print(f"{i:>3}  {float(price):>12.2f}  {float(qty):>14.5f}")

    if bids and asks:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        print()
        print(f"Best bid: {best_bid:.2f}")
        print(f"Best ask: {best_ask:.2f}")
        print(f"Spread:   {spread:.2f} USDT")

    bid_vol, ask_vol, imbalance = spocitaj_imbalance(raw)
    print()
    print("--- Queue imbalance (top N) ---")
    print(f"bid_vol:    {bid_vol:.5f} BTC")
    print(f"ask_vol:    {ask_vol:.5f} BTC")
    print(f"imbalance:  {imbalance:+.4f}   (−1 … +1)")
    if imbalance > 0:
        print("→ Viac čakajúcich kúpov (bid pressure)")
    elif imbalance < 0:
        print("→ Viac čakajúcich predajov (ask pressure)")
    else:
        print("→ Rovnováha")


def live_depth(symbol: str = SYMBOL, limit: int = LIMIT) -> None:
    """Q3: live partial book depth — priebežný imbalance + ALERT pri silnej stene."""
    try:
        import websocket
    except ImportError:
        print("Chýba balík websocket-client. Nainštaluj: pip3 install --user websocket-client")
        return

    stream = f"{symbol.lower()}@depth{limit}@100ms"
    url = f"wss://stream.binance.com:9443/ws/{stream}"
    n = 0
    last_print = 0.0
    last_imb = None
    # True = už sme v |imb| > prahu a alert sme vypísali (kým nespadne späť)
    in_strong_zone = False

    print(f"=== Order Book LIVE: {symbol} (top {limit}) ===")
    print("WebSocket depth — rovnaký imbalance ako snapshot.")
    print(f"ALERT pri |imb| > {LIVE_ALERT_IMB} (len pri vstupe do zóny, nie každý riadok).")
    print(f"Výpis max každých {LIVE_PRINT_EVERY_SEC}s. Ukonči: Ctrl+C\n")

    def on_message(_ws, message: str) -> None:
        nonlocal n, last_print, last_imb, in_strong_zone
        raw = json.loads(message)
        # Partial book: "bids"/"asks"; niektoré streamy majú "b"/"a"
        if "bids" not in raw and "b" in raw:
            raw = {"bids": raw["b"], "asks": raw["a"], "lastUpdateId": raw.get("lastUpdateId")}

        bid_vol, ask_vol, imbalance = spocitaj_imbalance(raw)
        n += 1

        best_bid = float(raw["bids"][0][0]) if raw.get("bids") else 0.0
        best_ask = float(raw["asks"][0][0]) if raw.get("asks") else 0.0

        # ALERT: len keď |imb| PREKROČÍ prah (nie pri každom LIVE riadku v zóne)
        strong = abs(imbalance) > LIVE_ALERT_IMB
        if strong and not in_strong_zone:
            if imbalance > 0:
                print(
                    f"*** ALERT BID WALL | imb {imbalance:+.4f} | "
                    f"bid {bid_vol:.3f} | ask {ask_vol:.3f} | "
                    f"{best_bid:.2f}/{best_ask:.2f} ***"
                )
            else:
                print(
                    f"*** ALERT ASK WALL | imb {imbalance:+.4f} | "
                    f"bid {bid_vol:.3f} | ask {ask_vol:.3f} | "
                    f"{best_bid:.2f}/{best_ask:.2f} ***"
                )
        in_strong_zone = strong

        now = time.time()
        if now - last_print < LIVE_PRINT_EVERY_SEC:
            return
        last_print = now

        delta_imb = "" if last_imb is None else f" | Δimb {imbalance - last_imb:+.3f}"
        last_imb = imbalance

        print(
            f"LIVE #{n:5d} | imb {imbalance:+.4f} | "
            f"bid {bid_vol:7.3f} | ask {ask_vol:7.3f} | "
            f"{best_bid:.2f}/{best_ask:.2f}{delta_imb}"
        )

    def on_error(_ws, error) -> None:
        print(f"WebSocket chyba: {error}")

    def on_close(_ws, close_status_code, _close_msg) -> None:
        print(f"\nSpojenie zatvorené ({close_status_code}). Updates: {n}")

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
        print(f"Updates: {n}")


def main_snapshot() -> None:
    raw = nacitaj_depth(SYMBOL, LIMIT)
    vypis_knihu(raw, SYMBOL)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].lower() == "live":
        live_depth(SYMBOL, LIMIT)
    else:
        main_snapshot()


if __name__ == "__main__":
    main()

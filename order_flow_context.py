"""
Order Flow — kontext CVD + IMB + Absorption (live) + ukladanie.

Sleduje naraz:
  - CVD z aggTrades (agresívny flow)
  - IMB z order book depth (fronta v knihe)
  - Absorption: silný |ΔCVD| za ~20s, ale cena takmer drží

Výpis + zápis (len čítanie trhu, žiadne obchodovanie):
  - CVD rastie (oproti predchádzajúcemu riadku) + IMB > 0 → BUY kontext (silný)
  - CVD klesá + IMB < 0 → SELL kontext (silný)
  - opačné smery → OPATRNOSŤ
  - Absorption (bullish/bearish tlak, cena drží) — len výpis/ALERT
  - Large Order: obchod ≥10× priemer posledných 100 (záloha 0.5 BTC)

Ukladanie (IBA SSD / Postgres na SSD):
  1) CSV:  /Volumes/WORK_SSD/TradingData/btc_bot/BTC/order_flow_context/
  2) PG:   tabuľka order_flow_context_log

Spusti: python3 order_flow_context.py
Ukonči: Ctrl+C
"""
from __future__ import annotations

import csv
import json
import os
import ssl
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

SYMBOL = "BTCUSDT"
DEPTH_LIMIT = 10
PRINT_EVERY_SEC = 1.0  # max 1 riadok za sekundu

# Absorption — prahy z context_log (~6748 riadkov): |ΔCVD| p90≈4.86 BTC / 20s
ABSORPTION_LOOKBACK_SEC = 20.0
MIN_CVD_CHANGE_ABS = 4.9       # ≈ p90 |ΔCVD| za 20s
MAX_PRICE_CHANGE_PCT = 0.05    # cena „drží“ (menej ako 0.05 %)

# Large Order — rovnaká logika ako order_flow_week1.py live
LARGE_ORDER_ROLLING = 100
LARGE_ORDER_MULT = 10.0
LARGE_ORDER_QTY_FALLBACK = 0.5  # kým < ROLLING obchodov

_BASE_SSD = Path(os.getenv("BASE_SSD_DIR", "/Volumes/WORK_SSD/TradingData/btc_bot"))
OUT_DIR = _BASE_SSD / SYMBOL.replace("USDT", "") / "order_flow_context"

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)

CSV_FIELDS = ("ts_utc", "symbol", "price", "cvd", "imb", "cvd_dir", "kontext")


def ensure_ssd() -> Path:
    """Všetky súbory len na SSD — bez SSD nič neukladáme."""
    if not _BASE_SSD.exists():
        raise RuntimeError(
            f"SSD nie je pripojené: {_BASE_SSD}\n"
            "Pripoj WORK_SSD a spusti znova. Nič sa neukladá na interný disk."
        )
    # ochrana: cesta musí byť na WORK_SSD
    resolved = _BASE_SSD.resolve()
    if "/Volumes/WORK_SSD" not in str(resolved):
        raise RuntimeError(f"BASE_SSD_DIR nie je na WORK_SSD: {resolved}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def csv_path_for_today(out_dir: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return out_dir / f"context_log_{day}.csv"


def open_csv(out_dir: Path):
    path = csv_path_for_today(out_dir)
    new_file = not path.exists() or path.stat().st_size == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()
    return path, f, writer


def pg_connect():
    try:
        import psycopg2
    except ImportError:
        print("PG: psycopg2 nie je nainštalované — ukladám len CSV.")
        return None
    try:
        conn = psycopg2.connect(**PGCFG)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS order_flow_context_log (
                    id          BIGSERIAL PRIMARY KEY,
                    ts_utc      TIMESTAMPTZ NOT NULL,
                    symbol      TEXT NOT NULL,
                    price       DOUBLE PRECISION,
                    cvd         DOUBLE PRECISION,
                    imb         DOUBLE PRECISION,
                    cvd_dir     TEXT,
                    kontext     TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_of_context_log_ts
                    ON order_flow_context_log (ts_utc);
                """
            )
        return conn
    except Exception as e:
        print(f"PG: nedá sa pripojiť ({e})")
        print("  Zapni: /Volumes/WORK_SSD/TradingData/btc_bot/postgres/zapni_postgres.sh")
        print("  Pokračujem len s CSV na SSD.")
        return None


def imbalance_from_book(bids, asks) -> tuple[float, float, float]:
    bid_vol = sum(float(q) for _, q in bids)
    ask_vol = sum(float(q) for _, q in asks)
    total = bid_vol + ask_vol
    imb = (bid_vol - ask_vol) / total if total > 0 else 0.0
    return bid_vol, ask_vol, imb


def cvd_smer(cvd_now: float, cvd_prev_print: float | None) -> str:
    """Porovná aktuálne CVD s hodnotou z predchádzajúceho výpisového riadku."""
    if cvd_prev_print is None:
        return "neznámy"
    if cvd_now > cvd_prev_print:
        return "rastie"
    if cvd_now < cvd_prev_print:
        return "klesá"
    return "flat"


def kontext(cvd_dir: str, imb: float) -> str:
    if cvd_dir == "rastie" and imb > 0:
        return "BUY kontext (silný)"
    if cvd_dir == "klesá" and imb < 0:
        return "SELL kontext (silný)"
    if cvd_dir in ("rastie", "klesá") and imb != 0:
        if (cvd_dir == "rastie" and imb < 0) or (cvd_dir == "klesá" and imb > 0):
            return "OPATRNOSŤ - signály sa rozchádzajú"
    if cvd_dir == "flat" or cvd_dir == "neznámy":
        return "čakám na CVD smer (ešte málo dát)"
    if imb == 0:
        return "IMB ~0 (kniha vyvážená)"
    return "neutrálny / slabý kontext"


def window_changes(
    hist: deque[tuple[float, float, float]], now: float, lookback: float
) -> tuple[float, float, float] | None:
    """
    Vráť (ΔCVD, Δprice_pct, lag_sec) oproti bodu ~lookback sekúnd dozadu.
    hist: (timestamp, cvd, price)
    """
    if len(hist) < 2:
        return None
    target = now - lookback
    best = hist[0]
    for pair in hist:
        if abs(pair[0] - target) < abs(best[0] - target):
            best = pair
    t_then, cvd_then, price_then = best
    lag = now - t_then
    # potrebujeme približne celé okno
    if lag < lookback * 0.75 or price_then <= 0:
        return None
    cvd_now = hist[-1][1]
    price_now = hist[-1][2]
    d_cvd = cvd_now - cvd_then
    d_price_pct = 100.0 * (price_now - price_then) / price_then
    return d_cvd, d_price_pct, lag


def detect_absorption(d_cvd: float, d_price_pct: float) -> str | None:
    """
    Silný agresívny tlak (|ΔCVD| veľké) + cena takmer drží (|Δprice| malé).
    """
    if abs(d_cvd) < MIN_CVD_CHANGE_ABS:
        return None
    if abs(d_price_pct) >= MAX_PRICE_CHANGE_PCT:
        return None
    if d_cvd > 0:
        return "ABSORPTION (bullish tlak, cena drží)"
    return "ABSORPTION (bearish tlak, cena drží)"


def main() -> None:
    try:
        import websocket
    except ImportError:
        print("Chýba websocket-client. Nainštaluj: pip3 install --user websocket-client")
        return

    out_dir = ensure_ssd()
    csv_file_path, csv_f, csv_writer = open_csv(out_dir)
    pg_conn = pg_connect()
    pg_ok = pg_conn is not None
    pg_fail_printed = False
    n_saved = 0

    sym = SYMBOL.lower()
    streams = f"{sym}@aggTrade/{sym}@depth{DEPTH_LIMIT}@100ms"
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    cvd = 0.0
    imb = 0.0
    bid_vol = 0.0
    ask_vol = 0.0
    price = 0.0
    n_trades = 0
    n_books = 0
    last_print = 0.0
    cvd_prev_print: float | None = None
    current_csv_day = datetime.now(timezone.utc).strftime("%Y%m%d")
    # (ts, cvd, price) pre absorption okno
    hist: deque[tuple[float, float, float]] = deque()
    abs_active = False  # edge-trigger ALERT
    prev_ctx: str | None = None
    # súhrn behu (len prechody / vstupy)
    abs_events: list[dict] = []  # {kind, ts, price, d_cvd, d_px}
    ctx_events: list[dict] = []  # {kind, ts, price, ctx}  kind=BUY|SELL
    qty_hist: deque[float] = deque(maxlen=LARGE_ORDER_ROLLING)
    large_events: list[dict] = []  # veľké obchody
    summary_printed = False

    print(f"=== Order Flow CONTEXT (CVD + IMB + Absorption + Large): {SYMBOL} ===")
    print(f"CVD smer = oproti predchádzajúcemu riadku | IMB z top {DEPTH_LIMIT}")
    print(
        f"Absorption: |ΔCVD|≥{MIN_CVD_CHANGE_ABS} BTC za {ABSORPTION_LOOKBACK_SEC:.0f}s "
        f"a |Δprice|<{MAX_PRICE_CHANGE_PCT}% (prah ≈ p90 z logov)"
    )
    print(
        f"Large Order: ≥{LARGE_ORDER_MULT:.0f}× priemer posledných {LARGE_ORDER_ROLLING} "
        f"(záloha {LARGE_ORDER_QTY_FALLBACK} BTC kým <{LARGE_ORDER_ROLLING})"
    )
    print(f"CSV → {csv_file_path}")
    print(
        f"PG  → order_flow_context_log"
        + (" (OK)" if pg_ok else " (vypnuté — len CSV)")
    )
    print("Len výpis + log — žiadne obchodovanie. Ctrl+C = koniec.\n")

    def print_run_summary() -> None:
        nonlocal summary_printed
        if summary_printed:
            return
        summary_printed = True

        bull = [e for e in abs_events if e["kind"] == "bullish"]
        bear = [e for e in abs_events if e["kind"] == "bearish"]
        buys = [e for e in ctx_events if e["kind"] == "BUY"]
        sells = [e for e in ctx_events if e["kind"] == "SELL"]

        print("\n" + "=" * 60)
        print("=== SÚHRN BEHU ===")
        print(f"trades={n_trades} books={n_books} | CVD {cvd:+.4f} | uložené riadky={n_saved}")
        print(f"CSV: {csv_file_path}")
        print()
        print("--- BUY / SELL kontext (len PRECHODY) ---")
        print(f"BUY  prechody: {len(buys)}")
        print(f"SELL prechody: {len(sells)}")
        print()
        print("--- ABSORPTION (len vstupy / udalosti) ---")
        print(f"bullish (CVD↑, cena drží): {len(bull)}")
        print(f"bearish (CVD↓, cena drží): {len(bear)}")
        print(f"spolu: {len(abs_events)}")

        def _fmt_ts(ts: datetime) -> str:
            return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S")

        if abs_events:
            print()
            print("Zoznam absorption udalostí:")
            for i, e in enumerate(abs_events, 1):
                print(
                    f"  {i:3d}. {_fmt_ts(e['ts'])} | {e['kind']:8s} | "
                    f"px {e['price']:.1f} | ΔCVD {e['d_cvd']:+.2f} | "
                    f"Δpx {e['d_px']:+.3f}%"
                )
        else:
            print("\nŽiadna absorption udalosť v tomto behu.")

        print()
        print("--- LARGE ORDERS ---")
        print(f"počet: {len(large_events)}")
        if large_events:
            for i, e in enumerate(large_events, 1):
                print(
                    f"  {i:3d}. {_fmt_ts(e['ts'])} | {e['side']:4s} | "
                    f"{e['qty']:.5f} BTC @ {e['price']:.1f} | "
                    f"priemer {e['avg']:.5f} | {e['mult']:.1f}x"
                )
        print("=" * 60)

    def rotate_csv_if_needed() -> None:
        nonlocal csv_file_path, csv_f, csv_writer, current_csv_day
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        if day == current_csv_day:
            return
        csv_f.flush()
        csv_f.close()
        current_csv_day = day
        csv_file_path, csv_f, csv_writer = open_csv(out_dir)
        print(f"\n[CSV] nový deň → {csv_file_path}\n")

    def save_row(ts: datetime, price_v: float, cvd_v: float, imb_v: float,
                 direction: str, ctx: str) -> None:
        nonlocal n_saved, pg_conn, pg_ok, pg_fail_printed
        rotate_csv_if_needed()
        row = {
            "ts_utc": ts.isoformat(),
            "symbol": SYMBOL,
            "price": f"{price_v:.2f}",
            "cvd": f"{cvd_v:.6f}",
            "imb": f"{imb_v:.6f}",
            "cvd_dir": direction,
            "kontext": ctx,
        }
        csv_writer.writerow(row)
        csv_f.flush()

        if pg_ok and pg_conn is not None:
            try:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO order_flow_context_log
                            (ts_utc, symbol, price, cvd, imb, cvd_dir, kontext)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (ts, SYMBOL, price_v, cvd_v, imb_v, direction, ctx),
                    )
            except Exception as e:
                pg_ok = False
                if not pg_fail_printed:
                    print(f"\nPG zápis zlyhal ({e}) — ďalej len CSV.\n")
                    pg_fail_printed = True
                try:
                    pg_conn.close()
                except Exception:
                    pass
                pg_conn = None

        n_saved += 1

    def on_message(_ws, message: str) -> None:
        nonlocal cvd, imb, bid_vol, ask_vol, price, n_trades, n_books
        nonlocal last_print, cvd_prev_print, abs_active, prev_ctx
        msg = json.loads(message)
        stream = msg.get("stream", "")
        data = msg.get("data", msg)
        now = time.time()

        if "aggTrade" in stream or ("e" in data and data.get("e") == "aggTrade"):
            qty = float(data["q"])
            price = float(data["p"])
            # m=True → agresor sell
            side = "sell" if data["m"] else "buy"
            delta = -qty if data["m"] else qty
            cvd += delta
            n_trades += 1

            # Large Order Detection (na každom obchode, nie len 1s print)
            if len(qty_hist) >= LARGE_ORDER_ROLLING:
                avg_qty = sum(qty_hist) / len(qty_hist)
                threshold = LARGE_ORDER_MULT * avg_qty
                use_relative = True
            else:
                avg_qty = sum(qty_hist) / len(qty_hist) if qty_hist else 0.0
                threshold = LARGE_ORDER_QTY_FALLBACK
                use_relative = False

            if qty >= threshold and threshold > 0:
                ts_lo = datetime.now(timezone.utc)
                if use_relative and avg_qty > 0:
                    mult = qty / avg_qty
                    print(
                        f"*** LARGE ORDER {side.upper():4s} {qty:.5f} @ {price:.2f} | "
                        f"priemer posledných {LARGE_ORDER_ROLLING} = {avg_qty:.5f} BTC, "
                        f"{mult:.1f}x nad priemerom | CVD {cvd:+.4f} ***"
                    )
                else:
                    mult = (qty / avg_qty) if avg_qty > 0 else float("inf")
                    print(
                        f"*** LARGE ORDER {side.upper():4s} {qty:.5f} @ {price:.2f} | "
                        f"záložný prah {LARGE_ORDER_QTY_FALLBACK} BTC "
                        f"(história {len(qty_hist)}/{LARGE_ORDER_ROLLING}) | "
                        f"CVD {cvd:+.4f} ***"
                    )
                large_events.append(
                    {
                        "ts": ts_lo,
                        "side": side.upper(),
                        "qty": qty,
                        "price": price,
                        "avg": avg_qty,
                        "mult": mult if mult != float("inf") else 0.0,
                    }
                )
            qty_hist.append(qty)

        elif "depth" in stream or "bids" in data or "b" in data:
            bids = data.get("bids") or data.get("b") or []
            asks = data.get("asks") or data.get("a") or []
            bid_vol, ask_vol, imb = imbalance_from_book(bids, asks)
            n_books += 1
            # ak ešte nemáme cenu z trade, mid z top of book
            if price <= 0 and bids and asks:
                price = (float(bids[0][0]) + float(asks[0][0])) / 2.0

        if now - last_print < PRINT_EVERY_SEC:
            return
        if n_trades == 0 or n_books == 0 or price <= 0:
            return
        last_print = now

        hist.append((now, cvd, price))
        while hist and now - hist[0][0] > ABSORPTION_LOOKBACK_SEC + 2.0:
            hist.popleft()

        direction = cvd_smer(cvd, cvd_prev_print)
        ctx = kontext(direction, imb)

        abs_label = None
        d_cvd = d_px = None
        changes = window_changes(hist, now, ABSORPTION_LOOKBACK_SEC)
        if changes is not None:
            d_cvd, d_px, _lag = changes
            abs_label = detect_absorption(d_cvd, d_px)

        ts = datetime.now(timezone.utc)
        line = (
            f"CVD {cvd:+7.3f} ({direction:6s}) | "
            f"IMB {imb:+.3f} | px {price:.1f} | "
            f"→ {ctx}"
        )
        if abs_label and d_cvd is not None and d_px is not None:
            line += (
                f" | {abs_label} "
                f"[ΔCVD {d_cvd:+.2f} / Δpx {d_px:+.3f}% / {ABSORPTION_LOOKBACK_SEC:.0f}s]"
            )
        print(line)

        # BUY/SELL: len prechod do nového kontextu
        if ctx != prev_ctx:
            if ctx == "BUY kontext (silný)":
                ctx_events.append(
                    {"kind": "BUY", "ts": ts, "price": price, "ctx": ctx}
                )
            elif ctx == "SELL kontext (silný)":
                ctx_events.append(
                    {"kind": "SELL", "ts": ts, "price": price, "ctx": ctx}
                )
            prev_ctx = ctx

        # ALERT + záznam len pri vstupe do absorption
        if abs_label and not abs_active and d_cvd is not None and d_px is not None:
            print(f"  *** ALERT {abs_label} ***")
            abs_active = True
            kind = "bullish" if d_cvd > 0 else "bearish"
            abs_events.append(
                {
                    "kind": kind,
                    "ts": ts,
                    "price": price,
                    "d_cvd": d_cvd,
                    "d_px": d_px,
                }
            )
        elif not abs_label:
            abs_active = False

        save_row(ts, price, cvd, imb, direction, ctx)
        cvd_prev_print = cvd

    def on_error(_ws, error) -> None:
        print(f"WebSocket chyba: {error}")

    def on_close(_ws, code, _msg) -> None:
        print(f"\nZatvorené ({code}).")
        print_run_summary()

    def on_open(_ws) -> None:
        print(f"Pripojené: {url}\n")

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    try:
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    except KeyboardInterrupt:
        print("\nCtrl+C — koniec.")
        try:
            ws.close()
        except Exception:
            pass
    finally:
        print_run_summary()
        try:
            csv_f.flush()
            csv_f.close()
        except Exception:
            pass
        if pg_conn is not None:
            try:
                pg_conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

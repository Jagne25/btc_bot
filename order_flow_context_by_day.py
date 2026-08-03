"""
Denný rozpad BUY vs SELL kontextu (order_flow_context_log / CSV).

Pre každý kalendárny deň (dátum z ts_utc):
  - smer trhu: cena začiatok→koniec dňa (z dostupných riadkov)
  - BUY/SELL prechody + úspešnosť o 60 s

Cieľ: pochopiť, prečo BUY (~50–52%) zaostáva za SELL (~56–57%).

Spusti: python3 order_flow_context_by_day.py
Výsledky: len na SSD (.../BTC/order_flow_context/)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
FORWARD_SEC = 60
FORWARD_TOLERANCE_SEC = 30

BUY_CTX = "BUY kontext (silný)"
SELL_CTX = "SELL kontext (silný)"
SIGNAL_CTX = {BUY_CTX, SELL_CTX}

_BASE_SSD = Path(os.getenv("BASE_SSD_DIR", "/Volumes/WORK_SSD/TradingData/btc_bot"))
OUT_DIR = _BASE_SSD / SYMBOL.replace("USDT", "") / "order_flow_context"
CSV_DIR = OUT_DIR

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)


def ensure_ssd() -> Path:
    if not _BASE_SSD.exists():
        raise RuntimeError(f"SSD nie je pripojené: {_BASE_SSD}")
    if "/Volumes/WORK_SSD" not in str(_BASE_SSD.resolve()):
        raise RuntimeError(f"BASE_SSD_DIR nie je na WORK_SSD: {_BASE_SSD}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def load_from_pg() -> pd.DataFrame | None:
    try:
        import psycopg2
    except ImportError:
        print("PG: psycopg2 chýba — skúšam CSV.")
        return None
    try:
        conn = psycopg2.connect(**PGCFG)
        df = pd.read_sql(
            """
            SELECT ts_utc, symbol, price, cvd, imb, cvd_dir, kontext
            FROM order_flow_context_log
            WHERE symbol = %s
            ORDER BY ts_utc
            """,
            conn,
            params=(SYMBOL,),
        )
        conn.close()
        if df.empty:
            print("PG: tabuľka je prázdna.")
            return None
        print(f"PG: načítaných {len(df)} riadkov z order_flow_context_log")
        return df
    except Exception as e:
        print(f"PG: {e}")
        print("  Skúšam CSV na SSD…")
        return None


def load_from_csv() -> pd.DataFrame:
    files = sorted(CSV_DIR.glob("context_log_*.csv"))
    if not files:
        raise RuntimeError(f"Žiadne CSV v {CSV_DIR}")
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    print(f"CSV: načítaných {len(df)} riadkov z {len(files)} súborov")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts_utc"], utc=True)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price", "ts"]).reset_index(drop=True)
    out["day"] = out["ts"].dt.strftime("%Y-%m-%d")
    out["kontext_prev"] = out["kontext"].shift(1)
    # prechod naprieč celým datasetom (nie reset cez polnoc)
    out["is_transition"] = out["kontext"] != out["kontext_prev"]
    return out


def attach_forward_price(df: pd.DataFrame) -> pd.DataFrame:
    left = df.copy()
    left["ts_target"] = left["ts"] + pd.Timedelta(seconds=FORWARD_SEC)
    right = df[["ts", "price"]].rename(columns={"ts": "ts_fwd", "price": "price_fwd"})
    merged = pd.merge_asof(
        left.sort_values("ts_target"),
        right.sort_values("ts_fwd"),
        left_on="ts_target",
        right_on="ts_fwd",
        direction="forward",
    ).sort_values("ts").reset_index(drop=True)
    lag = (merged["ts_fwd"] - merged["ts"]).dt.total_seconds()
    ok = (
        merged["price_fwd"].notna()
        & lag.notna()
        & (lag >= FORWARD_SEC)
        & (lag <= FORWARD_SEC + FORWARD_TOLERANCE_SEC)
    )
    merged["fwd_ok"] = ok
    merged["price_fwd"] = np.where(ok, merged["price_fwd"], np.nan)
    return merged


def day_market_move(g: pd.DataFrame) -> dict:
    """Prvá/posledná cena v dostupných riadkoch toho dňa."""
    p0 = float(g["price"].iloc[0])
    p1 = float(g["price"].iloc[-1])
    pct = 100.0 * (p1 - p0) / p0 if p0 else float("nan")
    if pct > 0.01:
        smer = "UP"
    elif pct < -0.01:
        smer = "DOWN"
    else:
        smer = "FLAT"
    return {
        "price_open": p0,
        "price_close": p1,
        "day_pct": pct,
        "smer": smer,
        "n_rows": len(g),
        "t_first": g["ts"].iloc[0],
        "t_last": g["ts"].iloc[-1],
    }


def signal_stats(g: pd.DataFrame, ctx: str, up: bool) -> dict:
    """Prechody do ctx s platným forward; úspech = cena hore (BUY) / dole (SELL)."""
    sig = g[
        g["is_transition"] & (g["kontext"] == ctx) & g["fwd_ok"]
    ].copy()
    n = len(sig)
    if n == 0:
        return {"n": 0, "wins": 0, "rate": float("nan")}
    if up:
        wins = int((sig["price_fwd"] > sig["price"]).sum())
    else:
        wins = int((sig["price_fwd"] < sig["price"]).sum())
    return {"n": n, "wins": wins, "rate": 100.0 * wins / n}


def run() -> None:
    out_dir = ensure_ssd()
    raw = load_from_pg()
    if raw is None:
        raw = load_from_csv()

    df = prepare(raw)
    df = attach_forward_price(df)

    rows = []
    for day, g in df.groupby("day", sort=True):
        m = day_market_move(g)
        buy = signal_stats(g, BUY_CTX, up=True)
        sell = signal_stats(g, SELL_CTX, up=False)
        rows.append(
            {
                "day": day,
                "smer": m["smer"],
                "day_pct": m["day_pct"],
                "price_open": m["price_open"],
                "price_close": m["price_close"],
                "n_rows": m["n_rows"],
                "buy_n": buy["n"],
                "buy_wins": buy["wins"],
                "buy_rate": buy["rate"],
                "sell_n": sell["n"],
                "sell_wins": sell["wins"],
                "sell_rate": sell["rate"],
                "t_first": m["t_first"],
                "t_last": m["t_last"],
            }
        )

    day_df = pd.DataFrame(rows)

    # celkom
    buy_all = signal_stats(df, BUY_CTX, up=True)
    sell_all = signal_stats(df, SELL_CTX, up=False)

    lines = [
        "=== Order Flow CONTEXT — rozpad podľa dní ===",
        f"Symbol: {SYMBOL} | signál = PRECHOD | forward {FORWARD_SEC}s",
        f"Dni: {day_df['day'].iloc[0]} → {day_df['day'].iloc[-1]} | riadkov: {len(df)}",
        "",
        f"{'dátum':<12} {'smer':<6} {'trh %':>8} {'BUY n':>7} {'BUY %':>7} "
        f"{'SELL n':>7} {'SELL %':>7}",
        "-" * 62,
    ]
    for r in rows:
        buy_s = f"{r['buy_rate']:.1f}" if r["buy_n"] else "n/a"
        sell_s = f"{r['sell_rate']:.1f}" if r["sell_n"] else "n/a"
        lines.append(
            f"{r['day']:<12} {r['smer']:<6} {r['day_pct']:>+7.2f}% "
            f"{r['buy_n']:>7} {buy_s:>7} "
            f"{r['sell_n']:>7} {sell_s:>7}"
        )
    lines.append("-" * 62)
    lines.append(
        f"{'CELKOM':<12} {'':<6} {'':>8} "
        f"{buy_all['n']:>7} {buy_all['rate']:.1f} "
        f"{sell_all['n']:>7} {sell_all['rate']:.1f}"
    )
    lines.append("")
    lines.append("Poznámky:")
    lines.append("  • trh % = (posledná − prvá cena v dostupných riadkoch toho dňa) / prvá")
    lines.append("    (nie nutne 00:00–24:00 — len kým bežal logger)")
    lines.append("  • BUY úspech = cena o 60s vyššie; SELL = cena o 60s nižšie")
    lines.append("  • dátum = kalendárny deň z ts_utc (UTC)")

    # jednoduchá korelácia smer vs edge
    up_days = [r for r in rows if r["smer"] == "UP" and r["buy_n"] and r["sell_n"]]
    down_days = [r for r in rows if r["smer"] == "DOWN" and r["buy_n"] and r["sell_n"]]
    if up_days:
        lines.append("")
        lines.append(
            f"Na UP dňoch  (n={len(up_days)}): "
            f"BUY avg {np.mean([r['buy_rate'] for r in up_days]):.1f}% | "
            f"SELL avg {np.mean([r['sell_rate'] for r in up_days]):.1f}%"
        )
    if down_days:
        lines.append(
            f"Na DOWN dňoch (n={len(down_days)}): "
            f"BUY avg {np.mean([r['buy_rate'] for r in down_days]):.1f}% | "
            f"SELL avg {np.mean([r['sell_rate'] for r in down_days]):.1f}%"
        )

    text = "\n".join(lines)
    print(text)

    csv_path = out_dir / "context_by_day.csv"
    summary_path = out_dir / "context_by_day_summary.txt"
    day_df.to_csv(csv_path, index=False)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nUložené na SSD:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    run()

"""
Backtest na order_flow_context_log (BUY/SELL silný kontext).

Signál = len PRECHOD kontextu (prvý riadok, kde kontext ≠ predchádzajúci).
Pokračovanie toho istého kontextu (BUY, BUY, BUY…) sa nepočíta znova.

Na každom prechode do BUY/SELL (silný):
  - vezmi price teraz
  - nájdi price o ~60 s neskôr (prvý dostupný záznam ≥ 60 s)
  - BUY úspech = cena vyššie; SELL úspech = cena nižšie

Baseline: na každom riadku (s platným forward) „vždy long/short“.

Dáta: PostgreSQL order_flow_context_log (fallback CSV na SSD).
Výsledky: len na SSD.

Spusti: python3 order_flow_context_backtest.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
FORWARD_SEC = 60
# max. oneskorenie nad FORWARD_SEC (ak nie je presne 60 s záznam)
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
    # prechod = kontext sa zmenil oproti predchádzajúcemu riadku (1. riadok = prechod)
    out["kontext_prev"] = out["kontext"].shift(1)
    out["is_transition"] = out["kontext"] != out["kontext_prev"]
    return out


def attach_forward_price(df: pd.DataFrame) -> pd.DataFrame:
    """Ku každému riadku pripoj cenu prvého záznamu ≥ ts + FORWARD_SEC."""
    left = df.copy()
    left["ts_target"] = left["ts"] + pd.Timedelta(seconds=FORWARD_SEC)
    right = df[["ts", "price"]].rename(columns={"ts": "ts_fwd", "price": "price_fwd"})
    merged = pd.merge_asof(
        left.sort_values("ts_target"),
        right.sort_values("ts_fwd"),
        left_on="ts_target",
        right_on="ts_fwd",
        direction="forward",
    )
    # vráť pôvodné poradie podľa ts
    merged = merged.sort_values("ts").reset_index(drop=True)
    lag = (merged["ts_fwd"] - merged["ts"]).dt.total_seconds()
    ok = (
        merged["price_fwd"].notna()
        & lag.notna()
        & (lag >= FORWARD_SEC)
        & (lag <= FORWARD_SEC + FORWARD_TOLERANCE_SEC)
    )
    merged["fwd_ok"] = ok
    merged["fwd_lag_sec"] = lag
    return merged


def summarize(success: pd.Series) -> dict:
    n = len(success)
    if n == 0:
        return {"n": 0, "wins": 0, "rate": float("nan")}
    wins = int(success.sum())
    return {"n": n, "wins": wins, "rate": 100.0 * wins / n}


def run() -> None:
    out_dir = ensure_ssd()
    raw = load_from_pg()
    if raw is None:
        raw = load_from_csv()

    df = prepare(raw)
    df = attach_forward_price(df)
    usable = df[df["fwd_ok"]].copy()

    t0, t1 = df["ts"].iloc[0], df["ts"].iloc[-1]
    span_h = (t1 - t0).total_seconds() / 3600.0

    # --- signály: len prechody kontextu do BUY/SELL ---
    n_buy_raw = int((usable["kontext"] == BUY_CTX).sum())
    n_sell_raw = int((usable["kontext"] == SELL_CTX).sum())
    transitions = usable[
        usable["is_transition"] & usable["kontext"].isin(SIGNAL_CTX)
    ].copy()
    buy = transitions[transitions["kontext"] == BUY_CTX].copy()
    sell = transitions[transitions["kontext"] == SELL_CTX].copy()
    buy["success"] = buy["price_fwd"] > buy["price"]
    sell["success"] = sell["price_fwd"] < sell["price"]
    signals = pd.concat([buy, sell], ignore_index=True)
    signals["side"] = np.where(signals["kontext"] == BUY_CTX, "BUY", "SELL")

    sb = summarize(buy["success"])
    ss = summarize(sell["success"])
    st = summarize(signals["success"])

    # --- baseline: vždy long/short na každom riadku s platným forward ---
    baseline_ok = usable["price_fwd"] > usable["price"]
    bl = summarize(baseline_ok)
    baseline_short = usable["price_fwd"] < usable["price"]
    bl_short = summarize(baseline_short)

    edge_buy = sb["rate"] - bl["rate"] if sb["n"] and bl["n"] else float("nan")
    edge_sell_vs_short = ss["rate"] - bl_short["rate"] if ss["n"] and bl_short["n"] else float("nan")
    edge_all_vs_long = st["rate"] - bl["rate"] if st["n"] and bl["n"] else float("nan")

    lines = [
        "=== Order Flow CONTEXT backtest (len PRECHODY kontextu) ===",
        f"Symbol: {SYMBOL} | forward {FORWARD_SEC}s (±{FORWARD_TOLERANCE_SEC}s)",
        "Signál = prvý riadok, kde kontext ≠ predchádzajúci (nie každé pokračovanie).",
        f"Dáta: {t0.isoformat()} → {t1.isoformat()} (~{span_h:.1f} h)",
        f"Riadkov celkom: {len(df)} | s platným forward: {len(usable)}",
        f"Raw BUY/SELL riadky (pred dedup): BUY={n_buy_raw} | SELL={n_sell_raw}",
        f"Prechody (nezávislé signály):     BUY={sb['n']} | SELL={ss['n']} | ALL={st['n']}",
        "",
        f"--- BUY  ({BUY_CTX}) ---",
        f"Počet signálov: {sb['n']}",
        f"Úspešných:      {sb['wins']}  (cena o {FORWARD_SEC}s vyššie)",
        f"Úspešnosť:      {sb['rate']:.1f}%" if sb["n"] else "Úspešnosť:      n/a",
        "",
        f"--- SELL ({SELL_CTX}) ---",
        f"Počet signálov: {ss['n']}",
        f"Úspešných:      {ss['wins']}  (cena o {FORWARD_SEC}s nižšie)",
        f"Úspešnosť:      {ss['rate']:.1f}%" if ss["n"] else "Úspešnosť:      n/a",
        "",
        "--- CELKOM (BUY+SELL podľa smeru) ---",
        f"Počet signálov: {st['n']}",
        f"Úspešných:      {st['wins']}",
        f"Úspešnosť:      {st['rate']:.1f}%" if st["n"] else "Úspešnosť:      n/a",
        "",
        "=== Baseline (porovnanie) ===",
        f"Baseline LONG  (vždy tipuj cena↑ o {FORWARD_SEC}s): "
        f"n={bl['n']} | úspešnosť={bl['rate']:.1f}%",
        f"Baseline SHORT (vždy tipuj cena↓ o {FORWARD_SEC}s): "
        f"n={bl_short['n']} | úspešnosť={bl_short['rate']:.1f}%",
        "",
        "Edge (signál − baseline):",
        f"  BUY  − baseline LONG:  {edge_buy:+.1f} p.b." if sb["n"] else "  BUY: n/a",
        f"  SELL − baseline SHORT: {edge_sell_vs_short:+.1f} p.b." if ss["n"] else "  SELL: n/a",
        f"  ALL  − baseline LONG:  {edge_all_vs_long:+.1f} p.b.  (len orientačne)"
        if st["n"]
        else "  ALL: n/a",
    ]
    text = "\n".join(lines)
    print(text)

    csv_path = out_dir / "context_backtest_transitions_60s.csv"
    summary_path = out_dir / "context_backtest_transitions_60s_summary.txt"
    cols = [
        "ts",
        "price",
        "price_fwd",
        "fwd_lag_sec",
        "cvd",
        "imb",
        "cvd_dir",
        "kontext",
        "side",
        "success",
    ]
    signals[cols].to_csv(csv_path, index=False)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nUložené na SSD:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    run()

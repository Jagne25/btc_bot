"""
Order Flow — backtest divergencie (cena vs CVD).

- Kód: tento súbor v projekte (Mac OK)
- Dáta / výsledky: VÝHRADNE SSD
  /Volumes/WORK_SSD/TradingData/btc_bot/BTC/order_flow_backtest/

Spusti:
  python3 order_flow_backtest.py

MODE:
  "bullish_validate" — 90 dní, LEN silná bullish_div + rozpad po 30-dňových obdobiach
  "compare"          — holé vs silné (všetky druhy divergencie)

Voliteľne PostgreSQL (tabuľka order_flow_klines).
Ak PG nie je dostupná, backtest beží a výsledky idú len na SSD.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))
except ImportError:
    pass

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
# bullish_validate → 90 dní; compare → môžeš dať 30
MODE = "bullish_validate"
DAYS = 90 if MODE == "bullish_validate" else 30
LOOKBACK = DAYS * 24 * 60
DIVERGENCE_LOOKBACK = 20   # minút na detekciu divergencie
FORWARD_MINUTES = 10       # čo spravila cena o N minút neskôr

# Silná divergencia (filter)
MIN_PRICE_CHANGE_PCT = 0.3
# Fixné 43.8 z 30d štúdie (porovnateľné); None = dopočítať priemer z aktuálnych dát
MIN_CVD_CHANGE = 43.8

_BASE_SSD = Path(os.getenv("BASE_SSD_DIR", "/Volumes/WORK_SSD/TradingData/btc_bot"))
OUT_DIR = _BASE_SSD / SYMBOL.replace("USDT", "") / "order_flow_backtest"

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def parquet_path(out_dir: Path) -> Path:
    return out_dir / f"klines_{SYMBOL}_{INTERVAL}_{DAYS}d.parquet"


def get_klines(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    """Binance klines s pagináciou dozadu (verejné API)."""
    url = "https://api.binance.com/api/v3/klines"
    rows, end_time = [], None
    while len(rows) < lookback:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        end_time = chunk[0][0] - 1
        time.sleep(0.05)
        if len(rows) >= lookback:
            break
        print(f"  stiahnuté {min(len(rows), lookback)}/{lookback} sviečok…", flush=True)

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]
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


def add_cvd(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["taker_sell"] = out["volume"] - out["taker_buy_base"]
    out["delta"] = out["taker_buy_base"] - out["taker_sell"]
    out["cvd"] = out["delta"].cumsum()
    return out


def load_or_fetch_klines(out_dir: Path) -> pd.DataFrame:
    """Preferuj parquet na SSD (rýchle); inak stiahni z Binance."""
    pq = parquet_path(out_dir)
    if pq.exists():
        print(f"1) Načítavam klines z SSD: {pq}")
        df = pd.read_parquet(pq)
        if "cvd" not in df.columns:
            df = add_cvd(df)
        print(f"   Hotovo: {len(df)} sviečok | {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")
        return df

    print(f"1) Sťahujem {LOOKBACK}× {INTERVAL} klines ({DAYS} dní)…")
    raw = get_klines(SYMBOL, INTERVAL, LOOKBACK)
    if raw.empty:
        return raw
    df = add_cvd(raw)
    print(f"   Hotovo: {len(df)} sviečok | {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")
    path = save_klines_ssd(df, out_dir)
    print(f"   Uložené: {path}")
    return df


def try_pg_save(df: pd.DataFrame, symbol: str, interval: str) -> str:
    """Uloží klines+CVD do PostgreSQL. Vracia status text."""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        return "PG: psycopg2 nie je nainštalované — skip"

    try:
        conn = psycopg2.connect(**PGCFG)
    except Exception as e:
        return (
            f"PG: nedostupné ({PGCFG['host']}:{PGCFG['port']}) — {e}\n"
            "  Poznámka: najprv zapni SSD image: .../postgres/zapni_postgres.sh"
        )

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS order_flow_klines (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open_time TIMESTAMPTZ NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    taker_buy_base DOUBLE PRECISION,
                    delta DOUBLE PRECISION,
                    cvd DOUBLE PRECISION,
                    PRIMARY KEY (symbol, interval, open_time)
                );
                """
            )
            rows = [
                (
                    symbol,
                    interval,
                    row.open_time.to_pydatetime() if hasattr(row.open_time, "to_pydatetime") else row.open_time,
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.volume),
                    float(row.taker_buy_base),
                    float(row.delta),
                    float(row.cvd),
                )
                for row in df.itertuples(index=False)
            ]
            execute_values(
                cur,
                """
                INSERT INTO order_flow_klines (
                    symbol, interval, open_time, open, high, low, close,
                    volume, taker_buy_base, delta, cvd
                ) VALUES %s
                ON CONFLICT (symbol, interval, open_time) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    taker_buy_base = EXCLUDED.taker_buy_base,
                    delta = EXCLUDED.delta,
                    cvd = EXCLUDED.cvd
                """,
                rows,
                page_size=1000,
            )
        return f"PG: uložené {len(df)} riadkov do order_flow_klines ({PGCFG['host']})"
    finally:
        conn.close()


def save_klines_ssd(df: pd.DataFrame, out_dir: Path) -> Path:
    path = parquet_path(out_dir)
    df.to_parquet(path, index=False)
    return path


def cvd_change_stats(df: pd.DataFrame, lookback: int = DIVERGENCE_LOOKBACK) -> dict:
    """Štatistiky |ΔCVD| a |Δcena %| za lookback minút — na nastavenie prahov."""
    shift = lookback - 1
    price_pct = (df["close"] / df["close"].shift(shift) - 1.0).abs() * 100
    cvd_abs = (df["cvd"] - df["cvd"].shift(shift)).abs()
    price_pct = price_pct.dropna()
    cvd_abs = cvd_abs.dropna()
    return {
        "price_mean": float(price_pct.mean()),
        "price_p75": float(price_pct.quantile(0.75)),
        "price_p90": float(price_pct.quantile(0.90)),
        "cvd_mean": float(cvd_abs.mean()),
        "cvd_median": float(cvd_abs.median()),
        "cvd_p75": float(cvd_abs.quantile(0.75)),
        "cvd_p90": float(cvd_abs.quantile(0.90)),
    }


def backtest_divergence(
    df: pd.DataFrame,
    lookback: int = DIVERGENCE_LOOKBACK,
    forward: int = FORWARD_MINUTES,
    min_price_change_pct: float = 0.0,
    min_cvd_change: float = 0.0,
    kinds: set[str] | None = None,
) -> pd.DataFrame:
    """
    Pre každý bar i:
    - ak cena a CVD idú opačne za `lookback` minút → kandidát
    - voliteľne: |Δcena|% >= min_price_change_pct a |ΔCVD| >= min_cvd_change
    - kinds: None = obe; alebo {"bullish_div"} / {"bearish_div"}
    - úspech: o `forward` minút cena šla v očakávanom smere
    """
    closes = df["close"].to_numpy(dtype=float)
    cvds = df["cvd"].to_numpy(dtype=float)
    times = df["open_time"].to_numpy()
    n = len(df)
    records = []
    allow = kinds  # None = všetky

    for i in range(lookback - 1, n - forward):
        i0 = i - (lookback - 1)
        p0, p1 = closes[i0], closes[i]
        c0, c1 = cvds[i0], cvds[i]

        if p0 == 0:
            continue
        price_chg_pct = abs(p1 / p0 - 1.0) * 100.0
        cvd_chg = abs(c1 - c0)

        if price_chg_pct < min_price_change_pct:
            continue
        if cvd_chg < min_cvd_change:
            continue

        price_up = p1 > p0
        price_down = p1 < p0
        cvd_up = c1 > c0
        cvd_down = c1 < c0

        if price_up and cvd_down:
            kind = "bearish_div"
            expect_up = False
        elif price_down and cvd_up:
            kind = "bullish_div"
            expect_up = True
        else:
            continue

        if allow is not None and kind not in allow:
            continue

        p_fwd = closes[i + forward]
        success = (p_fwd > p1) if expect_up else (p_fwd < p1)

        records.append(
            {
                "signal_time": times[i],
                "kind": kind,
                "price_then": float(p1),
                "price_fwd": float(p_fwd),
                "price_chg_pct": float(price_chg_pct),
                "cvd_chg": float(cvd_chg),
                "cvd_then": float(c1),
                "success": bool(success),
            }
        )

    return pd.DataFrame(records)


def assign_periods(signals: pd.DataFrame, df: pd.DataFrame, n_periods: int = 3) -> pd.DataFrame:
    """Rozdelí signály do n rovnakých časových okien podľa rozsahu klines (napr. 3× 30 dní)."""
    if signals.empty:
        return signals
    out = signals.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True)
    t0 = pd.to_datetime(df["open_time"].iloc[0], utc=True)
    t1 = pd.to_datetime(df["open_time"].iloc[-1], utc=True)
    edges = pd.date_range(t0, t1, periods=n_periods + 1)
    labels = []
    for i in range(n_periods):
        a, b = edges[i], edges[i + 1]
        labels.append(f"P{i + 1}: {a.date()} → {b.date()}")

    # interval [left, right) okrem posledného [left, right]
    cats = pd.cut(
        out["signal_time"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    # posledný bod / signály na konci → posledný bucket
    out["period"] = cats.astype(object)
    out.loc[out["period"].isna(), "period"] = labels[-1]
    return out


def format_period_breakdown(signals: pd.DataFrame) -> str:
    lines = ["Rozpad podľa obdobia:"]
    if signals.empty or "period" not in signals.columns:
        lines.append("  (žiadne dáta)")
        return "\n".join(lines)
    for period, g in signals.groupby("period", sort=False):
        s = summarize(g)
        lines.append(f"  {period}")
        lines.append(f"    n={s['n']} | win={s['wins']} | úspešnosť={s['rate']:.1f}%")
    return "\n".join(lines)


def summarize(signals: pd.DataFrame) -> dict:
    if signals.empty:
        return {"n": 0, "wins": 0, "rate": float("nan")}
    n = len(signals)
    wins = int(signals["success"].sum())
    return {"n": n, "wins": wins, "rate": 100.0 * wins / n}


def format_block(name: str, signals: pd.DataFrame, extras: str = "") -> str:
    lines = [f"--- {name} ---"]
    if extras:
        lines.append(extras)
    s = summarize(signals)
    if s["n"] == 0:
        lines.append("Žiadne signály.")
        return "\n".join(lines)
    lines.append(f"Počet signálov: {s['n']}")
    lines.append(f"Úspešných:      {s['wins']}")
    lines.append(f"Úspešnosť:      {s['rate']:.1f}%")
    for kind, g in signals.groupby("kind"):
        k_n = len(g)
        k_w = int(g["success"].sum())
        label = "cena↑ CVD↓ → očakávame dole" if kind == "bearish_div" else "cena↓ CVD↑ → očakávame hore"
        lines.append(f"  {kind}: n={k_n} | win={k_w} | {100.0 * k_w / k_n:.1f}%  ({label})")
    return "\n".join(lines)


def print_comparison(bare: pd.DataFrame, strong: pd.DataFrame, min_pct: float, min_cvd: float, stats: dict) -> str:
    lines = []
    lines.append("=== Order Flow Backtest — porovnanie ===")
    lines.append(f"Symbol: {SYMBOL} | {INTERVAL} | {DAYS} dní | okno {DIVERGENCE_LOOKBACK} min | forward {FORWARD_MINUTES} min")
    lines.append("")
    lines.append("Štatistiky |Δ| za 20 min (celé dáta):")
    lines.append(
        f"  cena %:  mean={stats['price_mean']:.3f} | p75={stats['price_p75']:.3f} | p90={stats['price_p90']:.3f}"
    )
    lines.append(
        f"  CVD BTC: mean={stats['cvd_mean']:.1f} | median={stats['cvd_median']:.1f} | "
        f"p75={stats['cvd_p75']:.1f} | p90={stats['cvd_p90']:.1f}"
    )
    lines.append("")
    lines.append(format_block("A) HOLÉ (bez filtra)", bare, "filter: žiadny"))
    lines.append("")
    lines.append(
        format_block(
            "B) SILNÉ (filter)",
            strong,
            f"filter: min_price_change_pct={min_pct}% | min_cvd_change={min_cvd:.1f} BTC (≥ priemer)",
        )
    )
    lines.append("")
    sb, ss = summarize(bare), summarize(strong)
    if sb["n"] and ss["n"] and not np.isnan(sb["rate"]) and not np.isnan(ss["rate"]):
        diff = ss["rate"] - sb["rate"]
        lines.append(f"Rozdiel úspešnosti (silné − holé): {diff:+.1f} p.b.")
        lines.append(f"Signálov menej o: {sb['n'] - ss['n']} ({100.0 * ss['n'] / sb['n']:.1f}% zo holých zostalo)")
    return "\n".join(lines)


def run_bullish_validate(df: pd.DataFrame, out_dir: Path) -> None:
    """90d (alebo DAYS): len silná bullish_div + rozpad na 3 obdobia."""
    min_pct = MIN_PRICE_CHANGE_PCT
    stats = cvd_change_stats(df, DIVERGENCE_LOOKBACK)
    min_cvd = MIN_CVD_CHANGE if MIN_CVD_CHANGE is not None else stats["cvd_mean"]

    print(f"\nFilter: min_price_change_pct={min_pct}% | min_cvd_change={min_cvd:.1f} BTC")
    print(f"(referencia |ΔCVD| mean v týchto dátach ≈ {stats['cvd_mean']:.1f} BTC)")
    print("Len: bullish_div (cena↓ CVD↑ → očakávame rast o 10 min)\n")

    signals = backtest_divergence(
        df,
        min_price_change_pct=min_pct,
        min_cvd_change=min_cvd,
        kinds={"bullish_div"},
    )
    signals = assign_periods(signals, df, n_periods=3)
    s = summarize(signals)

    lines = [
        "=== Order Flow Backtest — BULLISH silná divergencia ===",
        f"Symbol: {SYMBOL} | {INTERVAL} | {DAYS} dní | okno {DIVERGENCE_LOOKBACK} min | forward {FORWARD_MINUTES} min",
        f"Filter: min_price_change_pct={min_pct}% | min_cvd_change={min_cvd:.1f} BTC",
        "Kind: LEN bullish_div",
        "",
        f"CELKOM: n={s['n']} | win={s['wins']} | úspešnosť={s['rate']:.1f}%",
        "",
        format_period_breakdown(signals),
    ]
    text = "\n".join(lines)
    print(text)

    csv_path = out_dir / f"backtest_bullish_strong_{DAYS}d.csv"
    summary_path = out_dir / f"backtest_bullish_strong_{DAYS}d_summary.txt"
    signals.to_csv(csv_path, index=False)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nUložené na SSD:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


def run_compare(df: pd.DataFrame, out_dir: Path) -> None:
    stats = cvd_change_stats(df, DIVERGENCE_LOOKBACK)
    min_cvd = MIN_CVD_CHANGE if MIN_CVD_CHANGE is not None else stats["cvd_mean"]
    min_pct = MIN_PRICE_CHANGE_PCT
    print(f"   Priemerná |ΔCVD| ≈ {stats['cvd_mean']:.1f} BTC → min_cvd_change = {min_cvd:.1f}")
    print(f"   min_price_change_pct = {min_pct}%")

    bare = backtest_divergence(df, min_price_change_pct=0.0, min_cvd_change=0.0)
    strong = backtest_divergence(df, min_price_change_pct=min_pct, min_cvd_change=min_cvd)
    text = print_comparison(bare, strong, min_pct, min_cvd, stats)
    print()
    print(text)
    bare.to_csv(out_dir / "backtest_signals_bare.csv", index=False)
    strong.to_csv(out_dir / "backtest_signals_strong.csv", index=False)
    (out_dir / "backtest_summary.txt").write_text(text + "\n", encoding="utf-8")


def main():
    out_dir = ensure_ssd()
    print(f"SSD výstup: {out_dir}")
    print(f"MODE={MODE} | DAYS={DAYS} | LOOKBACK={LOOKBACK}\n")

    df = load_or_fetch_klines(out_dir)
    if df.empty:
        print("Žiadne klines.")
        return

    print("\n2) PostgreSQL (voliteľné)…")
    print(f"   {try_pg_save(df, SYMBOL, INTERVAL)}")

    if MODE == "bullish_validate":
        print("\n3) Backtest bullish silná divergencia (90d check)…")
        run_bullish_validate(df, out_dir)
    else:
        print("\n3) Backtest A) holé + B) silné…")
        run_compare(df, out_dir)


if __name__ == "__main__":
    main()

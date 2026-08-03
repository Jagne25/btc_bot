"""
Backtest Absorption udalostí z order_flow_context_log / CSV.

Absorption sa do SQL neukladá ako stĺpec — zrekonštruuje sa rovnakými
prahmi ako live (order_flow_context.py). Udalosť = VSTUP do absorption
(edge), nie každý riadok, kým absorption trvá (= ALERT moment).

Pre každú udalosť:
  - cena teraz
  - cena o 60 s a o 5 min
  - porovnanie |Δprice| a smerovej Δprice s baseline (všetky platné riadky)

Spusti: python3 order_flow_absorption_backtest.py
Výsledky: len na SSD (.../BTC/order_flow_context/)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"

# rovnaké prahy ako order_flow_context.py
ABSORPTION_LOOKBACK_SEC = 20.0
MIN_CVD_CHANGE_ABS = 4.9
MAX_PRICE_CHANGE_PCT = 0.05

FORWARD_HORIZONS = (60, 300)  # 60s, 5 min
FORWARD_TOLERANCE_SEC = 30

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
    out["cvd"] = pd.to_numeric(out["cvd"], errors="coerce")
    out = out.dropna(subset=["price", "cvd", "ts"]).reset_index(drop=True)
    return out


def attach_lookback(df: pd.DataFrame) -> pd.DataFrame:
    """ΔCVD a Δprice% za ABSORPTION_LOOKBACK_SEC dozadu."""
    left = df.copy()
    left["ts_target"] = left["ts"] - pd.Timedelta(seconds=ABSORPTION_LOOKBACK_SEC)
    right = df[["ts", "price", "cvd"]].rename(
        columns={"ts": "ts_past", "price": "price_past", "cvd": "cvd_past"}
    )
    m = pd.merge_asof(
        left.sort_values("ts_target"),
        right.sort_values("ts_past"),
        left_on="ts_target",
        right_on="ts_past",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=3),
    ).sort_values("ts").reset_index(drop=True)

    lag = (m["ts"] - m["ts_past"]).dt.total_seconds()
    ok = (
        m["ts_past"].notna()
        & lag.notna()
        & (lag >= ABSORPTION_LOOKBACK_SEC * 0.75)
        & (lag <= ABSORPTION_LOOKBACK_SEC + 3.0)
    )
    m["lb_ok"] = ok
    m["d_cvd"] = m["cvd"] - m["cvd_past"]
    m["d_price_pct_lb"] = 100.0 * (m["price"] - m["price_past"]) / m["price_past"]
    m["is_abs"] = (
        ok
        & (m["d_cvd"].abs() >= MIN_CVD_CHANGE_ABS)
        & (m["d_price_pct_lb"].abs() < MAX_PRICE_CHANGE_PCT)
    )
    m["abs_kind"] = np.where(
        ~m["is_abs"],
        None,
        np.where(m["d_cvd"] > 0, "bullish", "bearish"),
    )
    # edge = vstup do absorption (ALERT)
    prev = m["is_abs"].shift(1, fill_value=False)
    m["abs_entry"] = m["is_abs"] & ~prev
    return m


def attach_forward(df: pd.DataFrame, horizon_sec: int) -> pd.DataFrame:
    left = df.copy()
    left["ts_target"] = left["ts"] + pd.Timedelta(seconds=horizon_sec)
    right = df[["ts", "price"]].rename(columns={"ts": "ts_fwd", "price": "price_fwd"})
    m = pd.merge_asof(
        left.sort_values("ts_target"),
        right.sort_values("ts_fwd"),
        left_on="ts_target",
        right_on="ts_fwd",
        direction="forward",
    ).sort_values("ts").reset_index(drop=True)
    lag = (m["ts_fwd"] - m["ts"]).dt.total_seconds()
    ok = (
        m["price_fwd"].notna()
        & lag.notna()
        & (lag >= horizon_sec)
        & (lag <= horizon_sec + FORWARD_TOLERANCE_SEC)
    )
    col = f"fwd_{horizon_sec}"
    m[f"{col}_ok"] = ok
    m[f"{col}_price"] = m["price_fwd"]
    m[f"{col}_lag"] = lag
    m[f"{col}_ret_pct"] = np.where(
        ok, 100.0 * (m["price_fwd"] - m["price"]) / m["price"], np.nan
    )
    return m.drop(columns=["ts_target", "ts_fwd", "price_fwd"], errors="ignore")


def stats_block(name: str, rets: pd.Series) -> dict:
    r = rets.dropna()
    n = len(r)
    if n == 0:
        return {
            "name": name,
            "n": 0,
            "mean": float("nan"),
            "abs_mean": float("nan"),
            "median": float("nan"),
            "pct_up": float("nan"),
            "pct_down": float("nan"),
        }
    return {
        "name": name,
        "n": n,
        "mean": float(r.mean()),
        "abs_mean": float(r.abs().mean()),
        "median": float(r.median()),
        "pct_up": float(100.0 * (r > 0).mean()),
        "pct_down": float(100.0 * (r < 0).mean()),
    }


def fmt_stats(s: dict, baseline_abs: float | None = None) -> list[str]:
    if s["n"] == 0:
        return [f"  {s['name']}: n=0"]
    lines = [
        f"  {s['name']}: n={s['n']}",
        f"    mean Δprice:     {s['mean']:+.4f}%",
        f"    mean |Δprice|:   {s['abs_mean']:.4f}%",
        f"    median Δprice:   {s['median']:+.4f}%",
        f"    podiel cena↑:    {s['pct_up']:.1f}%",
        f"    podiel cena↓:    {s['pct_down']:.1f}%",
    ]
    if baseline_abs is not None and not np.isnan(baseline_abs) and baseline_abs > 0:
        ratio = s["abs_mean"] / baseline_abs
        lines.append(
            f"    vs baseline |Δ|: {ratio:.2f}× "
            f"({'väčší pohyb' if ratio > 1 else 'menší/rovnaký pohyb'})"
        )
    return lines


def run() -> None:
    out_dir = ensure_ssd()
    raw = load_from_pg()
    if raw is None:
        raw = load_from_csv()

    df = prepare(raw)
    df = attach_lookback(df)
    for h in FORWARD_HORIZONS:
        df = attach_forward(df, h)

    t0, t1 = df["ts"].iloc[0], df["ts"].iloc[-1]
    span_h = (t1 - t0).total_seconds() / 3600.0

    entries = df[df["abs_entry"]].copy()
    n_rows_abs = int(df["is_abs"].sum())
    n_entry = len(entries)
    n_bull = int((entries["abs_kind"] == "bullish").sum())
    n_bear = int((entries["abs_kind"] == "bearish").sum())

    lines = [
        "=== Order Flow ABSORPTION backtest ===",
        f"Symbol: {SYMBOL}",
        f"Detekcia: lookback {ABSORPTION_LOOKBACK_SEC:.0f}s | "
        f"|ΔCVD|≥{MIN_CVD_CHANGE_ABS} | |Δprice|<{MAX_PRICE_CHANGE_PCT}%",
        "Udalosť = VSTUP do absorption (edge / ALERT), nie každý riadok.",
        f"Dáta: {t0.isoformat()} → {t1.isoformat()} (~{span_h:.1f} h)",
        f"Riadkov: {len(df)} | riadkov v absorption: {n_rows_abs} | "
        f"ALERT vstupov: {n_entry} (bullish={n_bull}, bearish={n_bear})",
        "",
        "Poznámka: absorption nie je stĺpec v SQL — zrekonštruované z price+CVD.",
        "",
    ]

    event_rows = []
    for h in FORWARD_HORIZONS:
        col = f"fwd_{h}_ret_pct"
        ok_col = f"fwd_{h}_ok"
        label = f"{h}s" if h < 60 else (f"{h // 60} min" if h % 60 == 0 else f"{h}s")

        base = df.loc[df[ok_col], col]
        bl = stats_block("Baseline (všetky riadky)", base)

        all_e = entries.loc[entries[ok_col], col]
        bull_e = entries.loc[(entries["abs_kind"] == "bullish") & entries[ok_col], col]
        bear_e = entries.loc[(entries["abs_kind"] == "bearish") & entries[ok_col], col]

        s_all = stats_block("ALL absorption vstupy", all_e)
        s_bull = stats_block("bullish (CVD↑, cena držala)", bull_e)
        s_bear = stats_block("bearish (CVD↓, cena držala)", bear_e)

        lines.append(f"--- Horizont +{label} ---")
        lines.append(
            f"Baseline: n={bl['n']} | mean Δ={bl['mean']:+.4f}% | "
            f"mean |Δ|={bl['abs_mean']:.4f}% | ↑{bl['pct_up']:.1f}% ↓{bl['pct_down']:.1f}%"
        )
        lines.extend(fmt_stats(s_all, bl["abs_mean"]))
        lines.extend(fmt_stats(s_bull, bl["abs_mean"]))
        lines.extend(fmt_stats(s_bear, bl["abs_mean"]))

        # smerová interpretácia
        if s_bull["n"]:
            lines.append(
                f"  → po bullish absorption: mean Δ {s_bull['mean']:+.4f}% "
                f"(očakávanie: absorbovaný nákupný tlak → často neskôr ↓ alebo flat)"
            )
        if s_bear["n"]:
            lines.append(
                f"  → po bearish absorption: mean Δ {s_bear['mean']:+.4f}% "
                f"(očakávanie: absorbovaný predajný tlak → často neskôr ↑ alebo flat)"
            )
        lines.append("")

        for _, row in entries.iterrows():
            if not row[ok_col]:
                continue
            event_rows.append(
                {
                    "ts": row["ts"],
                    "kind": row["abs_kind"],
                    "price": row["price"],
                    "d_cvd": row["d_cvd"],
                    "d_price_pct_lb": row["d_price_pct_lb"],
                    "horizon_sec": h,
                    "price_fwd": row[f"fwd_{h}_price"],
                    "ret_pct": row[col],
                    "fwd_lag_sec": row[f"fwd_{h}_lag"],
                }
            )

    text = "\n".join(lines)
    print(text)

    csv_path = out_dir / "absorption_backtest_events.csv"
    summary_path = out_dir / "absorption_backtest_summary.txt"
    pd.DataFrame(event_rows).to_csv(csv_path, index=False)
    summary_path.write_text(text + "\n", encoding="utf-8")
    print("Uložené na SSD:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    run()

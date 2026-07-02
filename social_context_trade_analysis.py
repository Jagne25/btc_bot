# social_context_trade_analysis.py
# Social context analysis + EDGE score
# - Reads observe CSVs (from backtest_sltp.py logs/observe_*_4h_*.csv)
# - Joins social_regime from DB (public.social_regime_3h)
# - Computes forward return, MFE, MAE over horizon bars
# - Produces per-regime summary + EDGE score
#
# Safe: does NOT modify your backtest/bot code.

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =========================
# CONFIG
# =========================
load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))
_BASE_SSD = Path(os.getenv("BASE_SSD_DIR", "/Volumes/WORK_SSD/TradingData/btc_bot"))
_COIN_MAP = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "BNBUSDT": "BNB", "SOLUSDT": "SOL"}

def _logs_dir(symbol: str) -> Path:
    coin = _COIN_MAP.get(symbol, symbol.replace("USDT", ""))
    return _BASE_SSD / coin / "logs"

INTERVAL = "4h"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
MODELS = ["LR", "RF"]  # which observe files to process

# Horizon: how many bars forward to evaluate outcome
HORIZON_BARS = 6   # 6 bars on 4h = 24h

# Asof join settings (for regime overlay)
ASOF_DIRECTION = "backward"
ASOF_TOLERANCE = pd.Timedelta("12h")  # <-- changed from 4h/6h; helps matching buckets

# Cut-off: remove signals before first non-null regime appears (reduces NULL share)
CUT_TO_FIRST_NON_NULL_REGIME = True

# DB tables/views
REGIME_TABLE = "public.social_regime_3h"

# EDGE weights (simple, stable)
EDGE_W_FWD = 0.50
EDGE_W_MFE = 0.35
EDGE_W_MAE = 0.35  # subtract abs(MAE)

# =========================
# Helpers
# =========================

def ensure_utc(s: pd.Series) -> pd.Series:
    """Return tz-aware UTC timestamps."""
    return pd.to_datetime(s, utc=True, errors="coerce")


def safe_float_series(x: pd.Series) -> pd.Series:
    """Convert to float, coerce invalid to NaN."""
    return pd.to_numeric(x, errors="coerce").astype(float)


def load_observe_csv(symbol: str, model: str) -> Optional[pd.DataFrame]:
    """
    Expected observe columns (minimum):
      close_time, open, high, low, close, proba_up, social_count_total, pos
    """
    csv_path = _logs_dir(symbol) / f"observe_{symbol}_{INTERVAL}_{model}.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if "close_time" not in df.columns:
        return None

    # Normalize expected columns
    needed = ["close_time", "open", "high", "low", "close", "pos"]
    for c in needed:
        if c not in df.columns:
            return None

    # Some files may have social_count_total
    if "social_count_total" not in df.columns:
        df["social_count_total"] = np.nan

    if "proba_up" not in df.columns:
        df["proba_up"] = np.nan

    df["close_time_utc"] = ensure_utc(df["close_time"])
    df = df.sort_values("close_time_utc").reset_index(drop=True)

    # Types
    for c in ["open", "high", "low", "close", "proba_up", "social_count_total"]:
        df[c] = safe_float_series(df[c])

    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").fillna(0).astype(int)

    return df


def db_load_regimes(engine, symbol: str, t_min: pd.Timestamp, t_max: pd.Timestamp) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """
    Load regimes for symbol in [t_min, t_max], return df_soc and first_valid_regime_time.
    first_valid_regime_time = first bucket where social_regime is NOT NULL.
    """
    sql = text(f"""
        SELECT
            bucket_start_utc,
            social_regime,
            z_count,
            z_sent
        FROM {REGIME_TABLE}
        WHERE symbol = :symbol
          AND bucket_start_utc BETWEEN :t_min AND :t_max
        ORDER BY bucket_start_utc ASC
    """)

    df = pd.read_sql(sql, engine, params={"symbol": symbol, "t_min": t_min, "t_max": t_max})
    if df.empty:
        return df, None

    df["bucket_start_utc"] = ensure_utc(df["bucket_start_utc"])
    if "social_regime" not in df.columns:
        df["social_regime"] = None

    # Normalize regime strings
    df["social_regime"] = df["social_regime"].apply(lambda x: str(x).upper().strip() if pd.notna(x) else None)

    # First non-null regime time
    first_valid = df.loc[df["social_regime"].notna(), "bucket_start_utc"].min()
    first_valid = pd.to_datetime(first_valid, utc=True) if pd.notna(first_valid) else None

    # Ensure numeric
    if "z_count" in df.columns:
        df["z_count"] = safe_float_series(df["z_count"])
    else:
        df["z_count"] = np.nan

    if "z_sent" in df.columns:
        df["z_sent"] = safe_float_series(df["z_sent"])
    else:
        df["z_sent"] = np.nan

    return df, first_valid


def apply_cutoff(df_sig: pd.DataFrame, cutoff: Optional[pd.Timestamp], symbol: str, model: str) -> pd.DataFrame:
    if not CUT_TO_FIRST_NON_NULL_REGIME or cutoff is None:
        return df_sig

    before = len(df_sig)
    df_sig = df_sig[df_sig["close_time_utc"] >= cutoff].copy()
    after = len(df_sig)
    print(f"[CUT-OFF] {symbol} {INTERVAL} {model}: {after}/{before} signálov ostalo (cut {before-after}), cutoff={cutoff}")
    return df_sig


def asof_join_regime(df_sig: pd.DataFrame, df_soc: pd.DataFrame) -> pd.DataFrame:
    """
    Join nearest past bucket within tolerance.
    """
    if df_soc.empty:
        df_sig["bucket_start_utc"] = pd.NaT
        df_sig["social_regime"] = None
        df_sig["z_count"] = np.nan
        df_sig["z_sent"] = np.nan
        df_sig["null_reason"] = "NO_REGIME_TABLE_ROWS"
        return df_sig

    left = df_sig.sort_values("close_time_utc").copy()
    right = df_soc.sort_values("bucket_start_utc").copy()

    out = pd.merge_asof(
        left,
        right,
        left_on="close_time_utc",
        right_on="bucket_start_utc",
        direction=ASOF_DIRECTION,
        tolerance=ASOF_TOLERANCE,
    )

    out["null_reason"] = None
    out.loc[out["bucket_start_utc"].isna(), "null_reason"] = "NO_BUCKET_MATCH"
    out.loc[out["bucket_start_utc"].notna() & out["social_regime"].isna(), "null_reason"] = "REGIME_IS_NULL"
    out.loc[out["bucket_start_utc"].notna() & out["social_regime"].notna(), "null_reason"] = "OK"
    return out


def compute_forward_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward return %, MFE %, MAE % over HORIZON_BARS after the signal bar.
    Uses OHLC from observe.
    Assumes each row is a bar. We compute metrics even when pos==0 (but summary will filter if needed).
    """
    df = df.copy()
    close = df["close"].astype(float)
    entry = close

    # forward close at horizon
    fwd_close = close.shift(-HORIZON_BARS)
    df["fwd_ret_pct"] = (fwd_close / entry - 1.0) * 100.0

    # MFE/MAE: within next HORIZON_BARS bars (inclusive next bars)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)

    # For each t, look at window t+1 .. t+H
    mfe_vals = []
    mae_vals = []
    n = len(df)

    for i in range(n):
        j0 = i + 1
        j1 = min(i + HORIZON_BARS, n - 1)
        if j0 > j1:
            mfe_vals.append(np.nan)
            mae_vals.append(np.nan)
            continue

        entry_px = float(entry.iloc[i])
        win_high = float(np.nanmax(highs.iloc[j0:j1+1].values))
        win_low = float(np.nanmin(lows.iloc[j0:j1+1].values))

        mfe = (win_high / entry_px - 1.0) * 100.0
        mae = (win_low / entry_px - 1.0) * 100.0  # negative number typically

        mfe_vals.append(mfe)
        mae_vals.append(mae)

    df["mfe_pct"] = mfe_vals
    df["mae_pct"] = mae_vals

    return df


def compute_edge_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDGE = w_fwd * fwd + w_mfe * mfe - w_mae * abs(mae)
    """
    df = df.copy()

    fwd = safe_float_series(df["fwd_ret_pct"])
    mfe = safe_float_series(df["mfe_pct"])
    mae = safe_float_series(df["mae_pct"])

    df["edge_score"] = (
        EDGE_W_FWD * fwd +
        EDGE_W_MFE * mfe -
        EDGE_W_MAE * np.abs(mae)
    )

    return df


def summarize_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table by social_regime.
    """
    x = df.copy()
    x["social_regime"] = x["social_regime"].fillna("NULL")

    # Win definition: forward return positive (simple, same as before)
    x["is_win"] = (x["fwd_ret_pct"] > 0).astype(int)

    grp = x.groupby("social_regime", dropna=False)

    summary = grp.agg(
        n=("social_regime", "size"),
        win_rate_pct=("is_win", lambda s: float(np.nanmean(s) * 100.0) if len(s) else np.nan),
        avg_fwd_ret_pct=("fwd_ret_pct", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce"))) if len(s) else np.nan),
        med_fwd_ret_pct=("fwd_ret_pct", lambda s: float(np.nanmedian(pd.to_numeric(s, errors="coerce"))) if len(s) else np.nan),
        avg_mfe_pct=("mfe_pct", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce"))) if len(s) else np.nan),
        avg_mae_pct=("mae_pct", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce"))) if len(s) else np.nan),
        avg_edge=("edge_score", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce"))) if len(s) else np.nan),
        med_edge=("edge_score", lambda s: float(np.nanmedian(pd.to_numeric(s, errors="coerce"))) if len(s) else np.nan),
        share_pos_edge_pct=("edge_score", lambda s: float(np.nanmean((pd.to_numeric(s, errors="coerce") > 0).astype(float)) * 100.0) if len(s) else np.nan),
        n_long=("pos", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
        n_short=("pos", lambda s: int((pd.to_numeric(s, errors="coerce") == -1).sum())),
        avg_z_count=("z_count", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce"))) if len(s) else np.nan),
        avg_z_sent=("z_sent", lambda s: float(np.nanmean(pd.to_numeric(s, errors="coerce"))) if len(s) else np.nan),
        share_no_bucket_pct=("null_reason", lambda s: float(np.mean(pd.Series(s).astype(str).eq("NO_BUCKET_MATCH")) * 100.0) if len(s) else np.nan),
    ).reset_index()

    # nicer ordering: non-NULL first
    order = ["NORMAL", "BUZZ", "FOMO", "FUD", "NULL"]
    summary["__ord"] = summary["social_regime"].apply(lambda r: order.index(r) if r in order else 99)
    summary = summary.sort_values(["__ord", "social_regime"]).drop(columns=["__ord"])

    return summary


def process_symbol_model(engine, symbol: str, model: str) -> None:
    df_obs = load_observe_csv(symbol, model)
    if df_obs is None or df_obs.empty:
        return

    # "signals" = rows where pos != 0
    df_sig = df_obs[df_obs["pos"] != 0].copy()
    if df_sig.empty:
        return

    # Load regimes around signal times
    t_min = df_sig["close_time_utc"].min() - pd.Timedelta("24h")
    t_max = df_sig["close_time_utc"].max() + pd.Timedelta("24h")
    df_soc, first_valid = db_load_regimes(engine, symbol, t_min=t_min, t_max=t_max)

    # Cutoff (reduces NULL)
    df_sig = apply_cutoff(df_sig, first_valid, symbol, model)
    if df_sig.empty:
        return

    # Join regimes
    df_join = asof_join_regime(df_sig, df_soc)

    # Forward metrics + edge
    df_metrics = compute_forward_metrics(df_join)
    df_metrics = compute_edge_score(df_metrics)

    # Summary
    summary = summarize_by_regime(df_metrics)

    # Print
    horizon_hours = HORIZON_BARS * 4  # because 4h interval
    print("\n" + "=" * 78)
    print(f"[SOCIAL CONTEXT REPORT] {symbol} {INTERVAL} {model} | horizon={HORIZON_BARS} bars ({horizon_hours}h on {INTERVAL})")
    print(summary.to_string(index=False))

    # Save
    detail_path = _logs_dir(symbol) / f"social_context_detail_{symbol}_{INTERVAL}_{model}.csv"
    summary_path = _logs_dir(symbol) / f"social_context_summary_{symbol}_{INTERVAL}_{model}.csv"
    df_metrics.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(f"  detail : {detail_path}")
    print(f"  summary: {summary_path}")


def main():
    load_dotenv()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("Chýba DATABASE_URL v .env (napr. postgresql://user:pass@localhost:5433/trading)")

    engine = create_engine(db_url)

    for symbol in SYMBOLS:
        for model in MODELS:
            process_symbol_model(engine, symbol, model)


if __name__ == "__main__":
    main()
# backtest_block.py
# A/B sanity test: block signals based on social regime (FOMO/FUD)
# Reads signals from public.signals and joins social regime by merge_asof (backward).

import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)

SYMBOL   = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "4h")

# How far back to analyze (optional)
LIMIT_SIGNALS = int(os.getenv("BLOCK_LIMIT_SIGNALS", "5000"))
TOLERANCE_HOURS = int(os.getenv("BLOCK_TOLERANCE_HOURS", "4"))

def pg_read(sql: str, params=None) -> pd.DataFrame:
    with psycopg2.connect(**PGCFG) as con:
        with con.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
    return pd.DataFrame(rows)

def table_exists(table_name: str) -> bool:
    sql = """
    SELECT EXISTS (
      SELECT 1
      FROM information_schema.tables
      WHERE table_schema = 'public' AND table_name = %s
    ) AS ok
    """
    df = pg_read(sql, (table_name,))
    return bool(df.iloc[0]["ok"]) if len(df) else False

def main():
    print(f"SYMBOL={SYMBOL} INTERVAL={INTERVAL}")
    print(f"LIMIT_SIGNALS={LIMIT_SIGNALS} | tolerance={TOLERANCE_HOURS}h\n")

    # 1) Signals
    sig_sql = """
    SELECT symbol, interval, bar_time, pos
    FROM public.signals
    WHERE symbol = %s
      AND interval = %s
      AND pos IS NOT NULL
      AND pos <> 0
    ORDER BY bar_time DESC
    LIMIT %s
    """
    sig = pg_read(sig_sql, (SYMBOL, INTERVAL, LIMIT_SIGNALS))
    if sig.empty:
        print("No signals found.")
        return

    sig["bar_time"] = pd.to_datetime(sig["bar_time"], utc=True)
    sig = sig.sort_values("bar_time").reset_index(drop=True)

    # 2) Social source preference: regime table first
    use_regime = table_exists("social_regime_3h")
    use_buckets = table_exists("social_buckets_3h")

    social = pd.DataFrame()
    social_mode = None

    if use_regime:
        social_mode = "regime"
        soc_sql = """
        SELECT symbol, bucket_start_utc, social_regime, z_count, z_sent
        FROM public.social_regime_3h
        WHERE symbol = %s
        ORDER BY bucket_start_utc
        """
        social = pg_read(soc_sql, (SYMBOL,))
        if not social.empty:
            social["bucket_start_utc"] = pd.to_datetime(social["bucket_start_utc"], utc=True)

    if (social.empty or social_mode is None) and use_buckets:
        social_mode = "buckets"
        soc_sql = """
        SELECT symbol, bucket_start_utc, count_total
        FROM public.social_buckets_3h
        WHERE symbol = %s
        ORDER BY bucket_start_utc
        """
        social = pg_read(soc_sql, (SYMBOL,))
        if not social.empty:
            social["bucket_start_utc"] = pd.to_datetime(social["bucket_start_utc"], utc=True)

    if social.empty:
        print("No social data found (neither social_regime_3h nor social_buckets_3h).")
        print("A/B blocking can't run.\n")
        print("Signals preview:")
        print(sig.tail(10)[["bar_time","pos"]].to_string(index=False))
        return

    print(f"Social mode: {social_mode}")
    print(f"Social rows: {len(social)} | Signal rows: {len(sig)}\n")

    # 3) merge_asof (backward, 4h tolerance)
    tol = pd.Timedelta(hours=TOLERANCE_HOURS)

    merged = pd.merge_asof(
        sig.sort_values("bar_time"),
        social.sort_values("bucket_start_utc"),
        left_on="bar_time",
        right_on="bucket_start_utc",
        direction="backward",
        tolerance=tol,
    )

    # 4) A/B blocking rules
    merged["side"] = merged["pos"].map(lambda x: "LONG" if x > 0 else "SHORT")

    merged["blocked_reason"] = None

    if social_mode == "regime":
        # Block LONG in FOMO, block SHORT in FUD
        cond_long_block = (merged["pos"] > 0) & (merged["social_regime"] == "FOMO")
        cond_short_block = (merged["pos"] < 0) & (merged["social_regime"] == "FUD")

        merged.loc[cond_long_block, "blocked_reason"] = "BLOCK_LONG_FOMO"
        merged.loc[cond_short_block, "blocked_reason"] = "BLOCK_SHORT_FUD"
    else:
        # buckets-only fallback: no regime => can't block by FOMO/FUD
        merged["blocked_reason"] = None

    merged["is_blocked"] = merged["blocked_reason"].notna()

    # 5) Summary
    total = len(merged)
    blocked = int(merged["is_blocked"].sum())
    null_social = int(merged["bucket_start_utc"].isna().sum())

    print("=== DIAGNOSTICS ===")
    print(f"Joined NULL social: {null_social} ({100*null_social/total:.1f}%)")
    print(f"Blocked signals:    {blocked} ({100*blocked/total:.1f}%)")

    if social_mode == "regime":
        print("\nBlocked breakdown:")
        print(merged["blocked_reason"].value_counts(dropna=True).to_string())

        print("\nRegime breakdown (joined):")
        print(merged["social_regime"].fillna("NULL").value_counts().to_string())

    # 6) Show last rows
    cols = ["bar_time", "pos", "side", "bucket_start_utc"]
    if social_mode == "regime":
        cols += ["social_regime", "z_count", "z_sent", "blocked_reason"]
    else:
        cols += ["count_total", "blocked_reason"]

    print("\n=== LAST 15 ROWS (joined) ===")
    print(merged.tail(15)[cols].to_string(index=False))

    # 7) Save
    _data_dir = os.getenv("DATA_DIR")
    if not _data_dir or not os.path.exists(_data_dir):
        raise RuntimeError(f"DATA_DIR chýba alebo SSD nie je pripojené: {_data_dir}")
    os.makedirs(os.path.join(_data_dir, "logs"), exist_ok=True)
    out_path = os.path.join(_data_dir, "logs", f"block_ab_{SYMBOL}_{INTERVAL}.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
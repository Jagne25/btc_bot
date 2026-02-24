# social_zscore_count.py
# koľko hluku !! z_count → či je teraz okolo BTC nezvyčajne veľký rozruch
# ÚLOHA:
# - načíta social_buckets_3h z PostgreSQL
# - pre každý symbol (BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT)
#   spočíta Z-score pre count_total
# - výsledok uloží do stĺpca z_count (bez lookaheadu)
#
# Poznámka:
# - používame rolling okno ~30 dní:
#   30 dní * 8 bucketov/deň = 240 bucketov

import os
from datetime import datetime

import psycopg2
from psycopg2.extras import DictCursor, execute_batch
from dotenv import load_dotenv

import pandas as pd

# ----- KONFIGURÁCIA -----

# rolling okno v počte bucketov (~30 dní)
WINDOW_BUCKETS = 240
# minimálny počet bucketov, aby sme vôbec počítali Z-score
MIN_BUCKETS = 50
# minimálna std, aby sme sa vyhli deleniu nulou
MIN_STD = 1e-6

load_dotenv()

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)


def get_db_connection():
    """Pripojenie na PostgreSQL, autocommit vypnutý."""
    conn = psycopg2.connect(**PGCFG)
    conn.autocommit = False
    return conn


def fetch_all_buckets(conn):
    """
    Načíta všetky riadky zo social_buckets_3h,
    ktoré majú nenull count_total.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            """
            SELECT symbol,
                   bucket_start_utc,
                   count_total
            FROM public.social_buckets_3h
            WHERE count_total IS NOT NULL
            ORDER BY symbol, bucket_start_utc
            """
        )
        rows = cur.fetchall()
    return rows


def compute_zscore_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre každý symbol spočíta Z-score pre count_total
    pomocou rolling okna WINDOW_BUCKETS (bez lookaheadu).

    Vstup: df s columnmi: symbol, bucket_start_utc, count_total
    Výstup: df s navyše: z_count
    """
    if df.empty:
        df["z_count"] = pd.Series(dtype="float64")
        return df

    # konverzie typov
    df = df.copy()
    df["bucket_start_utc"] = pd.to_datetime(df["bucket_start_utc"], utc=True)
    df["count_total"] = pd.to_numeric(df["count_total"], errors="coerce")

    # index si uložíme, aby sme vedeli priradiť výsledky späť
    df = df.reset_index(drop=False).rename(columns={"index": "orig_index"})

    z_list = []

    # groupby podľa symbolu
    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("bucket_start_utc").reset_index(drop=True)

        # rolling priemer a std len z count_total
        rolling_mean = g["count_total"].rolling(
            WINDOW_BUCKETS, min_periods=MIN_BUCKETS
        ).mean()
        rolling_std = g["count_total"].rolling(
            WINDOW_BUCKETS, min_periods=MIN_BUCKETS
        ).std(ddof=0)

        z = (g["count_total"] - rolling_mean) / rolling_std

        # kde je std príliš malé alebo NaN -> z = NaN
        z[(rolling_std < MIN_STD) | (rolling_std.isna())] = pd.NA

        tmp = g[["orig_index", "symbol", "bucket_start_utc"]].copy()
        tmp["z_count"] = z
        z_list.append(tmp)

    out = pd.concat(z_list, ignore_index=True)

    # vrátime df s pôvodným indexom + z_count
    out = out.set_index("orig_index")
    df = df.set_index("orig_index")
    df["z_count"] = out["z_count"]

    # back to normal index
    df = df.reset_index(drop=True)
    return df


def update_zcount_in_db(conn, df: pd.DataFrame):
    """
    Uloží z_count späť do social_buckets_3h.

    UPDATE public.social_buckets_3h
    SET z_count = %s, updated_at = NOW()
    WHERE symbol = %s AND bucket_start_utc = %s;
    """
    # vyberieme len riadky, kde máme ne-NaN z_count
    df2 = df.copy()
    df2 = df2[df2["z_count"].notna()]

    if df2.empty:
        print("[zscore] Žiadne riadky na update (z_count je všade NaN).")
        return

    records = []
    for _, row in df2.iterrows():
        records.append(
            {
                "symbol": row["symbol"],
                "bucket_start_utc": row["bucket_start_utc"],
                "z_count": float(row["z_count"]),
            }
        )

    sql = """
        UPDATE public.social_buckets_3h
        SET z_count   = %(z_count)s,
            updated_at = NOW()
        WHERE symbol = %(symbol)s
          AND bucket_start_utc = %(bucket_start_utc)s;
    """

    with conn.cursor() as cur:
        execute_batch(cur, sql, records, page_size=500)

    conn.commit()
    print(f"[zscore] Updated z_count pre {len(records)} bucketov.")


def main():
    now_utc = datetime.utcnow()
    print(f"[zscore] Štartujem výpočet Z-score ({now_utc:%Y-%m-%d %H:%M:%S} UTC)")

    conn = get_db_connection()

    try:
        rows = fetch_all_buckets(conn)
        print(f"[zscore] Načítaných bucketov: {len(rows)}")

        if not rows:
            print("[zscore] Žiadne dáta v social_buckets_3h, končím.")
            return

        df = pd.DataFrame(rows, columns=["symbol", "bucket_start_utc", "count_total"])

        df = compute_zscore_per_symbol(df)

        update_zcount_in_db(conn, df)

        print("[zscore] Hotovo.")
    finally:
        conn.close()
        print("[zscore] Spojenie na DB zatvorené.")


if __name__ == "__main__":
    main()
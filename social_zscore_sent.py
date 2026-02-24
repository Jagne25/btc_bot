# social_zscore_sent.py
#= či je ten sentiment nezvyčajne pozitívny/negatívny oproti normálu, z_sent → či je nálada nezvyčajne pozitívna alebo negatívna
# ÚLOHA:
# - načíta social_buckets_3h z PostgreSQL
# - pre každý symbol spočíta Z-score pre sent_avg
# - HISTÓRIA je lookahead-free (použijeme shift(1), aby okno končilo na T-1)
# - do histórie berieme len buckety, kde count_total >= N_MIN a sent_avg IS NOT NULL
# - ak aktuálny bucket má count_total < N_MIN alebo sent_avg je NULL -> z_sent = NULL
# - ak std je ~0 -> z_sent = 0
#
# Použitie:
#   python social_zscore_sent.py
# alebo:
#   python social_zscore_sent.py --window_buckets 72 --min_points 30 --n_min 8

import os
from datetime import datetime

import psycopg2
from psycopg2.extras import DictCursor, execute_batch
from dotenv import load_dotenv
import pandas as pd

# ----- DEFAULTY (bootstrap) -----
WINDOW_BUCKETS = 72      # ~9 dní pri 3h
MIN_POINTS = 10          # minimum bodov v histórii, aby sme rátali z-score
N_MIN = 3                # minimum count_total, aby bucket nebol šum
MIN_STD = 1e-6           # ochrana proti deleniu nulou

load_dotenv()

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5433")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)

def get_db_connection():
    conn = psycopg2.connect(**PGCFG)
    conn.autocommit = False
    return conn

def fetch_all_buckets(conn):
    """
    Načíta buckety, kde máme aspoň count_total (sent_avg môže byť NULL).
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            """
            SELECT symbol,
                   bucket_start_utc,
                   count_total,
                   sent_avg
            FROM public.social_buckets_3h
            WHERE count_total IS NOT NULL
            ORDER BY symbol, bucket_start_utc
            """
        )
        return cur.fetchall()

def compute_zsent_per_symbol(df: pd.DataFrame,
                            window_buckets: int,
                            min_points: int,
                            n_min: int) -> pd.DataFrame:
    """
    Z-score pre sent_avg per symbol, lookahead-free.

    Lookahead-free trik:
    - z-score v čase T má používať mean/std len z minulosti (T-1, T-2,...)
    - preto spravíme x_hist = sent_avg.shift(1)
      a rolling mean/std rátame z x_hist.
    """
    if df.empty:
        df["z_sent"] = pd.Series(dtype="float64")
        return df

    df = df.copy()
    df["bucket_start_utc"] = pd.to_datetime(df["bucket_start_utc"], utc=True)
    df["count_total"] = pd.to_numeric(df["count_total"], errors="coerce")
    df["sent_avg"] = pd.to_numeric(df["sent_avg"], errors="coerce")

    df = df.reset_index(drop=False).rename(columns={"index": "orig_index"})

    z_list = []

    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("bucket_start_utc").reset_index(drop=True)

        # validita aktuálneho bucketu (či vôbec môže mať z_sent)
        valid_now = (g["count_total"] >= n_min) & (g["sent_avg"].notna())

        # História pre rolling: berieme len minulosť (shift(1))
        x = g["sent_avg"]
        x_hist = x.shift(1)

        # Do histórie chceme rátať len body, kde count_total >= n_min
        # takže tie, čo nespĺňajú n_min, v histórii "vymažeme" na NA
        x_hist = x_hist.where(g["count_total"].shift(1) >= n_min)

        rolling_mean = x_hist.rolling(window_buckets, min_periods=min_points).mean()
        rolling_std = x_hist.rolling(window_buckets, min_periods=min_points).std(ddof=0)

        z = (x - rolling_mean) / rolling_std

        # ak std je príliš malé alebo NaN -> z_sent = 0 (ak máme mean), inak NA
        # (produkčné rozhodnutie: std==0 -> 0)
        zero_std = (rolling_std < MIN_STD) | (rolling_std.isna())
        z[zero_std & rolling_mean.notna()] = 0
        z[zero_std & rolling_mean.isna()] = pd.NA

        # ak aktuálny bucket nie je validný -> z_sent = NA
        z[~valid_now] = pd.NA

        tmp = g[["orig_index", "symbol", "bucket_start_utc"]].copy()
        tmp["z_sent"] = z
        z_list.append(tmp)

    out = pd.concat(z_list, ignore_index=True)
    out = out.set_index("orig_index")
    df = df.set_index("orig_index")
    df["z_sent"] = out["z_sent"]
    df = df.reset_index(drop=True)
    return df

def update_zsent_in_db(conn, df: pd.DataFrame):
    """
    Uloží z_sent späť do social_buckets_3h.
    """
    df2 = df.copy()
    df2 = df2[df2["z_sent"].notna()]

    if df2.empty:
        print("[z_sent] Žiadne riadky na update (z_sent je všade NA).")
        return

    records = []
    for _, row in df2.iterrows():
        records.append(
            {
                "symbol": row["symbol"],
                "bucket_start_utc": row["bucket_start_utc"],
                "z_sent": float(row["z_sent"]),
            }
        )

    sql = """
        UPDATE public.social_buckets_3h
        SET z_sent = %(z_sent)s,
            updated_at = NOW()
        WHERE symbol = %(symbol)s
          AND bucket_start_utc = %(bucket_start_utc)s;
    """

    with conn.cursor() as cur:
        execute_batch(cur, sql, records, page_size=500)

    conn.commit()
    print(f"[z_sent] Updated z_sent pre {len(records)} bucketov.")

def main():
    now_utc = datetime.utcnow()
    print(f"[z_sent] Štart ({now_utc:%Y-%m-%d %H:%M:%S} UTC)")

    conn = get_db_connection()
    try:
        rows = fetch_all_buckets(conn)
        print(f"[z_sent] Načítaných bucketov: {len(rows)}")

        if not rows:
            print("[z_sent] Žiadne dáta v social_buckets_3h, končím.")
            return

        df = pd.DataFrame(rows, columns=["symbol", "bucket_start_utc", "count_total", "sent_avg"])

        df = compute_zsent_per_symbol(df, WINDOW_BUCKETS, MIN_POINTS, N_MIN)

        update_zsent_in_db(conn, df)

        print("[z_sent] Hotovo.")
    finally:
        conn.close()
        print("[z_sent] DB spojenie zatvorené.")

if __name__ == "__main__":
    main()
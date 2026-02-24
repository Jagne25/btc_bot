# social_bucketizer.py
#
# ÚLOHA:
# 1) Zoberie príspevky zo social_mentions za posledných X hodín
# 2) Hodí ich do 3h "boxov" (bucketov)
# 3) Spočíta, koľko príspevkov bolo v každom boxe
# 4) Spočíta priemerný sentiment v buckete (sent_avg) z sent_score
# 5) Zapíše výsledok do tabuľky social_buckets_3h

import os
from datetime import datetime, timedelta
from collections import defaultdict

import psycopg2
from psycopg2.extras import DictCursor, execute_batch
from dotenv import load_dotenv


# ================== KONFIGURÁCIA ==================

# Každý bucket je 3 hodiny
BUCKET_HOURS = 3

# Koľko hodín dozadu pozeráme vždy naraz (napr. 72 = posledné 3 dni)
LOOKBACK_HOURS = 24*30

# Načítame .env, aby sme vedeli PGHOST, PGPORT, atď.
load_dotenv()

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)


def get_db_connection():
    """
    Pripojenie na PostgreSQL.
    Vráti 'conn', cez ktorý robíme SELECT/INSERT.
    """
    conn = psycopg2.connect(**PGCFG)
    # Autocommit vypneme, commit spravíme ručne
    conn.autocommit = False
    return conn


# ================== POMOCNÉ FUNKCIE ==================

def floor_to_3h_bucket(dt_utc: datetime) -> datetime:
    """
    Zoberie čas a zaokrúhli ho dole na začiatok 3h okna.

    Príklad:
        10:05 -> 09:00
        11:59 -> 09:00
        12:00 -> 12:00
    """
    bucket_hour = (dt_utc.hour // BUCKET_HOURS) * BUCKET_HOURS
    return dt_utc.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)


def fetch_recent_mentions(conn, since_utc: datetime, until_utc: datetime):
    """
    Zoberie z social_mentions všetky riadky
    v intervale [since_utc, until_utc).

    Teraz berieme aj sent_score, aby sme vedeli rátať sent_avg.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            """
            SELECT symbol,
                   post_time_utc,
                   lang_code,
                   sent_score
            FROM public.social_mentions
            WHERE post_time_utc >= %s
              AND post_time_utc < %s
            """,
            (since_utc, until_utc),
        )
        rows = cur.fetchall()

    return rows


def build_buckets(rows):
    """
    Z raw riadkov (symbol, post_time_utc, lang_code, sent_score)
    spraví 3h buckety v pamäti.

    Výstup: dict
      key: (symbol, bucket_start_utc)
      value: dict s hodnotami:
        - count_total, count_en, count_czsk
        - sent_sum, sent_n  (na výpočet sent_avg)
    """

    # defaultdict = keď sa spýtame na neexistujúci key,
    # automaticky vytvorí nový záznam s danou štruktúrou
    buckets = defaultdict(
        lambda: {
            "count_total": 0,
            "count_en": 0,
            "count_czsk": 0,
            "sent_sum": 0.0,
            "sent_n": 0,
        }
    )

    for row in rows:
        symbol = row["symbol"]
        post_time_utc = row["post_time_utc"]
        lang_code = row["lang_code"]
        sent_score = row["sent_score"]  # -1 / 0 / +1 alebo None

        # Zistíme, do ktorého 3h bucketu patrí tento príspevok
        bucket_start = floor_to_3h_bucket(post_time_utc)
        key = (symbol, bucket_start)

        # Vždy zvýšime celkový počet
        buckets[key]["count_total"] += 1

        # Podľa jazyka zvýšime EN alebo CZSK counter
        if lang_code == "en":
            buckets[key]["count_en"] += 1
        elif lang_code == "czsk":
            buckets[key]["count_czsk"] += 1

        # Ak máme spočítaný sent_score, pridáme ho do súčtu
        if sent_score is not None:
            # sent_score je numeric v DB, pre istotu pretypujeme na float
            buckets[key]["sent_sum"] += float(sent_score)
            buckets[key]["sent_n"] += 1

    return buckets


def upsert_buckets(conn, buckets):
    """
    Zoberie všetky spočítané buckety a zapíše ich do tabuľky social_buckets_3h.

    Ak riadok pre (symbol, bucket_start_utc) ešte neexistuje -> INSERT.
    Ak už existuje                              -> UPDATE (prepíšeme counts + sent_avg).
    """
    if not buckets:
        print("[bucketizer] nič na zápis (žiadne social_mentions v danom intervale).")
        return

    records = []
    for (symbol, bucket_start_utc), counts in buckets.items():
        bucket_end_utc = bucket_start_utc + timedelta(hours=BUCKET_HOURS)

        # Výpočet priemerného sentimentu v buckete
        sent_avg = None
        sent_sum = counts.get("sent_sum", 0.0)
        sent_n = counts.get("sent_n", 0)
        if sent_n > 0:
            sent_avg = sent_sum / sent_n

        records.append(
            {
                "symbol": symbol,
                "bucket_start_utc": bucket_start_utc,
                "bucket_end_utc": bucket_end_utc,
                "count_total": counts["count_total"],
                "count_en": counts["count_en"],
                "count_czsk": counts["count_czsk"],
                "sent_avg": sent_avg,
            }
        )

    sql = """
        INSERT INTO public.social_buckets_3h (
            symbol,
            bucket_start_utc,
            bucket_end_utc,
            count_total,
            count_en,
            count_czsk,
            sent_avg,
            sent_p50,
            sent_p90,
            z_count,
            z_sent,
            inserted_at,
            updated_at
        )
        VALUES (
            %(symbol)s,
            %(bucket_start_utc)s,
            %(bucket_end_utc)s,
            %(count_total)s,
            %(count_en)s,
            %(count_czsk)s,
            %(sent_avg)s,
            NULL,
            NULL,
            NULL,
            NULL,
            NOW(),
            NOW()
        )
        ON CONFLICT (symbol, bucket_start_utc) DO UPDATE
        SET
            bucket_end_utc = EXCLUDED.bucket_end_utc,
            count_total    = EXCLUDED.count_total,
            count_en       = EXCLUDED.count_en,
            count_czsk     = EXCLUDED.count_czsk,
            sent_avg       = EXCLUDED.sent_avg,
            updated_at     = NOW();
    """

    with conn.cursor() as cur:
        execute_batch(cur, sql, records, page_size=500)

    conn.commit()
    print(f"[bucketizer] upsertol som {len(records)} bucketov.")


# ================== MAIN ==================

def main():
    """
    Hlavný "vstup" skriptu.

    1) Spočíta časové okno [now - LOOKBACK_HOURS, now]
    2) Natiahne príspevky z social_mentions
    3) Spraví 3h buckety (vrátane sent_avg)
    4) Zapíše ich do social_buckets_3h
    """
    now_utc = datetime.utcnow()
    since_utc = now_utc - timedelta(hours=LOOKBACK_HOURS)

    print(f"[bucketizer] bežím pre interval {since_utc} až {now_utc} (UTC).")

    conn = get_db_connection()

    try:
        rows = fetch_recent_mentions(conn, since_utc, now_utc)
        print(f"[bucketizer] načítaných raw mentions: {len(rows)}")

        buckets = build_buckets(rows)
        print(f"[bucketizer] spočítaných bucketov: {len(buckets)}")

        upsert_buckets(conn, buckets)

    finally:
        conn.close()
        print("[bucketizer] hotovo, spojenie na DB zatvorené.")


if __name__ == "__main__":
    main()
import os
import re
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

# ----------- DB CONFIG (to isté ako v bucketizeri) -----------

load_dotenv()

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)

def get_db_connection():
    conn = psycopg2.connect(**PGCFG)
    conn.autocommit = False
    return conn

# ----------- Jednoduchý slovníkový sentiment ------------

POSITIVE_WORDS_EN = [
    "pump", "pumps", "pumped",
    "bull", "bullish",
    "rally", "moon", "moonshot",
    "gain", "gains", "profit", "profits", "green",
    "surge", "spike", "breakout",
]

NEGATIVE_WORDS_EN = [
    "dump", "dumps", "dumped",
    "crash", "crashes", "crashed",
    "bear", "bearish",
    "fear", "panic", "panic sell",
    "selloff", "sell-off",
    "plunge", "collapse",
    "red", "liquidation", "rekt",
]

POSITIVE_WORDS_CZSK = [
    "rast", "rastu", "rastie",
    "růst", "roste",
    "zisk", "zisky", "ziskový",
    "silný", "silná", "silné",
    "býčí", "bull", "bullish",
    "pumpa", "pumpuje",
]

NEGATIVE_WORDS_CZSK = [
    "prepad", "prepadol", "padá", "pád", "spadne",
    "krach", "krize", "kríza",
    "strach", "obavy", "panika", "panické",
    "medveď", "medvedí", "bear", "bearish",
    "červený", "červené",
]

# pre rýchlosť z nich spravíme sety
POS_EN_SET = set(POSITIVE_WORDS_EN)
NEG_EN_SET = set(NEGATIVE_WORDS_EN)
POS_CZSK_SET = set(POSITIVE_WORDS_CZSK)
NEG_CZSK_SET = set(NEGATIVE_WORDS_CZSK)

TOKEN_RE = re.compile(r"[a-záäčďéěíľĺňóôřšťúůýž]+")

def simple_sentiment(text: str, lang_code: str) -> int:
    """
    Vracia -1, 0 alebo +1 podľa toho,
    či je text skôr negatívny / neutrálny / pozitívny.
    """
    if not text:
        return 0

    text = text.lower()
    tokens = TOKEN_RE.findall(text)

    if not tokens:
        return 0

    pos = 0
    neg = 0

    if lang_code == "en":
        for t in tokens:
            if t in POS_EN_SET:
                pos += 1
            if t in NEG_EN_SET:
                neg += 1
    elif lang_code == "czsk":
        for t in tokens:
            if t in POS_CZSK_SET:
                pos += 1
            if t in NEG_CZSK_SET:
                neg += 1
    else:
        # neznámy jazyk – zatiaľ neutrál
        return 0

    score = 0
    if pos > neg:
        score = 1
    elif neg > pos:
        score = -1
    else:
        score = 0

    return score

# ----------- DB operácie ------------

def fetch_unscored_mentions(conn, limit: int = 500):
    """
    Zoberie články, kde ešte sent_score je NULL.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            """
            SELECT id, raw_text, lang_code
            FROM public.social_mentions
            WHERE sent_score IS NULL
              AND lang_code IN ('en', 'czsk')
            ORDER BY inserted_at ASC
            LIMIT %s;
            """,
            (limit,),
        )
        return cur.fetchall()

def update_sent_scores(conn, rows):
    if not rows:
        return 0

    with conn.cursor() as cur:
        for row in rows:
            score = simple_sentiment(row["raw_text"], row["lang_code"])
            cur.execute(
                """
                UPDATE public.social_mentions
                SET sent_score = %s
                WHERE id = %s;
                """,
                (score, row["id"]),
            )
    conn.commit()
    return len(rows)

def main():
    conn = get_db_connection()
    try:
        total = 0
        while True:
            rows = fetch_unscored_mentions(conn, limit=500)
            if not rows:
                break
            n = update_sent_scores(conn, rows)
            total += n
            print(f"[sentiment] označených {n} príspevkov...")
        print(f"[sentiment] hotovo, celkovo označených: {total}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
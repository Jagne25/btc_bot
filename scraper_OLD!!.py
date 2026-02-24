import os
import sys
import logging
from datetime import datetime, timedelta, timezone

import requests
import psycopg2
from dotenv import load_dotenv

# ===================== LOGGING =====================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s: %(message)s',
)
log = logging.getLogger("scraper_social")

# ===================== CONFIG =====================

# načítaj .env (používame rovnaký .env ako bot)
load_dotenv()

# DB parametre z .env
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")

# CryptoPanic API key – pridaj si do .env:
# CRYPTOPANIC_API_KEY=...
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")

if not CRYPTOPANIC_API_KEY:
    log.error("CRYPTOPANIC_API_KEY not set in .env")
    sys.exit(1)

# mapovanie našich symbolov na skratky pre CryptoPanic
SYMBOL_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BNBUSDT": "BNB",
}

# koľko hodín dozadu chceme brať správy
LOOKBACK_HOURS = 24


# ===================== DB HELPERS =====================

def get_pg_conn():
    """Otvorí spojenie na Postgres."""
    return psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
    )


def upsert_mentions(conn, rows):
    """
    Vloží zoznam záznamov do public.social_mentions.
    rows = list dictov, každý má key:
      symbol, source, external_id, post_time_utc, text, lang_code
    is_retweet / is_reply nastavíme na FALSE.
    """
    if not rows:
        return

    with conn.cursor() as cur:
        sql = """
        INSERT INTO public.social_mentions (
            symbol,
            source,
            external_id,
            post_time_utc,
            text,
            lang_code,
            is_retweet,
            is_reply
        )
        VALUES (
            %(symbol)s,
            %(source)s,
            %(external_id)s,
            %(post_time_utc)s,
            %(text)s,
            %(lang_code)s,
            FALSE,
            FALSE
        )
        ON CONFLICT (symbol, source, external_id) DO NOTHING;
        """
        cur.executemany(sql, rows)
    conn.commit()


# ===================== CRYPTOPANIC SCRAPER =====================

def fetch_cryptopanic(symbol: str, since: datetime):
    """
    Stiahne zoznam správ pre daný symbol z CryptoPanic.
    Vracia list dictov pripravených pre upsert_mentions.
    """
    if symbol not in SYMBOL_MAP:
        raise ValueError(f"Unknown symbol {symbol}")

    asset = SYMBOL_MAP[symbol]  # BTC / ETH / SOL / BNB

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "currencies": asset,
        "kind": "news",      # správy (nie ‚media‘)
        "filter": "news",    # len news, bez „rising/jokes“ atď.
        "regions": "en",     # len angličtina zatiaľ
    }

    log.info(f"{symbol}: Fetch CryptoPanic news")
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    for item in data.get("results", []):
        # id – použijeme ako external_id (unikát pre deň/zdroj)
        external_id = str(item.get("id"))
        published_at = item.get("published_at") or item.get("created_at")

        # CryptoPanic má timestamp ako ISO string, skonvertujeme na UTC datetime
        # (ak by tam bola zóna, datetime.fromisoformat ju vie prebrať)
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        # filtrovanie podľa 'since'
        if dt < since:
            continue

        title = item.get("title") or ""
        source_title = item.get("source", {}).get("title") or ""
        # text = nadpis + názov portálu
        full_text = f"{title} ({source_title})".strip()

        rows.append(
            {
                "symbol": symbol,
                "source": "cryptopanic",
                "external_id": external_id,
                "post_time_utc": dt.astimezone(timezone.utc),
                "text": full_text,
                "lang_code": "en",
            }
        )

    log.info(f"{symbol}: fetched {len(rows)} rows from CryptoPanic")
    return rows


# ===================== MAIN =====================

def main():
    log.info("scraper_social: start")

    # časový rozsah – posledných X hodín
    now_utc = datetime.now(timezone.utc)
    since = now_utc - timedelta(hours=LOOKBACK_HOURS)

    conn = get_pg_conn()
    try:
        total = 0
        for symbol in SYMBOL_MAP.keys():
            try:
                rows = fetch_cryptopanic(symbol, since)
                upsert_mentions(conn, rows)
                total += len(rows)
            except Exception as e:
                log.error("ERROR %s: %s", symbol, e)

        log.info("scraper_social: done, inserted ~%s rows", total)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
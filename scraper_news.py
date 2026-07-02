import os
import logging
from datetime import datetime, timezone
from typing import Iterable, Dict, Any

import requests
import feedparser
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

# --------------------------------------------------
#  ZÁKLAD: načítanie .env + logging
# --------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Mapa ticker -> tvoj futures symbol
SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "XBT": "BTCUSDT",   # občas sa používa takto
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
}

# --------------------------------------------------
#  DB pomocné funkcie
# --------------------------------------------------
def get_db_conn():
    """Pripojenie na PostgreSQL podľa PG* z .env."""
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE", "trading"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
    )


def upsert_mention(cur: DictCursor, data: Dict[str, Any]) -> None:
    """
    Vloží 1 záznam do social_mentions.
    Ak už existuje (source, external_id), nič nespraví.
    """
    cur.execute(
        """
        INSERT INTO public.social_mentions
            (source, symbol, external_id,
             is_retweet, is_reply, lang_code,
             post_time_utc, raw_text)
        VALUES
            (%(source)s, %(symbol)s, %(external_id)s,
             FALSE, FALSE, %(lang_code)s,
             %(post_time_utc)s, %(raw_text)s)
        ON CONFLICT (source, external_id) DO NOTHING;
        """,
        data,
    )

# --------------------------------------------------
#  1) CryptoPanic public feed (bez API key)
# --------------------------------------------------
def fetch_cryptopanic() -> Iterable[Dict[str, Any]]:
    """
    Berie important news z CryptoPanic public feedu.
    Bez API key, limitovaný, ale stabilný.
    """
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "kind": "news",
        "filter": "important",
        "public": "true",  # dôležité pre free feed
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logging.error("CryptoPanic request failed: %s", e)
        return []

    payload = resp.json()
    results = payload.get("results", []) or []

    for item in results:
        # currencies = zoznam coinov, ktorých sa news týka
        currencies = item.get("currencies") or []
        codes = [c.get("code") for c in currencies if c.get("code")]

        symbol = None
        for code in codes:
            if code in SYMBOL_MAP:
                symbol = SYMBOL_MAP[code]
                break

        if symbol is None:
            # nič pre naše 4 coiny, preskoč
            continue

        # unikát ID (aby fungoval UNIQUE)
        external_id = f"cryptopanic:{item.get('id')}"

        # čas publikovania
        published_at = item.get("published_at")
        if published_at:
            # napr. "2025-11-17T14:20:00+00:00"
            post_time = datetime.fromisoformat(
                published_at.replace("Z", "+00:00")
            )
        else:
            post_time = datetime.now(timezone.utc)

        title = item.get("title") or ""
        url_news = item.get("url") or ""
        raw_text = f"{title} {url_news}".strip()

        lang_code = item.get("language") or None

        yield {
            "source": "cryptopanic",
            "symbol": symbol,
            "external_id": external_id,
            "lang_code": lang_code,
            "post_time_utc": post_time.astimezone(timezone.utc),
            "raw_text": raw_text,
        }

# --------------------------------------------------
#  2) RSS (Reddit + news weby)
# --------------------------------------------------
RSS_FEEDS = [
    # (identifikátor_zdroja, URL)
    ("reddit_bitcoin", "https://www.reddit.com/r/Bitcoin/.rss"),
    ("reddit_crypto", "https://www.reddit.com/r/CryptoCurrency/.rss"),
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
]


def guess_symbol_from_title(title: str) -> str | None:
    """
    Úplne jednoduché hádanie coinu podľa textu.
    Stačí na prototyp.
    """
    t = title.lower()

    if "bitcoin" in t or "btc" in t:
        return "BTCUSDT"
    if "ethereum" in t or "eth" in t:
        return "ETHUSDT"
    if "solana" in t or "sol " in t:
        return "SOLUSDT"
    if "bnb" in t or "binance coin" in t:
        return "BNBUSDT"

    return None


def fetch_rss() -> Iterable[Dict[str, Any]]:
    """Prejde definované RSS feedy a vytvorí záznamy pre social_mentions."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; btc-scraper/1.0)"}
    for source_name, feed_url in RSS_FEEDS:
        try:
            resp = requests.get(feed_url, headers=headers, timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as e:
            logging.error("RSS %s parse failed: %s", source_name, e)
            continue

        for entry in feed.entries:
            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")

            symbol = guess_symbol_from_title(title)
            if symbol is None:
                # nič pre naše coiny, preskoč
                continue

            # nejaké ID – vezmeme 'id' alebo link
            entry_id = getattr(entry, "id", "") or link
            external_id = f"{source_name}:{entry_id}"

            # published_parsed je time.struct_time
            published = getattr(entry, "published_parsed", None)
            if published:
                post_time = datetime(
                    year=published.tm_year,
                    month=published.tm_mon,
                    day=published.tm_mday,
                    hour=published.tm_hour,
                    minute=published.tm_min,
                    second=published.tm_sec,
                    tzinfo=timezone.utc,
                )
            else:
                post_time = datetime.now(timezone.utc)

            raw_text = f"{title} {link}".strip()

            yield {
                "source": source_name,
                "symbol": symbol,
                "external_id": external_id,
                "lang_code": "en",  # RSS väčšinou EN
                "post_time_utc": post_time,
                "raw_text": raw_text,
            }

# --------------------------------------------------
#  MAIN
# --------------------------------------------------
def main():
    logging.info("scraper_news: start")
    conn = get_db_conn()
    cur = conn.cursor(cursor_factory=DictCursor)

    inserted = 0

    # 1) CryptoPanic
    for row in fetch_cryptopanic():
        upsert_mention(cur, row)
        inserted += 1

    # 2) RSS (Reddit + news)
    for row in fetch_rss():
        upsert_mention(cur, row)
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    logging.info("scraper_news: done, inserted ~%s rows (vrátane dup, ktoré sa ignorovali)", inserted)


if __name__ == "__main__":
    main()
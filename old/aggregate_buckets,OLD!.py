import os, datetime as dt
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import statistics as stats

load_dotenv()

PGCFG = dict(
    host=os.getenv("PGHOST","localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE","trading"),
    user=os.getenv("PGUSER","postgres"),
    password=os.getenv("PGPASSWORD","")
)

SYMBOLS = [s.strip().upper() for s in os.getenv("SOCIAL_SYMBOLS","BTCUSDT").split(",") if s.strip()]
BUCKET_H = 3
ROLL_DAYS_FOR_Z = 30

def _conn():
    return psycopg2.connect(**PGCFG)

def bucket_bounds(now=None):
    now = now or dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # 3h bucket aligned to UTC: 0,3,6,...
    start_hour = (now.hour // BUCKET_H) * BUCKET_H
    start = now.replace(hour=start_hour)
    end = start + dt.timedelta(hours=BUCKET_H)
    return start, end

def get_mentions(con, symbol, start, end):
    with con.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT post_time_utc, sent_score, lang_code
            FROM public.social_mentions
            WHERE symbol=%s
              AND post_time_utc >= %s
              AND post_time_utc < %s
        """, (symbol, start, end))
        return cur.fetchall()

def get_hist_sent(con, symbol, until_end):
    with con.cursor() as cur:
        cur.execute("""
            SELECT sent_score
            FROM public.social_mentions
            WHERE symbol=%s
              AND post_time_utc >= %s::timestamptz - interval '%s days'
              AND post_time_utc < %s
              AND sent_score IS NOT NULL
        """, (symbol, until_end, ROLL_DAYS_FOR_Z, until_end))
        return [row[0] for row in cur.fetchall()]

def p50(xs):
    if not xs: return None
    return float(stats.median(xs))
def p90(xs):
    if not xs: return None
    xs = sorted(xs)
    k = int(0.9*(len(xs)-1))
    return float(xs[k])

def upsert_bucket(con, symbol, start, end, sent_list, count_en, count_czsk):
    count_total = len(sent_list)
    sent_avg = float(sum(sent_list)/count_total) if count_total else None
    sent_p50 = p50(sent_list)
    sent_p90 = p90(sent_list)

    # Z-skóre sentimentu oproti 30d histórii
    hist = get_hist_sent(con, symbol, end)
    if hist:
        mu = sum(hist)/len(hist)
        # stdev (popul.)
        var = sum((x-mu)**2 for x in hist)/len(hist)
        sd = var**0.5
        z_sent = (sent_avg - mu)/sd if (sent_avg is not None and sd>1e-9) else None
    else:
        z_sent = None

    z_count = None  # (voliteľne by si vedel rátať aj Z počtu zmienok)

    with con.cursor() as cur:
        cur.execute("""
        INSERT INTO public.social_buckets_3h
          (symbol, bucket_start_utc, bucket_end_utc,
           count_total, count_en, count_czsk,
           sent_avg, sent_p50, sent_p90,
           z_count, z_sent, inserted_at, updated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW(), NOW())
        ON CONFLICT (symbol, bucket_start_utc) DO UPDATE SET
           count_total=EXCLUDED.count_total,
           count_en=EXCLUDED.count_en,
           count_czsk=EXCLUDED.count_czsk,
           sent_avg=EXCLUDED.sent_avg,
           sent_p50=EXCLUDED.sent_p50,
           sent_p90=EXCLUDED.sent_p90,
           z_count=EXCLUDED.z_count,
           z_sent=EXCLUDED.z_sent,
           bucket_end_utc=EXCLUDED.bucket_end_utc,
           updated_at=NOW();
        """, (symbol, start, end, count_total, count_en, count_czsk,
              sent_avg, sent_p50, sent_p90, z_count, z_sent))
    return count_total

def main():
    start, end = bucket_bounds()
    with _conn() as con:
        for sym in SYMBOLS:
            rows = get_mentions(con, sym, start, end)
            sent = [r["sent_score"] for r in rows if r["sent_score"] is not None]
            count_en = sum(1 for r in rows if r["lang_code"] == "en")
            count_czsk = sum(1 for r in rows if r["lang_code"] == "czsk")
            n = upsert_bucket(con, sym, start, end, sent, count_en, count_czsk)
            print(f"[{sym}] bucket {start:%Y-%m-%d %H:%M}Z → n={n}")
    print("Done.")

if __name__ == "__main__":
    main()
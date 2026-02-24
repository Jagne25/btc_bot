# db_mirror.py
import os
import psycopg2
from psycopg2.extras import Json

def _conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE", "trading"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "")
    )

def insert_signal(row: dict):
    """
    Jednoduchy, jednofazovy INSERT do public.signals.
    Ziadne 'utc_time', iba stlpce, ktore v tabulke mas.
    Na nepovinne featury (atr_pct, vol_rel_20, dist_*) posielame NULL, ak nie su v row.
    Idempotencia cez UNIQUE(symbol, interval, bar_time).
    """
    sql = """
    INSERT INTO public.signals (
        created_at, bar_time,
        symbol, interval, model,
        proba_up, side, close,
        sl_price, partial_tp, final_tp,
        trail_k, qty_suggest,
        env_version, atr_pct, vol_rel_20, dist_hh_50, dist_ll_50,
        raw
    )
    VALUES (
        NOW(), %(bar_time)s,
        %(symbol)s, %(interval)s, %(model)s,
        %(proba_up)s, %(side)s, %(close)s,
        %(sl_price)s, %(partial_tp)s, %(final_tp)s,
        %(trail_k)s, %(qty_suggest)s,
        %(env_version)s, %(atr_pct)s, %(vol_rel_20)s, %(dist_hh_50)s, %(dist_ll_50)s,
        %(raw)s
    )
    ON CONFLICT (symbol, interval, bar_time) DO NOTHING;
    """
    params = {
        "bar_time":    row["bar_time"],
        "symbol":      row["symbol"],
        "interval":    row["interval"],
        "model":       row.get("model"),
        "proba_up":    row.get("proba_up"),
        "side":        row.get("side"),
        "close":       row.get("close"),
        "sl_price":    row.get("sl_price"),
        "partial_tp":  row.get("partial_tp"),
        "final_tp":    row.get("final_tp"),
        "trail_k":     row.get("trail_k"),
        "qty_suggest": row.get("qty_suggest"),
        # zober z row alebo z ENV
        "env_version": row.get("env_version") or os.getenv("ENV_VERSION"),
        # tieto nemusia byt v out_row; NULL je OK
        "atr_pct":     row.get("atr_pct"),
        "vol_rel_20":  row.get("vol_rel_20"),
        "dist_hh_50":  row.get("dist_hh_50"),
        "dist_ll_50":  row.get("dist_ll_50"),
        "raw":         Json(row),
    }

    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
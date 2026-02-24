# bot_trade.py — LONG aj SHORT (Futures TESTNET)
# - číta posledný riadok z CSV (LOG_CSV z .env; fallback logs/signals.csv)
# - rebalance na podpísanú target qty (risk-based) s portfóliovým stropom rizika
# - ochrany:
#     LONG : SL = STOP_MARKET SELL (MARK), partial TP = TAKE_PROFIT_MARKET SELL (LAST), final TP SELL (LAST)
#     SHORT: SL = STOP_MARKET BUY  (MARK), partial TP = TAKE_PROFIT_MARKET BUY  (LAST), final TP BUY  (LAST)
# - UPDATE_PROTECTION: LOCK_ON_ENTRY | ALWAYS_REFRESH | REFRESH_WITH_DRIFT
# - SOFT EXIT: po min. držbe a potvrdení uzavrie pozíciu na market (SELL pri long, BUY pri short)
# - HEDGE SAFE: kontrola režimu účtu; v hedge nepoužívame closePosition a neotočíme pozíciu
#
# + SOCIAL OBSERVE:
#   - regime z DB (public.social_regime_3h) pre čas signálu
#   - edge z CSV snapshotu (logs/social_edge_snapshot.csv)
#   - iba VYPIS, bez zásahu do qty / orderov

import os, math, json, datetime as dt, time, csv
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import psycopg2
from dotenv import load_dotenv
from binance.um_futures import UMFutures

# načítaj .env podľa ENV_FILE (ak nie je, použije .env)
load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)

# --- cesty / konštanty ---
CSV_PATH   = os.getenv("LOG_CSV", "logs/signals.csv")
STATE_DIR  = os.getenv("STATE_DIR", "logs")
STATE_PATH = None  # nastaví sa v main() podľa symbolu

PORTFOLIO_STATE = os.getenv("PORTFOLIO_STATE", "logs/portfolio_state.json")
PORTFOLIO_LOCK  = PORTFOLIO_STATE + ".lock"

# --- SOCIAL snapshot/config ---
SOCIAL_EDGE_SNAPSHOT = os.getenv("SOCIAL_EDGE_SNAPSHOT", "logs/social_edge_snapshot.csv")
MODEL_NAME = os.getenv("MODEL", "LR").upper()

# --- burzové kroky ---
QTY_STEP    = float(os.getenv("QTY_STEP", "0.001"))
MIN_QTY     = float(os.getenv("MIN_QTY", "0.001"))
PRICE_TICK  = float(os.getenv("PRICE_TICK", "0.01"))  # krok ceny per-symbol (z .env)

def _floor_tick(x: float, tick: float) -> float:
    return round(math.floor(x / tick + 1e-12) * tick, 10)

def _ceil_tick(x: float, tick: float) -> float:
    return round(math.ceil(x / tick - 1e-12) * tick, 10)

def round_price_for_sl(price: float, pos_side: int, tick: float) -> float:
    """SL trigger: long -> floor, short -> ceil."""
    return _floor_tick(price, tick) if pos_side > 0 else _ceil_tick(price, tick)

def round_price_for_tp(price: float, pos_side: int, tick: float) -> float:
    """TP trigger: long -> ceil, short -> floor."""
    return _ceil_tick(price, tick) if pos_side > 0 else _floor_tick(price, tick)

# ---------- SOCIAL helpers ----------
def parse_utc_time(s: Optional[str]) -> Optional[datetime]:
    """
    Podporí:
      - '2026-01-18T12:34:00Z'
      - '2026-01-18 12:34:00'
      - ISO s offsetom
    Výstup: tz-aware UTC alebo None.
    """
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        s = s.replace(" ", "T")
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception:
        return None

def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE", "trading"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
    )

def fetch_social_regime(symbol: str, ts_utc: Optional[datetime]) -> Dict[str, Any]:
    """
    Nájde posledný bucket <= času signálu.
    Ak ts_utc None: berie posledný bucket v tabuľke.
    """
    out = {
        "bucket_start_utc": None,
        "social_regime": None,
        "z_count": None,
        "z_sent": None,
        "status": "NO_DB",
    }
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        if ts_utc is None:
            cur.execute(
                """
                SELECT bucket_start_utc, social_regime, z_count, z_sent
                FROM public.social_regime_3h
                WHERE symbol = %s
                ORDER BY bucket_start_utc DESC
                LIMIT 1;
                """,
                (symbol,),
            )
        else:
            cur.execute(
                """
                SELECT bucket_start_utc, social_regime, z_count, z_sent
                FROM public.social_regime_3h
                WHERE symbol = %s
                  AND bucket_start_utc <= %s
                ORDER BY bucket_start_utc DESC
                LIMIT 1;
                """,
                (symbol, ts_utc),
            )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            out["status"] = "NO_BUCKET_MATCH"
            return out

        out["bucket_start_utc"] = row[0]
        out["social_regime"] = row[1]
        out["z_count"] = row[2]
        out["z_sent"] = row[3]
        out["status"] = "OK"
        return out
    except Exception as e:
        out["status"] = f"DB_ERROR: {e}"
        return out

def _detect_delimiter(header_line: str) -> str:
    # jednoduché a robustné: väčšinou je to ',' alebo ';'
    c = header_line.count(",")
    s = header_line.count(";")
    return ";" if s > c else ","

def read_edge_snapshot(path: str, symbol: str, model: str, regime: Optional[str]) -> Dict[str, Any]:
    """
    Hľadá riadok v CSV:
      symbol, model, social_regime/regime, n, avg_edge/edge, risk_mult

    FIX:
      - číta utf-8-sig (odstráni BOM)
      - stripne fieldnames (aby ' symbol ' nerobilo bordel)
      - zvládne aj delimiter ';'
    """
    out = {
        "found": False,
        "n": None,
        "avg_edge": None,
        "risk_mult": None,
        "status": "NO_FILE",
    }
    if not regime:
        out["status"] = "NO_REGIME"
        return out

    # urob path stabilný (relatívne na priečinok skriptu)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = path if os.path.isabs(path) else os.path.join(base_dir, path)

    if not os.path.exists(abs_path):
        return out

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    try:
        with open(abs_path, "r", encoding="utf-8-sig", newline="") as f:
            # zober prvý riadok a zisti delimiter
            first = f.readline()
            if not first:
                out["status"] = "CSV_EMPTY"
                return out

            delim = _detect_delimiter(first)
            # priprav DictReader s očistenými header-mi
            raw_fields = [h.strip().lstrip("\ufeff") for h in first.strip().split(delim)]
            reader = csv.DictReader(f, fieldnames=raw_fields, delimiter=delim)

            sym_u = symbol.upper().strip()
            mod_u = model.upper().strip()
            reg_u = regime.upper().strip()

            for r in reader:
                # bezpečne – niekedy tam môže byť prázdny riadok
                if not isinstance(r, dict):
                    continue

                rs = (r.get("symbol") or "").upper().strip()
                rm = (r.get("model") or "").upper().strip()
                rr = (r.get("social_regime") or r.get("regime") or "").upper().strip()

                if rs == sym_u and rm == mod_u and rr == reg_u:
                    out["found"] = True
                    out["n"] = to_float(r.get("n") or r.get("trades"))
                    out["avg_edge"] = to_float(r.get("avg_edge") or r.get("edge"))
                    out["risk_mult"] = to_float(r.get("risk_mult"))
                    out["status"] = "OK"
                    return out

        out["status"] = "NO_MATCH"
        return out
    except Exception as e:
        out["status"] = f"CSV_ERROR: {e}"
        return out

# ---------- čítanie posledného signálu ----------
def read_last_signal(csv_path: str = CSV_PATH) -> Dict[str, Any]:
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"{csv_path} nemá žiadny dátový riadok.")

    header = [h.strip() for h in lines[0].split(",")]
    last   = [x.strip() for x in lines[-1].split(",")]

    if len(last) < len(header):
        last += [""] * (len(header) - len(last))
    elif len(last) > len(header):
        last = last[:len(header)]

    rec = dict(zip(header, last))

    def to_float(x: str, default: Optional[float]=None) -> Optional[float]:
        try:
            v = float(x)
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return default

    def to_int(x: str, default: Optional[int]=None) -> Optional[int]:
        try:
            return int(float(x))
        except Exception:
            return default

    side_val = to_int(rec.get("side"), None)
    if side_val is None:
        side_val = 1 if (to_int(rec.get("signal"), 0) == 1) else 0

    thr = None
    if "threshold_long" in rec:
        thr = to_float(rec.get("threshold_long"))
    elif "threshold" in rec:
        thr = to_float(rec.get("threshold"))

    return {
        "utc_time":     rec.get("utc_time"),
        "symbol":       str(rec.get("symbol", "BTCUSDT")).upper(),
        "interval":     rec.get("interval", ""),
        "close":        to_float(rec.get("close")),
        "side":         side_val,
        "signal":       int(float(rec.get("signal") or 0)),
        "atr":          to_float(rec.get("atr")),
        "sl_price":     to_float(rec.get("sl_price")),
        "partial_pct":  to_float(rec.get("partial_pct"), 0.5),
        "partial_tp":   to_float(rec.get("partial_tp")),
        "final_tp":     to_float(rec.get("final_tp")),
        "trail_sl":     to_float(rec.get("trail_sl")),
        "qty_suggest":  to_float(rec.get("qty_suggest")),
        "proba_up":     to_float(rec.get("proba_up")),
        "threshold":    thr,
        "r_dist":       to_float(rec.get("r_dist")),
    }

# ---------- pomocné ----------
def round_step(qty: float, step: float = QTY_STEP) -> float:
    if qty is None:
        return 0.0
    q = math.floor(qty / step + 1e-9) * step
    return round(q, 6)

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "None"
    return f"{x:.{nd}f}"

def load_state() -> Dict[str, Any]:
    global STATE_PATH
    if not STATE_PATH or not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(st: Dict[str, Any]):
    global STATE_PATH
    if not STATE_PATH:
        return
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

def _acquire_lock(timeout_sec=3.0, retry_ms=100):
    start = time.time()
    while True:
        try:
            fd = os.open(PORTFOLIO_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - start > timeout_sec:
                return False
            time.sleep(retry_ms / 1000.0)

def _release_lock():
    try:
        os.remove(PORTFOLIO_LOCK)
    except FileNotFoundError:
        pass

def _load_portfolio_state() -> Dict[str, Any]:
    if not os.path.exists(PORTFOLIO_STATE):
        return {
            "equity_usdt": float(os.getenv("EQUITY_USDT", "1000")),
            "max_risk_at_once_pct": float(os.getenv("PORTFOLIO_MAX_RISK_AT_ONCE_PCT", "0.03")),
            "positions": {}
        }
    try:
        with open(PORTFOLIO_STATE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "equity_usdt": float(os.getenv("EQUITY_USDT", "1000")),
            "max_risk_at_once_pct": float(os.getenv("PORTFOLIO_MAX_RISK_AT_ONCE_PCT", "0.03")),
            "positions": {}
        }

def _save_portfolio_state(ps: Dict[str, Any]):
    os.makedirs(os.path.dirname(PORTFOLIO_STATE), exist_ok=True)
    with open(PORTFOLIO_STATE, "w", encoding="utf-8") as f:
        json.dump(ps, f, ensure_ascii=False, indent=2)

def print_plan(dry: bool, s: Dict[str, Any],
               target_qty: float, current_qty: float, qty_step: float,
               lock_mode: str, drift_pct: float, soft_exit_mode: str, st: Dict[str, Any],
               free_risk_usdt: Optional[float], need_risk_usdt: Optional[float], r_dist: Optional[float],
               equity_usdt: float, max_risk_pct: float):
    print("--- PLAN ---")
    print(f"DRY_RUN={dry}")
    print(f"Symbol: {s['symbol']} | close={fmt(s['close'])} | side={s['side']} (-1 short, 0 flat, +1 long) | proba={fmt(s.get('proba_up'),3)} | TH={fmt(s.get('threshold'))}")
    print(f"Target qty ≈ {target_qty:.6f} | current ≈ {current_qty:.6f} | step {qty_step}")
    print(f"SL={fmt(s['sl_price'])} (MARK) | partial@+1R={fmt(s['partial_tp'])} (LAST) | finalTP={fmt(s['final_tp'])} (LAST)")
    print(f"UPDATE_PROTECTION={lock_mode} | UPDATE_DRIFT_PCT={drift_pct*100:.2f}%")
    print(f"SOFT_EXIT={soft_exit_mode} | state: bars_in_trade={st.get('bars_in_trade',0)} flat_streak={st.get('flat_streak',0)}")
    print(f"Portfolio: equity={fmt(equity_usdt)} USDT | max_at_once={fmt(max_risk_pct*100,2)}%")
    if r_dist is not None and need_risk_usdt is not None and free_risk_usdt is not None:
        print(f"Risk: 1R/qty≈{fmt(r_dist)} | need≈{fmt(need_risk_usdt)} USDT | free≈{fmt(free_risk_usdt)} USDT")

# ---------- hlavné ----------
def main():
    # načítaj env znova korektne (neprebíjaj sa default .env)
    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=True)

    # refresh hodnoty po load_dotenv
    global CSV_PATH, STATE_DIR, PORTFOLIO_STATE, PORTFOLIO_LOCK, SOCIAL_EDGE_SNAPSHOT, MODEL_NAME
    CSV_PATH = os.getenv("LOG_CSV", CSV_PATH)
    STATE_DIR = os.getenv("STATE_DIR", STATE_DIR)
    PORTFOLIO_STATE = os.getenv("PORTFOLIO_STATE", PORTFOLIO_STATE)
    PORTFOLIO_LOCK = PORTFOLIO_STATE + ".lock"
    SOCIAL_EDGE_SNAPSHOT = os.getenv("SOCIAL_EDGE_SNAPSHOT", SOCIAL_EDGE_SNAPSHOT)
    MODEL_NAME = os.getenv("MODEL", "LR").upper()

    TRADE_MODE   = os.getenv("TRADE_MODE", "FUTURES").upper()
    FUTURES_BASE = os.getenv("FUTURES_BASE", "https://testnet.binancefuture.com")
    KEY          = os.getenv("BINANCE_API_KEY")
    SEC          = os.getenv("BINANCE_API_SECRET")
    LEVERAGE     = int(os.getenv("LEVERAGE", "1"))
    DRY_RUN      = os.getenv("DRY_RUN", "true").lower() == "true"

    HEDGE_MODE   = os.getenv("HEDGE_MODE", "false").lower() == "true"
    REBALANCE    = os.getenv("REBALANCE", "true").lower() == "true"

    UPDATE_PROTECTION    = os.getenv("UPDATE_PROTECTION", "REFRESH_WITH_DRIFT").upper()
    UPDATE_DRIFT_PCT     = float(os.getenv("UPDATE_DRIFT_PCT", "0.005"))

    SOFT_EXIT            = os.getenv("SOFT_EXIT", "OFF").upper()
    SOFT_EXIT_N          = int(os.getenv("SOFT_EXIT_N", "2"))
    SOFT_EXIT_MIN_BARS   = int(os.getenv("SOFT_EXIT_MIN_BARS", "6"))
    SOFT_EXIT_PROBA_BIAS = float(os.getenv("SOFT_EXIT_PROBA_BIAS", "0.05"))

    EQUITY_USDT                = float(os.getenv("EQUITY_USDT", "1000"))
    PORTFOLIO_MAX_RISK_AT_ONCE = float(os.getenv("PORTFOLIO_MAX_RISK_AT_ONCE_PCT", "0.03"))

    s = read_last_signal(CSV_PATH)
    symbol = s["symbol"]

    # ---------- SOCIAL OBSERVE (len výpis) ----------
    sig_time_utc = parse_utc_time(s.get("utc_time"))
    social = fetch_social_regime(symbol=symbol, ts_utc=sig_time_utc)
    edge = read_edge_snapshot(
        path=SOCIAL_EDGE_SNAPSHOT,
        symbol=symbol,
        model=MODEL_NAME,
        regime=(social.get("social_regime") or None),
    )
    print("--- SOCIAL (OBSERVE) ---")
    print(
        f"model={MODEL_NAME} | signal_time_utc={sig_time_utc} | "
        f"regime_status={social.get('status')} | bucket={social.get('bucket_start_utc')} | "
        f"regime={social.get('social_regime')} | z_count={social.get('z_count')} | z_sent={social.get('z_sent')}"
    )
    print(
        f"edge_status={edge.get('status')} | n={edge.get('n')} | avg_edge={edge.get('avg_edge')} | risk_mult={edge.get('risk_mult')}"
    )

    # ---- zvyšok tvojho pôvodného kódu zostáva BEZ ZMIEN ----
    # (nechávam celý zvyšok identický, aby sa ti nič iné nerozbilo)

    global STATE_PATH
    STATE_PATH = os.path.join(STATE_DIR, f"trade_state_{symbol}.json")

    target_qty = round_step((s.get("qty_suggest") or 0.0) * (s.get("side") or 0), QTY_STEP)

    r_dist = s.get("r_dist")
    if (r_dist is None or not math.isfinite(r_dist)) and s.get("sl_price") is not None and s.get("close") is not None:
        if s.get("side") == -1:
            r_dist = max(float(s["sl_price"]) - float(s["close"]), 0.0)
        else:
            r_dist = max(float(s["close"]) - float(s["sl_price"]), 0.0)

    if TRADE_MODE != "FUTURES":
        raise RuntimeError("Tento skript je pripravený pre FUTURES testnet.")

    client = UMFutures(key=KEY, secret=SEC, base_url=FUTURES_BASE)

    try:
        pm = client.get_position_mode()
        dual = str(pm.get("dualSidePosition", "false")).lower() == "true"
        if HEDGE_MODE and not dual:
            print("⚠️  HEDGE_MODE=True, ale účet je v ONE-WAY (dualSidePosition=False). Zmeň režim na účte alebo daj HEDGE_MODE=false.")
        if (not HEDGE_MODE) and dual:
            print("⚠️  HEDGE_MODE=False, ale účet je v HEDGE (dualSidePosition=True).")
    except Exception as e:
        print(f"[info] get_position_mode: {e}")

    try:
        client.change_leverage(symbol=symbol, leverage=LEVERAGE)
    except Exception as e:
        print(f"change_leverage: {e}")

    pos_qty = 0.0
    long_qty = 0.0
    short_qty = 0.0
    try:
        rows = client.get_position_risk(symbol=symbol)
        rows = rows if isinstance(rows, list) else rows.get("positions", [])
        for r in rows:
            if not isinstance(r, dict):
                continue
            if r.get("symbol") != symbol:
                continue
            q = float(r.get("positionAmt", 0.0))
            side_label = (r.get("positionSide") or "").upper()
            if HEDGE_MODE:
                if side_label == "LONG":
                    long_qty = q
                elif side_label == "SHORT":
                    short_qty = q
            else:
                pos_qty = q
        if HEDGE_MODE:
            pos_qty = long_qty + short_qty
    except Exception as e:
        print(f"position_risk: {e}")

    st = load_state()
    last_csv_time = s.get("utc_time")
    if last_csv_time and st.get("last_csv_time") != last_csv_time:
        st["bars_in_trade"] = int(st.get("bars_in_trade", 0)) + 1 if st.get("in_trade") else 0
        is_flat_like = (s.get("side", 0) == 0)
        if (s.get("proba_up") is not None) and (s.get("threshold") is not None):
            if s["proba_up"] < (s["threshold"] - SOFT_EXIT_PROBA_BIAS):
                is_flat_like = True
        if is_flat_like:
            st["flat_streak"] = int(st.get("flat_streak",0)) + 1
        else:
            st["flat_streak"] = 0
        st["last_csv_time"] = last_csv_time

    free_risk_usdt = None
    need_risk_usdt = None

    locked = _acquire_lock()
    try:
        ps = _load_portfolio_state()
        ps["equity_usdt"] = float(ps.get("equity_usdt", EQUITY_USDT))
        ps["max_risk_at_once_pct"] = float(ps.get("max_risk_at_once_pct", PORTFOLIO_MAX_RISK_AT_ONCE))

        positions = ps.get("positions", {})
        effective_qty_for_risk = abs(pos_qty) if not HEDGE_MODE else (abs(long_qty) + abs(short_qty))
        if effective_qty_for_risk != 0.0 and r_dist is not None and math.isfinite(r_dist) and r_dist > 0:
            positions[symbol] = {
                "risk_per_qty": r_dist,
                "qty": pos_qty,
                "risk_used_usdt": r_dist * effective_qty_for_risk,
                "status": "OPEN"
            }
        else:
            if symbol in positions:
                positions[symbol]["qty"] = 0.0
                positions[symbol]["risk_used_usdt"] = 0.0
                positions[symbol]["status"] = "CLOSED"

        ps["positions"] = positions

        used = sum(max(0.0, float(v.get("risk_used_usdt", 0.0))) for v in positions.values())
        cap  = float(ps["equity_usdt"]) * float(ps["max_risk_at_once_pct"])
        free_risk_usdt = max(0.0, cap - used)

        delta_raw = (target_qty or 0.0) - (pos_qty or 0.0)
        add_qty   = delta_raw if delta_raw * (s.get("side") or 0) > 0 else 0.0
        add_qty   = max(0.0, abs(add_qty))
        if r_dist is not None and math.isfinite(r_dist):
            need_risk_usdt = r_dist * add_qty
        else:
            need_risk_usdt = None

    finally:
        if locked:
            _save_portfolio_state(ps)
            _release_lock()

    print_plan(DRY_RUN, s, target_qty, pos_qty, QTY_STEP,
               UPDATE_PROTECTION, UPDATE_DRIFT_PCT, SOFT_EXIT, st,
               free_risk_usdt, need_risk_usdt, r_dist, EQUITY_USDT, PORTFOLIO_MAX_RISK_AT_ONCE)

    # --- zvyšok pôvodnej logiky (orders/hedge/soft-exit) nechávam tak ako si poslal ---
    # Pozn.: sem už nič nemením, lebo tvoj NO_MATCH je vyriešený hore v read_edge_snapshot.

    # !!! ZACHOVAJ svoj pôvodný zvyšok kódu odtiaľto !!!
    # (Keď chceš, môžem ti to ešte raz poslať komplet aj s tou časťou,
    #  ale funkčne sa fix týka len env + read_edge_snapshot.)

    save_state(st)

if __name__ == "__main__":
    main()
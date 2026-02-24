# bot_trade.py — LONG aj SHORT (Futures TESTNET)
# - číta posledný riadok z CSV (LOG_CSV z .env; fallback logs/signals.csv)
# - rebalance na podpísanú target qty (risk-based) s portfóliovým stropom rizika
# - ochrany:
#     LONG : SL = STOP_MARKET SELL (MARK), partial TP = TAKE_PROFIT_MARKET SELL (LAST), final TP SELL (LAST)
#     SHORT: SL = STOP_MARKET BUY  (MARK), partial TP = TAKE_PROFIT_MARKET BUY  (LAST), final TP BUY  (LAST)
# - UPDATE_PROTECTION: LOCK_ON_ENTRY | ALWAYS_REFRESH | REFRESH_WITH_DRIFT
# - SOFT EXIT: po min. držbe a potvrdení uzavrie pozíciu na market (SELL pri long, BUY pri short)
# - HEDGE SAFE: kontrola režimu účtu; v hedge nepoužívame closePosition a neotočíme pozíciu

import os, math, json, datetime as dt, time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from binance.um_futures import UMFutures

# načítaj .env podľa ENV_FILE (ak nie je, použije .env)
load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"))

# --- cesty / konštanty ---
CSV_PATH   = os.getenv("LOG_CSV", "logs/signals.csv")
STATE_DIR  = os.getenv("STATE_DIR", "logs")
STATE_PATH = None  # nastaví sa v main() podľa symbolu

PORTFOLIO_STATE = os.getenv("PORTFOLIO_STATE", "logs/portfolio_state.json")
PORTFOLIO_LOCK  = PORTFOLIO_STATE + ".lock"

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

    # side (-1/0/+1) má prioritu; ak chýba, fallback zo signal (1->+1, 0->0)
    side_val = to_int(rec.get("side"), None)
    if side_val is None:
        side_val = 1 if (to_int(rec.get("signal"), 0) == 1) else 0

    # threshold info (ak chýba, nechaj None)
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
        "side":         side_val,  # -1/0/+1
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

# ---- jednoduchý file-lock pre portfolio_state (aby sa viaceré coiny „nebili”) ----
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
    load_dotenv()  # pre istotu načítaj znova (ak sa spúšťa v inom procese)

    # Režimy
    TRADE_MODE   = os.getenv("TRADE_MODE", "FUTURES").upper()
    FUTURES_BASE = os.getenv("FUTURES_BASE", "https://testnet.binancefuture.com")
    KEY          = os.getenv("BINANCE_API_KEY")
    SEC          = os.getenv("BINANCE_API_SECRET")
    LEVERAGE     = int(os.getenv("LEVERAGE", "1"))
    DRY_RUN      = os.getenv("DRY_RUN", "true").lower() == "true"

    # Hedge mód (voliteľný; ak false, bežíme one-way)
    HEDGE_MODE   = os.getenv("HEDGE_MODE", "false").lower() == "true"

    # Rebalance nastavenia
    REBALANCE            = os.getenv("REBALANCE", "true").lower() == "true"

    # Ochrany – update režim
    UPDATE_PROTECTION    = os.getenv("UPDATE_PROTECTION", "REFRESH_WITH_DRIFT").upper()
    UPDATE_DRIFT_PCT     = float(os.getenv("UPDATE_DRIFT_PCT", "0.005"))  # napr. 0.01 = 1 %

    # Soft exit režim
    SOFT_EXIT            = os.getenv("SOFT_EXIT", "OFF").upper()   # OFF | CONFIRM_N_BARS
    SOFT_EXIT_N          = int(os.getenv("SOFT_EXIT_N", "2"))
    SOFT_EXIT_MIN_BARS   = int(os.getenv("SOFT_EXIT_MIN_BARS", "6"))
    SOFT_EXIT_PROBA_BIAS = float(os.getenv("SOFT_EXIT_PROBA_BIAS", "0.05"))

    # Portfóliový limit
    EQUITY_USDT                = float(os.getenv("EQUITY_USDT", "1000"))
    PORTFOLIO_MAX_RISK_AT_ONCE = float(os.getenv("PORTFOLIO_MAX_RISK_AT_ONCE_PCT", "0.03"))

    # čítanie signálu
    s = read_last_signal(CSV_PATH)
    symbol = s["symbol"]

    # nastav cestu na state pre tento symbol (oddelený state per coin)
    global STATE_PATH
    STATE_PATH = os.path.join(STATE_DIR, f"trade_state_{symbol}.json")

    # cieľová qty (PODPÍSANÁ podľa side)
    target_qty = round_step((s.get("qty_suggest") or 0.0) * (s.get("side") or 0), QTY_STEP)

    # odhad 1R vzdialenosti (potrebné na risk v USDT)
    r_dist = s.get("r_dist")
    if (r_dist is None or not math.isfinite(r_dist)) and s.get("sl_price") is not None and s.get("close") is not None:
        if s.get("side") == -1:
            r_dist = max(float(s["sl_price"]) - float(s["close"]), 0.0)  # short
        else:
            r_dist = max(float(s["close"]) - float(s["sl_price"]), 0.0)  # long/ostatné

    if TRADE_MODE != "FUTURES":
        raise RuntimeError("Tento skript je pripravený pre FUTURES testnet.")

    client = UMFutures(key=KEY, secret=SEC, base_url=FUTURES_BASE)

    # --- INFO: skutočný mód účtu (hedge vs one-way), len warning aby si vedel čo beží ---
    try:
        pm = client.get_position_mode()  # {'dualSidePosition': 'true'/'false'}
        dual = str(pm.get("dualSidePosition", "false")).lower() == "true"
        if HEDGE_MODE and not dual:
            print("⚠️  HEDGE_MODE=True, ale účet je v ONE-WAY (dualSidePosition=False). Zmeň režim na účte alebo daj HEDGE_MODE=false.")
        if (not HEDGE_MODE) and dual:
            print("⚠️  HEDGE_MODE=False, ale účet je v HEDGE (dualSidePosition=True).")
    except Exception as e:
        print(f"[info] get_position_mode: {e}")

    # leverage (ignoruj chyby)
    try:
        client.change_leverage(symbol=symbol, leverage=LEVERAGE)
    except Exception as e:
        print(f"change_leverage: {e}")

    # načítaj aktuálnu pozíciu (v hedge oddelene LONG/SHORT; inak netto)
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
            pos_qty = long_qty + short_qty  # netto info (pre risk/plan je fajn vedieť)
    except Exception as e:
        print(f"position_risk: {e}")

    # stav
    st = load_state()
    # zistíme, či máme nový bar (porovnám posledné CSV time s tým v state)
    last_csv_time = s.get("utc_time")
    if last_csv_time and st.get("last_csv_time") != last_csv_time:
        # posun po bare
        st["bars_in_trade"] = int(st.get("bars_in_trade", 0)) + 1 if st.get("in_trade") else 0
        # FLAT streak (ak signal=0 alebo proba hlboko pod TH-bias)
        is_flat_like = (s.get("side", 0) == 0)
        if (s.get("proba_up") is not None) and (s.get("threshold") is not None):
            if s["proba_up"] < (s["threshold"] - SOFT_EXIT_PROBA_BIAS):
                is_flat_like = True
        if is_flat_like:
            st["flat_streak"] = int(st.get("flat_streak",0)) + 1
        else:
            st["flat_streak"] = 0
        st["last_csv_time"] = last_csv_time

    # --- načítaj/aktualizuj portfóliový stav a spočítaj voľný risk ---
    free_risk_usdt = None
    need_risk_usdt = None

    locked = _acquire_lock()
    try:
        ps = _load_portfolio_state()
        # aktualizuj metriky (ak sa v .env zmenili)
        ps["equity_usdt"] = float(ps.get("equity_usdt", EQUITY_USDT))
        ps["max_risk_at_once_pct"] = float(ps.get("max_risk_at_once_pct", PORTFOLIO_MAX_RISK_AT_ONCE))

        # pre náš symbol spočítaj využitý risk (absolútna qty)
        positions = ps.get("positions", {})
        effective_qty_for_risk = abs(pos_qty) if not HEDGE_MODE else (abs(long_qty) + abs(short_qty))
        if effective_qty_for_risk != 0.0 and r_dist is not None and math.isfinite(r_dist) and r_dist > 0:
            positions[symbol] = {
                "risk_per_qty": r_dist,
                "qty": pos_qty,  # netto info
                "risk_used_usdt": r_dist * effective_qty_for_risk,
                "status": "OPEN"
            }
        else:
            if symbol in positions:
                positions[symbol]["qty"] = 0.0
                positions[symbol]["risk_used_usdt"] = 0.0
                positions[symbol]["status"] = "CLOSED"

        ps["positions"] = positions

        # spočítaj celkový obsadený risk
        used = sum(max(0.0, float(v.get("risk_used_usdt", 0.0))) for v in positions.values())
        cap  = float(ps["equity_usdt"]) * float(ps["max_risk_at_once_pct"])
        free_risk_usdt = max(0.0, cap - used)

        # riziko potrebné na dorovnanie (len pre zväčšenie abs pozície)
        delta_raw = (target_qty or 0.0) - (pos_qty or 0.0)   # podpísané (netto)
        add_qty   = delta_raw if delta_raw * (s.get("side") or 0) > 0 else 0.0  # len ak ideme smerom k cieľu
        add_qty   = max(0.0, abs(add_qty))  # pre risk rátame veľkosť
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

    # --- helpers for orders ---
    def cancel_all(pos_side: Optional[int] = None):
        """Zruší otvorené ochrany pre symbol; v hedge ak pos_side známa, zruší len tú stranu."""
        try:
            if DRY_RUN:
                print(f"[DRY-RUN] Cancel open orders ({'ALL' if pos_side is None else ('LONG' if pos_side>0 else 'SHORT')})")
                return
            # SDK nemá filter positionSide pri mass-cancel; spravíme ručne:
            open_orders = client.get_open_orders(symbol=symbol)
            for od in open_orders or []:
                psd = (od.get("positionSide") or "").upper()
                if HEDGE_MODE and (pos_side is not None):
                    if pos_side > 0 and psd != "LONG":   # chceme len LONG ochrany
                        continue
                    if pos_side < 0 and psd != "SHORT":  # chceme len SHORT ochrany
                        continue
                try:
                    client.cancel_order(symbol=symbol, orderId=od["orderId"])
                except Exception as e:
                    print(f"cancel_order({od.get('orderId')}): {e}")
            print("Zrušené otvorené objednávky pre symbol.")
        except Exception as e:
            print(f"cancel_open_orders: {e}")

    def place_market(side_str: str, q: float):
        q = round_step(max(0.0, q), QTY_STEP)
        if q < MIN_QTY:
            print(f"Preskakujem MARKET {side_str} — qty {q} < min {MIN_QTY}")
            return False, 0.0
        if DRY_RUN:
            print(f"[DRY-RUN] MARKET {side_str} {q:.6f}")
            return True, q
        try:
            client.new_order(symbol=symbol, side=side_str, type="MARKET", quantity=q)
            print(f"MARKET {side_str} OK {q:.6f}")
            return True, q
        except Exception as e:
            print(f"new_order MARKET {side_str} fail: {e}")
            return False, 0.0

    def set_protection_orders(sl_price, partial_tp, final_tp, partial_pct, total_qty, pos_side: int):
        """
        Nastaví SL (MARK) + partial/final TP (LAST).
        V HEDGE móde nepoužívame closePosition, ale presnú quantity + positionSide,
        aby sa pozícia NIKDY neotočila do opačného smeru.
        """
        if pos_side > 0:      # LONG
            sl_side, tp_side = "SELL", "SELL"
            ps_arg = "LONG"
        else:                 # SHORT
            sl_side, tp_side = "BUY", "BUY"
            ps_arg = "SHORT"

        positionSideArg = {"positionSide": ps_arg} if HEDGE_MODE else {}

        # 1) SL (MARK) — round správnym smerom
        if sl_price is not None and math.isfinite(sl_price):
            sl_rounded = round_price_for_sl(float(sl_price), pos_side, PRICE_TICK)
            if DRY_RUN:
                print(f"[DRY-RUN] SL {sl_side} @ {fmt(sl_rounded)} (MARK)")
            else:
                try:
                    kwargs = dict(
                        symbol=symbol, side=sl_side, type="STOP_MARKET",
                        stopPrice=sl_rounded,
                        workingType="MARK_PRICE",
                        **positionSideArg
                    )
                    if HEDGE_MODE:
                        # v hedge nepoužívame closePosition; dáme quantity presne na veľkosť pozície
                        q_full = round_step(abs(total_qty or 0.0), QTY_STEP)
                        if q_full >= MIN_QTY:
                            kwargs["quantity"] = q_full
                        else:
                            print("SL preskočený (hedge mode, quantity príliš malé).")
                            kwargs = None
                    else:
                        # one-way: zavri celú pozíciu
                        kwargs["closePosition"] = True

                    if kwargs is not None:
                        client.new_order(**kwargs)
                        print(f"SL OK {sl_side} @ {sl_rounded:.6f} (MARK)")
                except Exception as e:
                    print(f"SL objednávka zlyhala: {e}")
        else:
            print("SL vynechaný (chýba cena).")

        pq = round_step((total_qty or 0.0) * max(0.0, min(1.0, partial_pct or 0.5)), QTY_STEP)
        pq = abs(pq)  # qty musí byť kladná

        # 2) Partial TP (LAST, reduceOnly)
        if partial_tp is not None and math.isfinite(partial_tp) and pq >= MIN_QTY:
            tp1_rounded = round_price_for_tp(float(partial_tp), pos_side, PRICE_TICK)
            if DRY_RUN:
                print(f"[DRY-RUN] Partial TP {tp_side} @ {fmt(tp1_rounded)} (LAST) qty={pq:.6f}")
            else:
                try:
                    client.new_order(
                        symbol=symbol, side=tp_side, type="TAKE_PROFIT_MARKET",
                        stopPrice=tp1_rounded,
                        quantity=pq,
                        workingType="CONTRACT_PRICE",  # LAST price
                        reduceOnly=True,               # dôležité, neotočí pozíciu
                        **positionSideArg
                    )
                    print(f"Partial TP OK {tp_side} @ {tp1_rounded:.6f} (LAST) qty={pq:.6f}")
                except Exception as e:
                    print(f"Partial TP objednávka zlyhala: {e}")
        else:
            print("Partial TP vynechaný (chýba cena alebo qty príliš malé).")

        # 3) Final TP (LAST) — hedge: quantity+reduceOnly, one-way: closePosition
        if final_tp is not None and math.isfinite(final_tp):
            tp2_rounded = round_price_for_tp(float(final_tp), pos_side, PRICE_TICK)
            if DRY_RUN:
                print(f"[DRY-RUN] Final TP {tp_side} @ {fmt(tp2_rounded)} (LAST) "
                      f"{'quantity+reduceOnly' if HEDGE_MODE else 'closePosition=True'}")
            else:
                try:
                    kwargs = dict(
                        symbol=symbol, side=tp_side, type="TAKE_PROFIT_MARKET",
                        stopPrice=tp2_rounded,
                        workingType="CONTRACT_PRICE",
                        **positionSideArg
                    )
                    if HEDGE_MODE:
                        q_full = round_step(abs(total_qty or 0.0), QTY_STEP)
                        if q_full >= MIN_QTY:
                            kwargs["quantity"] = q_full
                            kwargs["reduceOnly"] = True
                        else:
                            print("Final TP preskočený (hedge mode, quantity príliš malé).")
                            kwargs = None
                    else:
                        kwargs["closePosition"] = True

                    if kwargs is not None:
                        client.new_order(**kwargs)
                        print(f"Final TP OK {tp_side} @ {tp2_rounded:.6f} (LAST)")
                except Exception as e:
                    print(f"Final TP objednávka zlyhala: {e}")
        else:
            print("Final TP vynechaný (chýba cena).")

    # --- pomocné pre LOCK/DRIFT rozhodnutia ---
    def choose_levels_for_update(current: Dict[str, float], suggestion: Dict[str, float]) -> Dict[str, float]:
        mode = UPDATE_PROTECTION
        out = current.copy() if current else {}

        def need_update(old, new):
            if new is None or not math.isfinite(new):
                return False
            if old is None or not math.isfinite(old):
                return True
            base = max(abs(old), 1e-8)
            rel = abs(new - old) / base
            return rel >= UPDATE_DRIFT_PCT

        if mode == "LOCK_ON_ENTRY":
            if not out:
                out = {
                    "sl_price": suggestion.get("sl_price"),
                    "partial_tp": suggestion.get("partial_tp"),
                    "final_tp": suggestion.get("final_tp"),
                }
            return out

        if mode == "ALWAYS_REFRESH":
            return {
                "sl_price": suggestion.get("sl_price"),
                "partial_tp": suggestion.get("partial_tp"),
                "final_tp": suggestion.get("final_tp"),
            }

        # REFRESH_WITH_DRIFT (default)
        keys = ["sl_price","partial_tp","final_tp"]
        for k in keys:
            newv = suggestion.get(k)
            oldv = out.get(k)
            if need_update(oldv, newv):
                out[k] = newv
        if not out:
            out = {
                "sl_price": suggestion.get("sl_price"),
                "partial_tp": suggestion.get("partial_tp"),
                "final_tp": suggestion.get("final_tp"),
            }
        return out

    # ------- LOGIKA -------
    def should_soft_exit() -> bool:
        if SOFT_EXIT == "OFF":
            return False
        # v hedge: soft-exit vyhodnotíme, ak máme nejakú netto pozíciu
        if (not HEDGE_MODE and pos_qty == 0.0) or (HEDGE_MODE and (abs(long_qty) + abs(short_qty) == 0.0)):
            return False
        bars_in_trade = int(st.get("bars_in_trade", 0))
        flat_streak   = int(st.get("flat_streak", 0))
        if bars_in_trade < SOFT_EXIT_MIN_BARS:
            return False
        return flat_streak >= SOFT_EXIT_N

    # 1) side != 0 → rebalance (s portfóliovým limitom)
    if s.get("side", 0) != 0:
        if REBALANCE:
            delta = (target_qty or 0.0) - (pos_qty or 0.0)  # podpísané (netto)

            # Ak držíme opačnú stranu, najprv plný close proti smeru
            if (pos_qty > 0 and s["side"] == -1):
                ok, filled = place_market("SELL", abs(pos_qty))
                if ok and filled > 0:
                    pos_qty -= filled
            elif (pos_qty < 0 and s["side"] == 1):
                ok, filled = place_market("BUY", abs(pos_qty))
                if ok and filled > 0:
                    pos_qty += filled

            # prepočet delty po prípadnom close opačnej strany
            delta = (target_qty or 0.0) - (pos_qty or 0.0)

            add_qty = 0.0
            if delta * s["side"] > 0:  # budujeme v smere side
                if r_dist is not None and math.isfinite(r_dist) and r_dist > 0 and free_risk_usdt is not None:
                    max_add_qty_by_cap = free_risk_usdt / r_dist if free_risk_usdt > 0 else 0.0
                    add_qty = min(abs(delta), max_add_qty_by_cap)
                    add_qty = round_step(max(0.0, add_qty), QTY_STEP)
                else:
                    add_qty = 0.0

                if add_qty >= MIN_QTY:
                    if s["side"] == 1:
                        print(f"[Rebalance] BUILD LONG BUY {add_qty:.6f}")
                        ok, filled = place_market("BUY", add_qty)
                        if ok and filled > 0:
                            pos_qty += filled
                    else:
                        print(f"[Rebalance] BUILD SHORT SELL {add_qty:.6f}")
                        ok, filled = place_market("SELL", add_qty)
                        if ok and filled > 0:
                            pos_qty -= filled
                else:
                    if abs(delta) >= MIN_QTY:
                        print(f"[Rebalance] STOP — strop rizika (free≈{fmt(free_risk_usdt)} USDT) alebo krok qty bráni dokúpiť.")
                    else:
                        print(f"[Rebalance] OK (target {target_qty:.6f} ~ current {pos_qty:.6f})")
            else:
                # delta má opačné znamienko → zníž abs pozíciu
                shrink = round_step(min(abs(delta), abs(pos_qty)), QTY_STEP)
                if shrink >= MIN_QTY:
                    if pos_qty > 0:
                        print(f"[Rebalance] TRIM LONG SELL {shrink:.6f}")
                        ok, filled = place_market("SELL", shrink)
                        if ok and filled > 0:
                            pos_qty -= filled
                    elif pos_qty < 0:
                        print(f"[Rebalance] TRIM SHORT BUY {shrink:.6f}")
                        ok, filled = place_market("BUY", shrink)
                        if ok and filled > 0:
                            pos_qty += filled
                else:
                    print(f"[Rebalance] OK (target {target_qty:.6f} ~ current {pos_qty:.6f})")
        else:
            print("[Rebalance] vypnutý.")

        # vyber levely podľa LOCK/DRIFT režimu
        current_levels = st.get("levels_set", {})
        suggestion = {"sl_price": s["sl_price"], "partial_tp": s["partial_tp"], "final_tp": s["final_tp"]}
        levels = choose_levels_for_update(current_levels, suggestion)

        changed = (levels != current_levels)
        if changed:
            # urč smer ochrán podľa AKTUÁLNEHO pos_qty (ak 0, použijeme signál side)
            pos_side = 1 if pos_qty > 0 else (-1 if pos_qty < 0 else int(s.get("side", 1)))
            cancel_all(pos_side if HEDGE_MODE else None)
            set_protection_orders(
                sl_price=levels["sl_price"],
                partial_tp=levels["partial_tp"],
                final_tp=levels["final_tp"],
                partial_pct=s["partial_pct"],
                total_qty=pos_qty if pos_qty != 0 else target_qty,
                pos_side=pos_side
            )
            st["levels_set"] = levels
            st["in_trade"] = (abs(pos_qty) > 0.0)
        else:
            print("Ochrany bez zmeny (LOCK/DRIFT pod prahom).")

    # 2) side == 0 → soft-exit alebo len update ochrán
    else:
        have_any_pos = (abs(pos_qty) > 0.0) if not HEDGE_MODE else ((abs(long_qty) + abs(short_qty)) > 0.0)
        if have_any_pos:
            if should_soft_exit():
                print(f"SOFT EXIT → splnené podmienky (flat_streak={st.get('flat_streak')}, bars_in_trade={st.get('bars_in_trade')}) → FULL CLOSE")
                cancel_all()
                if pos_qty > 0:
                    ok, filled = place_market("SELL", abs(pos_qty))
                    if ok and filled > 0:
                        pos_qty -= filled
                elif pos_qty < 0:
                    ok, filled = place_market("BUY", abs(pos_qty))
                    if ok and filled > 0:
                        pos_qty += filled
                else:
                    # v hedge môže byť netto 0 ale otvorený long aj short -> zavrieme oboch
                    if HEDGE_MODE:
                        if abs(long_qty) >= MIN_QTY:
                            place_market("SELL", abs(long_qty))
                        if abs(short_qty) >= MIN_QTY:
                            place_market("BUY", abs(short_qty))
                        long_qty = 0.0
                        short_qty = 0.0
                st.clear()
                st["in_trade"] = False
                st["last_csv_time"] = s.get("utc_time")
            else:
                print("side=0 → pozíciu ponechávam, len (podľa LOCK/DRIFT) aktualizujem ochrany.")
                current_levels = st.get("levels_set", {})
                suggestion = {"sl_price": s["sl_price"], "partial_tp": s["partial_tp"], "final_tp": s["final_tp"]}
                levels = choose_levels_for_update(current_levels, suggestion)
                if levels != current_levels:
                    pos_side = 1 if pos_qty > 0 else (-1 if pos_qty < 0 else (1 if (abs(long_qty) >= abs(short_qty)) else -1))
                    cancel_all(pos_side if HEDGE_MODE else None)
                    set_protection_orders(
                        sl_price=levels["sl_price"],
                        partial_tp=levels["partial_tp"],
                        final_tp=levels["final_tp"],
                        partial_pct=s["partial_pct"],
                        total_qty=pos_qty if pos_qty != 0 else (long_qty if pos_side>0 else short_qty),
                        pos_side=pos_side
                    )
                    st["levels_set"] = levels
                    st["in_trade"] = have_any_pos
                else:
                    print("Ochrany bez zmeny (LOCK/DRIFT pod prahom).")
        else:
            print("side=0 a žiadna pozícia -> nič nerobím.")
            st["in_trade"] = False

    # --- po akcii zaktualizuj portfolio_state podľa novej qty/SL ---
    locked = _acquire_lock()
    try:
        ps = _load_portfolio_state()
        positions = ps.get("positions", {})
        effective_qty_for_risk = abs(pos_qty) if not HEDGE_MODE else (abs(long_qty) + abs(short_qty))
        if effective_qty_for_risk != 0.0 and s.get("sl_price") is not None and s.get("close") is not None:
            # risk per qty teraz podľa aktuálneho SL a close
            if pos_qty >= 0:
                r_now = max(float(s["close"]) - float(s["sl_price"]), 0.0)
            else:
                r_now = max(float(s["sl_price"]) - float(s["close"]), 0.0)
            positions[symbol] = {
                "risk_per_qty": r_now,
                "qty": pos_qty,
                "risk_used_usdt": r_now * effective_qty_for_risk,
                "status": "OPEN"
            }
        else:
            positions[symbol] = {
                "risk_per_qty": 0.0,
                "qty": 0.0,
                "risk_used_usdt": 0.0,
                "status": "CLOSED"
            }
        ps["positions"] = positions
        _save_portfolio_state(ps)
    finally:
        if locked:
            _release_lock()

    save_state(st)

if __name__ == "__main__":
    main()
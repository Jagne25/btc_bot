"""
walk_forward.py — spustí backtest 8x na rôznych historických oknách
a vypíše súhrnnú tabuľku výsledkov pre LR aj RF.

Správny walk-forward: každé okno testuje INÉ historické obdobie.
Sviečky sa stiahnu raz (max dostupné) a každé okno vyreže svoju časť.

Spustenie:
    ENV_FILE=env python walk_forward.py
"""

import os        # práca so súborovým systémom a env premennými
import re        # regex — hľadanie textu v outpute backtestu
import subprocess  # spustenie iného python skriptu ako podproces
import sys       # sys.executable = cesta k aktuálnemu python interpreteru
import requests  # stiahnutie sviečok z Binance API

# --- konfigurácia okien ---
FORWARD      = 3000    # každé okno testuje na 3000 sviečkach dopredu
VAL          = 1000    # validačná sada (na výber TH)
MIN_TRAIN    = 8000    # minimálny počet sviečok na tréning
MAX_DD_LIMIT = -12.0   # okno neprešlo ak MaxDD je horšie ako -12%
MIN_TRADES   = 20      # okno neprešlo ak menej ako 20 obchodov

ENV_FILE = os.getenv("ENV_FILE", ".env")  # načítaj ENV_FILE z prostredia


def load_env(path):
    """Načítaj env súbor a vráť dict hodnôt."""
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def fetch_all_klines(symbol, interval, max_candles=20000):
    """Stiahni čo najviac sviečok z Binance API (od najstaršej po najnovšiu)."""
    url = "https://api.binance.com/api/v3/klines"
    out, end_time = [], None
    while len(out) < max_candles:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        out.extend(chunk)
        oldest_open_ms = chunk[0][0]
        end_time = oldest_open_ms - 1
        if len(out) >= max_candles:
            break
    return out


def run_backtest(lookback, end_ms, model):
    """
    Spustí backtest_sltp.py pre konkrétne okno.
    end_ms = unix timestamp (ms) poslednej sviečky okna
    Backtest stiahne lookback sviečok končiace na tomto čase.
    """
    env = os.environ.copy()       # skopíruj aktuálne env premenné
    env["ENV_FILE"]   = ENV_FILE  # nastav ktorý .env súbor použiť
    env["LOOKBACK"]   = str(lookback)  # veľkosť okna
    env["FORWARD"]    = str(FORWARD)   # FWD časť
    env["VAL"]        = str(VAL)       # VAL časť
    env["MODEL"]      = model          # LR alebo RF
    env["WF_END_MS"]  = str(int(end_ms)) if end_ms else ""  # unix ms koniec okna

    result = subprocess.run(
        [sys.executable, "backtest_sltp.py"],  # spusti backtest
        capture_output=True,
        text=True,
        env=env
    )
    return result.stdout


def parse_best(output, model):
    """
    Nájde výsledok pre daný model v outpute backtestu.
    Vráti (vynos, maxdd, trades) alebo (None, None, None).
    """
    pattern = rf"MODEL {model}.*?BEST na VAL:.*?FWD: vynos ([-\d.]+)% \| MaxDD ([-\d.]+)% \| obchody (\d+)"
    m = re.search(pattern, output, re.DOTALL)
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def parse_buyhold(output):
    """Nájde Buy&Hold výsledok v outpute."""
    m = re.search(r"Buy&Hold close->close: ([-\d.]+)%", output)
    return float(m.group(1)) if m else None


def passes(vynos, maxdd, trades):
    """Vráti True ak okno splnilo všetky podmienky."""
    if vynos is None:
        return False
    return vynos > 0 and maxdd > MAX_DD_LIMIT and trades >= MIN_TRADES


# --- načítaj env ---
cfg = load_env(ENV_FILE)
SYMBOL   = cfg.get("SYMBOL", "BTCUSDT")
INTERVAL = cfg.get("INTERVAL", "4h")

# --- interval v ms ---
INTERVAL_MS = {
    "1h": 3600_000, "2h": 7200_000, "4h": 14400_000,
    "6h": 21600_000, "8h": 28800_000, "12h": 43200_000,
    "1d": 86400_000, "3d": 259200_000, "1w": 604800_000
}.get(INTERVAL, 14400_000)  # default 4h

# --- zisti čas poslednej sviečky z Binance ---
print("Sťahujem čas poslednej sviečky...", flush=True)
_r = requests.get(
    "https://api.binance.com/api/v3/klines",
    params={"symbol": SYMBOL, "interval": INTERVAL, "limit": 1},
    timeout=20
)
_r.raise_for_status()
LATEST_END_MS = _r.json()[0][6]  # close_time poslednej sviečky v ms

# --- vypočítaj okná ---
# okno 8 (najnovšie): končí na LATEST_END_MS
# okno 7: končí o FORWARD sviečok skôr
# okno 1 (najstaršie): končí o 7*FORWARD sviečok skôr
NUM_WINDOWS = 8
LOOKBACK    = MIN_TRAIN + VAL + FORWARD  # fixná veľkosť každého okna

windows = []
for i in range(NUM_WINDOWS):
    offset_candles = (NUM_WINDOWS - 1 - i) * FORWARD  # o koľko sviečok je okno posunuté dozadu
    end_ms = LATEST_END_MS - offset_candles * INTERVAL_MS  # unix ms koniec okna
    windows.append((LOOKBACK, end_ms, offset_candles))

# --- hlavička ---
print(f"\n{'='*72}")
print(f"WALK-FORWARD  |  FORWARD={FORWARD} sviečok  |  VAL={VAL}  |  TRAIN~{MIN_TRAIN}")
print(f"Limit: MaxDD > {MAX_DD_LIMIT}%  |  Min trades: {MIN_TRADES}")
print(f"{'='*72}\n")

header = f"{'okno':>4} | {'end_offset':>10} | {'B&H':>7} | {'model':>4} | {'FWD vynos':>10} | {'MaxDD':>8} | {'trades':>6} | {'pass?':>5}"
print(header)
print("-" * len(header))

summary = {"LR": [], "RF": []}

for i, (lb, end_ms, offset) in enumerate(windows, 1):
    print(f"  >> Spúšťam okno {i}/{NUM_WINDOWS}: offset={offset} sviečok dozadu ...", flush=True)

    bh_printed = False
    for model in ["LR", "RF"]:
        output = run_backtest(lb, end_ms, model)
        vynos, maxdd, trades = parse_best(output, model)
        bh = parse_buyhold(output) if not bh_printed else None
        p  = "✓" if passes(vynos, maxdd, trades) else "✗"
        summary[model].append(passes(vynos, maxdd, trades))

        v_str  = f"{vynos:+.2f}%" if vynos  is not None else "    N/A"
        d_str  = f"{maxdd:.2f}%"  if maxdd  is not None else "    N/A"
        t_str  = str(trades)       if trades is not None else "N/A"
        bh_str = f"{bh:+.1f}%"    if bh     is not None else "    N/A"

        print(f"{i:>4} | {offset:>10} | {bh_str:>7} | {model:>4} | {v_str:>10} | {d_str:>8} | {t_str:>6} | {p:>5}")
        bh_printed = True

# --- súhrn ---
print(f"\n{'='*72}")
print("SÚHRN:")
for model in ["LR", "RF"]:
    passed  = sum(summary[model])
    total   = len(summary[model])
    verdict = "✓ PREŠIEL" if passed >= int(total * 0.65) else "✗ NEPREŠIEL"
    print(f"  {model}: {passed}/{total} okien pozitívnych  →  {verdict}")
print(f"{'='*72}\n")

# observe_spike_returns.py ukazuje:THRESHOLD a SPIKES koľko krát sme TH použili 4,5,6,7,8 a SPIKES kolko krat sa spomenulo za posledne 2 týždne
#
# Cieľ:
# - nájsť "spikes" v social_count_total (napr. >= 4, 5, 6, 7, 8)
# - zistiť, o koľko % sa cena pohla po 4h, 8h, 12h, 24h
# - spraviť prehľadnú tabuľku za každý symbol

from pathlib import Path
import pandas as pd

# ----- KONFIGURÁCIA -----

BASE_DIR = Path(r"C:\btc_bot")
LOGS_DIR = BASE_DIR / "logs"

INTERVAL = "4h"               # zhoduje sa s observe CSV
BAR_HOURS = 4                 # jedna sviečka = 4 hodiny
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

# prahy, ktoré chceme testovať (spikes)
THRESHOLDS = [4, 5, 6, 7, 8]

# horizonty, po koľkých hodinách rátame výnosy
HORIZONS_HOURS = [4, 8, 12, 24]


def load_observe_df(symbol: str) -> pd.DataFrame | None:
    """Načíta observe CSV pre daný symbol, vráti DataFrame alebo None."""
    csv_path = LOGS_DIR / f"observe_{symbol}_{INTERVAL}_LR.csv"
    if not csv_path.exists():
        print(f"\n[{symbol}] CSV neexistuje, preskakujem: {csv_path}")
        return None

    df = pd.read_csv(csv_path, parse_dates=["close_time"])
    # uistíme sa, že sú tam potrebné stĺpce
    needed = ["close_time", "close", "social_count_total"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[{symbol}] Chýbajú stĺpce: {missing} – preskakujem.")
        return None

    # zoradíme podľa času
    df = df.sort_values("close_time").reset_index(drop=True)
    return df


def add_future_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Do DF pridá stĺpce:
      ret_4h, ret_8h, ret_12h, ret_24h  (výnos v %)
    Výnos = (close_future / close_now - 1) * 100
    """
    df = df.copy()

    for h in HORIZONS_HOURS:
        steps = h // BAR_HOURS  # koľko 4h sviečok je daný horizont
        future_close = df["close"].shift(-steps)
        ret_pct = (future_close / df["close"] - 1.0) * 100.0
        col_name = f"ret_{h}h"
        df[col_name] = ret_pct

    return df


def analyze_spikes_for_symbol(symbol: str) -> None:
    """Pre daný symbol vypočíta štatistiky výnosov po spikoch."""
    df = load_observe_df(symbol)
    if df is None:
        return

    # pridáme budúce výnosy
    df = add_future_returns(df)

    # zoberieme len posledných ~14 dní, kde social_count_total > 0
    last_time = df["close_time"].max()
    cutoff_time = last_time - pd.Timedelta(days=14)
    mask_base = (df["close_time"] >= cutoff_time) & (df["social_count_total"] > 0)
    df_recent = df[mask_base].copy()

    if df_recent.empty:
        print(f"\n[{symbol}] Žiadne dáta za posledných 14 dní so social_count_total > 0.")
        return

    print(f"\n========== {symbol} ==========")
    print(f"Počet riadkov v okne (14 dní, social_count_total > 0): {len(df_recent)}")

    # tabuľka výsledkov
    rows = []

    for thr in THRESHOLDS:
        # spike = social_count_total >= threshold
        df_spikes = df_recent[df_recent["social_count_total"] >= thr].copy()

        # nesmie byť NaN v budúcich výnosoch (koniec série)
        for h in HORIZONS_HOURS:
            df_spikes = df_spikes[df_spikes[f"ret_{h}h"].notna()]

        count_spikes = len(df_spikes)
        if count_spikes == 0:
            # žiadny spike pre daný threshold
            row = {"threshold": thr, "spikes": 0}
            for h in HORIZONS_HOURS:
                row[f"avg_ret_{h}h"] = None
            rows.append(row)
            continue

        # spočítame priemerné výnosy
        result_row = {
            "threshold": thr,
            "spikes": count_spikes,
        }
        for h in HORIZONS_HOURS:
            col = f"ret_{h}h"
            avg_ret = df_spikes[col].mean()
            result_row[f"avg_ret_{h}h"] = avg_ret

        rows.append(result_row)

    result_df = pd.DataFrame(rows)

    # rozumnejšie zaokrúhlenie - najprv pretypujeme na čísla
    for h in HORIZONS_HOURS:
        col = f"avg_ret_{h}h"
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce").round(3)

    print("\nVýnosy po spikoch (v %):")
    print(result_df.to_string(index=False))


def main() -> None:
    for symbol in SYMBOLS:
        analyze_spikes_for_symbol(symbol)


if __name__ == "__main__":
    main()
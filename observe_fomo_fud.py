# observe_fomo_fud.py   pre mna dôležite ukazuje FUD - cena padla mi nakupujeme lebo čakama že cena pojde hore FOMO cena išla hore mi skorej pradavame lebo čakame že cena padne
#
# Cieľ:
# - nájsť social spiky (social_count_total >= threshold)
# - zistiť, čo robila cena 12h PRED spikom (FOMO vs FUD)
# - zvlášť spočítať výnosy po FOMO spikoch a po FUD spikoch
#
# FOMO = cena 12h pred spikom rástla (pre_ret_12h > 0)
# FUD  = cena 12h pred spikom padala (pre_ret_12h < 0)

from pathlib import Path
import pandas as pd

# ----- KONFIGURÁCIA -----

BASE_DIR = Path(r"C:\btc_bot")
LOGS_DIR = BASE_DIR / "logs"

INTERVAL = "4h"         # zhoduje sa s observe CSV
BAR_HOURS = 4           # jedna sviečka = 4 hodiny
PRE_HOURS = 12          # okno pred spikom = 12h (3 sviečky)
SYMBOLS = ["BTCUSDT"]   # kludne pridáš ETHUSDT, BNBUSDT, SOLUSDT neskôr

THRESHOLDS = [4, 5, 6, 7, 8]       # úrovne social spiku
HORIZONS_HOURS = [4, 8, 12, 24]    # kam pozeráme dopredu


def load_observe_df(symbol: str) -> pd.DataFrame | None:
    """Načíta observe CSV pre daný symbol, vráti DataFrame alebo None."""
    csv_path = LOGS_DIR / f"observe_{symbol}_{INTERVAL}_LR.csv"
    if not csv_path.exists():
        print(f"\n[{symbol}] CSV neexistuje, preskakujem: {csv_path}")
        return None

    df = pd.read_csv(csv_path, parse_dates=["close_time"])
    needed = ["close_time", "close", "social_count_total"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"[{symbol}] Chýbajú stĺpce: {missing} – preskakujem.")
        return None

    df = df.sort_values("close_time").reset_index(drop=True)
    return df


def add_pre_and_future_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pridá:
      - pre_ret_12h  (pohyb ceny za 12h PRED aktuálnym barom, v %)
      - ret_4h, ret_8h, ret_12h, ret_24h (výnosy PO bare, v %)
    """
    df = df.copy()

    # 1) PRED spikom: 12h = 3 bary dozadu
    bars_back = PRE_HOURS // BAR_HOURS
    pre_close = df["close"].shift(bars_back)  # close 3 bary dozadu
    df["pre_ret_12h"] = (df["close"] / pre_close - 1.0) * 100.0

    # 2) PO spiku: budúce výnosy
    for h in HORIZONS_HOURS:
        steps = h // BAR_HOURS  # koľko 4h sviečok je daný horizont
        future_close = df["close"].shift(-steps)
        col_name = f"ret_{h}h"
        df[col_name] = (future_close / df["close"] - 1.0) * 100.0

    return df


def analyze_fomo_fud_for_symbol(symbol: str) -> None:
    """Pre daný symbol spraví FOMO/FUD analýzu social spike-ov."""
    df = load_observe_df(symbol)
    if df is None:
        return

    df = add_pre_and_future_returns(df)

    # Zoberieme len posledných 14 dní, kde máme social dáta aj pre_ret_12h
    last_time = df["close_time"].max()
    cutoff_time = last_time - pd.Timedelta(days=14)
    mask_recent = (
        (df["close_time"] >= cutoff_time)
        & (df["social_count_total"] > 0)
        & (df["pre_ret_12h"].notna())
    )
    df_recent = df[mask_recent].copy()

    if df_recent.empty:
        print(f"\n[{symbol}] Žiadne dáta v posledných 14 dňoch pre FOMO/FUD analýzu.")
        return

    print(f"\n========== {symbol} ==========")
    print(f"Počet riadkov v okne (14 dní, social_count_total > 0): {len(df_recent)}")

    fomo_rows = []
    fud_rows = []

    for thr in THRESHOLDS:
        # social spike = social_count_total >= threshold
        df_spikes = df_recent[df_recent["social_count_total"] >= thr].copy()

        # odstránime riadky, kde chýbajú budúce výnosy (koniec série)
        for h in HORIZONS_HOURS:
            df_spikes = df_spikes[df_spikes[f"ret_{h}h"].notna()]

        if df_spikes.empty:
            # žiadne spiky pre daný threshold
            row_fomo = {"threshold": thr, "spikes": 0}
            row_fud = {"threshold": thr, "spikes": 0}
            for h in HORIZONS_HOURS:
                row_fomo[f"avg_ret_{h}h"] = None
                row_fud[f"avg_ret_{h}h"] = None
            fomo_rows.append(row_fomo)
            fud_rows.append(row_fud)
            continue

        # Rozdelenie na FOMO (pre_ret_12h > 0) a FUD (pre_ret_12h < 0)
        df_fomo = df_spikes[df_spikes["pre_ret_12h"] > 0].copy()
        df_fud = df_spikes[df_spikes["pre_ret_12h"] < 0].copy()

        # FOMO výsledky
        if df_fomo.empty:
            row_fomo = {"threshold": thr, "spikes": 0}
            for h in HORIZONS_HOURS:
                row_fomo[f"avg_ret_{h}h"] = None
        else:
            row_fomo = {
                "threshold": thr,
                "spikes": len(df_fomo),
            }
            for h in HORIZONS_HOURS:
                col = f"ret_{h}h"
                row_fomo[f"avg_ret_{h}h"] = df_fomo[col].mean()

        # FUD výsledky
        if df_fud.empty:
            row_fud = {"threshold": thr, "spikes": 0}
            for h in HORIZONS_HOURS:
                row_fud[f"avg_ret_{h}h"] = None
        else:
            row_fud = {
                "threshold": thr,
                "spikes": len(df_fud),
            }
            for h in HORIZONS_HOURS:
                col = f"ret_{h}h"
                row_fud[f"avg_ret_{h}h"] = df_fud[col].mean()

        fomo_rows.append(row_fomo)
        fud_rows.append(row_fud)

    fomo_df = pd.DataFrame(fomo_rows)
    fud_df = pd.DataFrame(fud_rows)

    # zaokrúhlenie – pretypujeme na čísla, None -> NaN
    for h in HORIZONS_HOURS:
        col = f"avg_ret_{h}h"
        fomo_df[col] = pd.to_numeric(fomo_df[col], errors="coerce").round(3)
        fud_df[col] = pd.to_numeric(fud_df[col], errors="coerce").round(3)

    print("\nFOMO spiky (pre_ret_12h > 0) – priemerné výnosy v %:")
    print(fomo_df.to_string(index=False))

    print("\nFUD spiky (pre_ret_12h < 0) – priemerné výnosy v %:")
    print(fud_df.to_string(index=False))


def main() -> None:
    for symbol in SYMBOLS:
        analyze_fomo_fud_for_symbol(symbol)


if __name__ == "__main__":
    main()
# observe_analysis.py tu máme grafy, ukazuje nam ako to vyzera koľko je hluku,a ako ide cena

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----- KONFIG -----
BASE_DIR = Path(r"C:\btc_bot")
LOGS_DIR = BASE_DIR / "logs"

INTERVAL = "4h"  # zhoduje sa s názvom observe CSV
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]  # ktoré chceš analyzovať


def analyze_symbol(symbol: str) -> None:
    """Načíta observe CSV pre daný symbol a spraví graf za posledných 14 dní."""

    csv_path = LOGS_DIR / f"observe_{symbol}_{INTERVAL}_LR.csv"
    if not csv_path.exists():
        print(f"\n[{symbol}] CSV neexistuje, preskakujem: {csv_path}")
        return

    print(f"\n=== {symbol} ===")
    print("Načítavam CSV z:", csv_path)

    # 1) Načítanie CSV + parsovanie dátumu
    df = pd.read_csv(csv_path, parse_dates=["close_time"])

    # 2) Vyberieme len stĺpce, ktoré teraz potrebujeme
    cols = ["close_time", "close", "social_count_total", "pos"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[{symbol}] Chýbajú stĺpce: {missing} – preskakujem.")
        return

    df_small = df[cols]

    # 3) Nastavíme časové okno: posledných 14 dní
    last_time = df_small["close_time"].max()
    cutoff_time = last_time - pd.Timedelta(days=14)

    # 4) Filter:
    #    - posledných 14 dní
    #    - len záznamy, kde social_count_total > 0 (naozaj máme social dáta)
    mask = (df_small["close_time"] >= cutoff_time) & (df_small["social_count_total"] > 0)
    df_recent = df_small[mask].sort_values("close_time")

    if df_recent.empty:
        print(f"[{symbol}] Žiadne dáta za posledných 14 dní so social_count_total > 0.")
        return

    print("\nPosledné riadky po filtrovaní:")
    print(df_recent.tail())

    # 5) Graf: cena (čiara) + social_count_total (stĺpce)
    fig, ax1 = plt.subplots()

    ax1.plot(df_recent["close_time"], df_recent["close"])
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"{symbol} close price")

    ax2 = ax1.twinx()
    ax2.bar(df_recent["close_time"], df_recent["social_count_total"], alpha=0.3)
    ax2.set_ylabel("social_count_total (3h buckets)")

    plt.title(f"{symbol} close vs. social_count_total (posledných 14 dní)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    for symbol in SYMBOLS:
        analyze_symbol(symbol)


if __name__ == "__main__":
    main()
from pathlib import Path
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import psycopg2
from dotenv import load_dotenv


# ----- KONFIGURÁCIA -----

BASE_DIR = Path(r"C:\btc_bot")
LOGS_DIR = BASE_DIR / "logs"

INTERVAL = "4h"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
]

BAR_HOURS = 4
PRE_HOURS = 12          # okno, podľa ktorého určujeme FOMO/FUD (price-based)
SPIKE_THRESHOLD = 5     # od tejto hodnoty social_count_total je to "spike"
LOOKBACK_DAYS = 14      # koľko dní dozadu kreslíme

SMA_SHORT = 50          # tvoj doterajší trend
SMA_LONG = 200          # dlhodobý trend
EMA_FAST = 8            # rýchle momentum
EMA_MED = 21            # stredné momentum

# --- DB overlay (social-based) ---
USE_SOCIAL_DB_OVERLAY = True  # ak nechceš DB overlay, daj False

load_dotenv()

PGCFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5433")),
    dbname=os.getenv("PGDATABASE", "trading"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", ""),
)


# ----- HELPERS -----

def ensure_utc(s: pd.Series) -> pd.Series:
    """
    Zabezpečí, že datetime je tz-aware v UTC.
    Ak je tz-naive, berieme ho ako UTC (typicky tvoje close_time býva UTC).
    """
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def for_plot_time(s_utc: pd.Series) -> pd.Series:
    """Matplotlibu často vyhovuje tz-naive datetime."""
    return s_utc.dt.tz_convert(None)


def fetch_social_regime(symbol: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """
    Stiahne social režim z VIEW public.social_regime_3h pre daný symbol a časový rozsah.
    Vracia DF: bucket_start_utc, z_count, z_sent, social_regime
    """
    sql = """
    SELECT bucket_start_utc, z_count, z_sent, social_regime
    FROM public.social_regime_3h
    WHERE symbol = %s
      AND bucket_start_utc >= %s
      AND bucket_start_utc <= %s
    ORDER BY bucket_start_utc;
    """
    with psycopg2.connect(**PGCFG) as conn:
        df = pd.read_sql(sql, conn, params=(symbol, start_utc, end_utc))

    df["bucket_start_utc"] = ensure_utc(df["bucket_start_utc"])
    return df


# ----- LOAD & FEATURE FUNKCIE -----

def load_observe_df(symbol: str) -> pd.DataFrame | None:
    """Načíta observe CSV pre daný symbol, vráti DataFrame alebo None."""
    csv_path = LOGS_DIR / f"observe_{symbol}_{INTERVAL}_LR.csv"
    print(f"\nNačítavam CSV pre {symbol} z: {csv_path}")

    if not csv_path.exists():
        print(f"CSV neexistuje, preskakujem: {csv_path}")
        return None

    df = pd.read_csv(csv_path, parse_dates=["close_time"])
    needed = ["close_time", "open", "high", "low", "close", "social_count_total"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Chýbajú stĺpce: {missing} – preskakujem.")
        return None

    df = df.sort_values("close_time").reset_index(drop=True)
    return df


def add_pre_ret_12h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pridá stĺpec pre_ret_12h:
    - percentuálny pohyb ceny za 12h PRED aktuálnym barom.
    """
    df = df.copy()
    bars_back = PRE_HOURS // BAR_HOURS  # 12h = 3 bary dozadu pri 4h
    pre_close = df["close"].shift(bars_back)
    df["pre_ret_12h"] = (df["close"] / pre_close - 1.0) * 100.0
    return df


def add_ma_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Pridá SMA/EMA iba pre graf."""
    df = df.copy()
    close = df["close"]

    df[f"sma{SMA_SHORT}"] = close.rolling(SMA_SHORT).mean()
    df[f"sma{SMA_LONG}"] = close.rolling(SMA_LONG).mean()

    df[f"ema{EMA_FAST}"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df[f"ema{EMA_MED}"] = close.ewm(span=EMA_MED, adjust=False).mean()

    return df


# ----- KRESLENIE CANDLESTICK -----

def plot_candles(ax, df: pd.DataFrame):
    times = mdates.date2num(for_plot_time(df["close_time_utc"]))

    for t, o, h, l, c in zip(times, df["open"], df["high"], df["low"], df["close"]):
        color = "green" if c >= o else "red"
        ax.vlines(t, l, h, color=color, linewidth=1)
        ax.vlines(t, o, c, color=color, linewidth=4)


# ----- Hlavná logika pre jeden symbol -----

def plot_symbol(df: pd.DataFrame, symbol: str) -> None:
    df = add_pre_ret_12h(df)
    df = add_ma_ema(df)

    df = df.copy()
    df["close_time_utc"] = ensure_utc(df["close_time"])

    last_time = df["close_time_utc"].max()
    cutoff_time = last_time - pd.Timedelta(days=LOOKBACK_DAYS)

    mask = (
        (df["close_time_utc"] >= cutoff_time)
        & (df["social_count_total"] > 0)
        & (df["pre_ret_12h"].notna())
    )
    df_recent = df[mask].copy()

    if df_recent.empty:
        print(f"{symbol}: Žiadne dáta v posledných dňoch pre FOMO/FUD graf.")
        return

    spike_mask = df_recent["social_count_total"] >= SPIKE_THRESHOLD
    fomo_mask = spike_mask & (df_recent["pre_ret_12h"] > 0)
    fud_mask = spike_mask & (df_recent["pre_ret_12h"] < 0)

    df_fomo = df_recent[fomo_mask]
    df_fud = df_recent[fud_mask]

    last_row = df_recent.iloc[-1]
    last_pre_ret = last_row["pre_ret_12h"]
    last_social = int(last_row["social_count_total"])

    if last_pre_ret > 0:
        current_context = "FOMO (cena posledných 12h rástla)"
    elif last_pre_ret < 0:
        current_context = "FUD (cena posledných 12h padala)"
    else:
        current_context = "NEUTRAL (žiadny výrazný pohyb za 12h)"

    print(f"\n=== {symbol} – aktuálny kontext (price-based) ===")
    print(f"close_time:   {last_row['close_time_utc']}")
    print(f"close:        {last_row['close']:.2f}")
    print(f"pre_ret_12h:  {last_pre_ret:.3f} %")
    print(f"social_cnt:   {last_social}")
    print(f"Kontext:      {current_context}")
    print(f"FOMO spiky (social >= {SPIKE_THRESHOLD}): {len(df_fomo)}")
    print(f"FUD  spiky (social >= {SPIKE_THRESHOLD}): {len(df_fud)}")

    # --- DB OVERLAY: social-based režim ---
    if USE_SOCIAL_DB_OVERLAY:
        try:
            start_utc = df_recent["close_time_utc"].min() - pd.Timedelta(hours=6)
            end_utc = df_recent["close_time_utc"].max() + pd.Timedelta(hours=6)
            df_soc = fetch_social_regime(symbol, start_utc, end_utc)

            df_recent = df_recent.sort_values("close_time_utc")
            df_soc = df_soc.sort_values("bucket_start_utc")

            df_recent = pd.merge_asof(
                df_recent,
                df_soc,
                left_on="close_time_utc",
                right_on="bucket_start_utc",
                direction="backward"
            )

            # ✅ DEBUG: ukáž, čo sa naozaj prenieslo do df_recent
            print(f"[DB overlay] {symbol}: režimy v df_recent:")
            print(df_recent["social_regime"].value_counts(dropna=False))

        except Exception as e:
            print(f"[WARN] Social DB overlay sa nepodaril: {e}")
            df_recent["z_count"] = np.nan
            df_recent["z_sent"] = np.nan
            df_recent["social_regime"] = None
    else:
        df_recent["z_count"] = np.nan
        df_recent["z_sent"] = np.nan
        df_recent["social_regime"] = None

    times_plot = for_plot_time(df_recent["close_time_utc"])

    fig, (ax_price, ax_social, ax_context) = plt.subplots(
        3, 1, sharex=True, figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    plot_candles(ax_price, df_recent)

    ax_price.plot(times_plot, df_recent[f"sma{SMA_SHORT}"], label=f"SMA{SMA_SHORT}", linestyle="--", alpha=0.8)
    ax_price.plot(times_plot, df_recent[f"sma{SMA_LONG}"], label=f"SMA{SMA_LONG}", linestyle=":", alpha=0.8)
    ax_price.plot(times_plot, df_recent[f"ema{EMA_FAST}"], label=f"EMA{EMA_FAST}", alpha=0.8)
    ax_price.plot(times_plot, df_recent[f"ema{EMA_MED}"], label=f"EMA{EMA_MED}", alpha=0.8)

    ax_price.scatter(
        for_plot_time(df_fomo["close_time_utc"]),
        df_fomo["close"],
        label="FOMO spiky (price)",
        marker="^",
        color="red",
        zorder=5,
    )
    for _, row in df_fomo.iterrows():
        ax_price.text(
            for_plot_time(pd.Series([row["close_time_utc"]])).iloc[0],
            row["high"],
            "FOMO\nTOP",
            fontsize=8,
            color="red",
            ha="center",
            va="bottom",
        )

    ax_price.scatter(
        for_plot_time(df_fud["close_time_utc"]),
        df_fud["close"],
        label="FUD spiky (price)",
        marker="v",
        color="green",
        zorder=5,
    )
    for _, row in df_fud.iterrows():
        ax_price.text(
            for_plot_time(pd.Series([row["close_time_utc"]])).iloc[0],
            row["low"],
            "FUD\nBOTTOM",
            fontsize=8,
            color="green",
            ha="center",
            va="top",
        )

    # ✅ SOCIAL-BASED markery (DEBUG: veľké + texty)
    if USE_SOCIAL_DB_OVERLAY and "social_regime" in df_recent.columns:
        df_social_fomo = df_recent[df_recent["social_regime"] == "FOMO"]
        df_social_fud  = df_recent[df_recent["social_regime"] == "FUD"]
        df_social_buzz = df_recent[df_recent["social_regime"] == "BUZZ"]

        ax_price.scatter(for_plot_time(df_social_fomo["close_time_utc"]), df_social_fomo["high"],
                         marker="o", s=120, label="FOMO (social)", zorder=20)
        ax_price.scatter(for_plot_time(df_social_fud["close_time_utc"]), df_social_fud["low"],
                         marker="o", s=120, label="FUD (social)", zorder=20)
        ax_price.scatter(for_plot_time(df_social_buzz["close_time_utc"]), df_social_buzz["close"],
                         marker="o", s=80, label="BUZZ (social)", zorder=20)

        for _, r in df_social_fomo.iterrows():
            ax_price.text(
                for_plot_time(pd.Series([r["close_time_utc"]])).iloc[0],
                r["high"],
                "S-FOMO",
                fontsize=9,
                ha="center",
                va="bottom",
                zorder=21
            )
        for _, r in df_social_fud.iterrows():
            ax_price.text(
                for_plot_time(pd.Series([r["close_time_utc"]])).iloc[0],
                r["low"],
                "S-FUD",
                fontsize=9,
                ha="center",
                va="top",
                zorder=21
            )
        for _, r in df_social_buzz.iterrows():
            ax_price.text(
                for_plot_time(pd.Series([r["close_time_utc"]])).iloc[0],
                r["close"],
                "S-BUZZ",
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=21
            )

    ax_price.set_ylabel(f"{symbol} price")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, linestyle="--", alpha=0.3)

    textstr = (
        f"AKTUÁLNE (price-based):\n"
        f"12h zmena: {last_pre_ret:.2f} %\n"
        f"social:    {last_social}\n"
        f"{current_context}"
    )
    ax_price.text(
        0.99, 0.99, textstr,
        transform=ax_price.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax_social.bar(times_plot, df_recent["social_count_total"], width=0.03, align="center")
    ax_social.axhline(SPIKE_THRESHOLD, linestyle="--", alpha=0.7)
    ax_social.set_ylabel("social_count_total")
    ax_social.grid(True, linestyle="--", alpha=0.3)

    context_values = []
    for val in df_recent["pre_ret_12h"]:
        if pd.isna(val):
            context_values.append(0.0)
        elif val > 0:
            context_values.append(1.0)
        elif val < 0:
            context_values.append(-1.0)
        else:
            context_values.append(0.0)

    context_values = pd.Series(context_values, index=times_plot)

    fomo_bar_mask = context_values > 0
    fud_bar_mask = context_values < 0

    ax_context.bar(context_values.index[fomo_bar_mask], context_values[fomo_bar_mask],
                   width=0.03, align="center", color="red",
                   label="FOMO kontext (price pre_ret_12h > 0)")
    ax_context.bar(context_values.index[fud_bar_mask], context_values[fud_bar_mask],
                   width=0.03, align="center", color="green",
                   label="FUD kontext (price pre_ret_12h < 0)")

    ax_context.set_ylim(-1.5, 1.5)
    ax_context.set_yticks([-1, 0, 1])
    ax_context.set_yticklabels(["FUD", "", "FOMO"])
    ax_context.grid(True, linestyle="--", alpha=0.3)
    ax_context.set_xlabel("time")
    ax_context.legend(loc="upper left")

    ax_context.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_context.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax_context.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle(
        f"{symbol} – cena, social spiky a FOMO/FUD (price-based) + social_regime (DB)\n"
        f"Posledných {LOOKBACK_DAYS} dní, SPIKE_THRESHOLD = {SPIKE_THRESHOLD}, PRE_HOURS = {PRE_HOURS}\n"
        f"SMA{SMA_SHORT}, SMA{SMA_LONG}, EMA{EMA_FAST}, EMA{EMA_MED}"
    )
    plt.tight_layout()
    plt.show()


def main() -> None:
    for symbol in SYMBOLS:
        df = load_observe_df(symbol)
        if df is None:
            continue
        plot_symbol(df, symbol)


if __name__ == "__main__":
    main()
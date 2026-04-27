"""
PyTorch B1 — neurónová sieť na pullback systém
Label: 1 = pullback obchod vyhral, 0 = prehral
Otázka: dokáže sieť predpovedať či konkrétny pullback obchod vyhrá?
"""

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

from features import build_features

FEATURE_COLS = [
    "ret1", "ret5", "vol10",
    "rsi14", "zscore20", "atr_pct",
    "trend_slope", "above_slow",
    "pos_in_range_50", "vol_rel_20",
    "macd_hist", "adx14", "plus_di", "minus_di"
]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# ── 1. DÁTA ──────────────────────────────────────────────────────────────────

def fetch_data(symbol="BTCUSDT", interval="4h", limit=10950):
    url = "https://api.binance.com/api/v3/klines"
    rows, end_time = [], None
    while len(rows) < limit:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        r = requests.get(url, params=params)
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        end_time = chunk[0][0] - 1
        if len(rows) >= limit:
            break
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("open_time").drop_duplicates(subset="open_time").reset_index(drop=True)
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df

# ── 2. BACKTEST INDIKÁTORY ────────────────────────────────────────────────────
# Tieto indikátory sú len pre vstupné podmienky backttestu (nie pre features)
# Používame _bt suffix aby sme predišli konfliktu s features.py

def add_bt_indicators(df):
    df["ma20"]  = df["close"].rolling(20).mean()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    df["ma200_slope"] = (df["ma200"] - df["ma200"].shift(5)) / df["ma200"] * 100

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"]       = tr.rolling(14).mean()
    df["atr_pct_bt"] = df["atr"] / df["close"] * 100
    df["dist_ma20"] = (df["close"] - df["ma20"]) / df["ma20"] * 100

    high_diff = df["high"].diff()
    low_diff  = -df["low"].diff()
    plus_dm   = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm  = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr_adx   = tr.ewm(span=14, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_adx
    minus_di  = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_adx
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx14_bt"] = dx.ewm(span=14, adjust=False).mean()

    return df

# ── 3. BACKTEST (Experiment 4 — najlepšie parametre) ─────────────────────────

def run_backtest(df, trail_atr=3.0):
    trades = []
    in_trade = False

    for i in range(2, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i-1]

        if not in_trade:
            trend_up      = row["close"] > row["ma200"]
            ma200_rising  = row["ma200_slope"] > 0
            adx_ok        = row["adx14_bt"] > 25
            dist          = row["dist_ma20"]
            pullback_ok   = (dist < -1.0) and (dist > -8.0)
            volatility_ok = row["atr_pct_bt"] < 5.0
            breakout      = row["close"] > prev["high"]

            if trend_up and ma200_rising and pullback_ok and volatility_ok and breakout and adx_ok:
                in_trade    = True
                entry_price = row["close"]
                entry_time  = row["open_time"]
                entry_i     = i
                trail_stop  = entry_price - trail_atr * row["atr"]

        else:
            new_stop = row["close"] - trail_atr * row["atr"]
            if new_stop > trail_stop:
                trail_stop = new_stop

            if row["close"] < trail_stop:
                pnl = (row["close"] - entry_price) / entry_price * 100
                trades.append({"entry": entry_time, "pnl": pnl})
                in_trade = False

    return pd.DataFrame(trades) if trades else pd.DataFrame(columns=["entry","pnl"])

# ── 4. BUILD DATASET ─────────────────────────────────────────────────────────
# Pre každý coin: spusti backtest → zisti entry dátumy → vytiahni features

def build_dataset(symbols):
    all_rows = []

    for symbol in symbols:
        print(f"  {symbol}...")
        df_raw = fetch_data(symbol)

        # Dva df: jeden pre features, jeden pre backtest podmienky
        df_feat = build_features(df_raw.copy())
        df_feat = df_feat.set_index("open_time")

        df_bt = add_bt_indicators(df_raw.copy())
        df_bt = df_bt.dropna().reset_index(drop=True)

        trades = run_backtest(df_bt)
        if len(trades) == 0:
            print(f"    Žiadne obchody.")
            continue

        matched = 0
        for _, trade in trades.iterrows():
            if trade["entry"] not in df_feat.index:
                continue
            feat_row = df_feat.loc[trade["entry"]]
            if any(pd.isna(feat_row[col]) for col in FEATURE_COLS):
                continue
            all_rows.append({
                "entry":  trade["entry"],
                "symbol": symbol,
                "label":  1 if trade["pnl"] > 0 else 0,
                **{col: float(feat_row[col]) for col in FEATURE_COLS}
            })
            matched += 1

        print(f"    {len(trades)} obchodov → {matched} s features")

    return pd.DataFrame(all_rows)

# ── 5. NEURÓNOVÁ SIEŤ ─────────────────────────────────────────────────────────

class BTCNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            # Bez Sigmoid — BCEWithLogitsLoss ho zahrnie vnútri (numericky stabilnejšie)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ── 6. TRÉNING ────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, input_size, epochs=300):
    model = BTCNet(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # pos_weight = n_prehier / n_výhier → chyba na výhre sa počíta viac
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f}")

    return model

# ── 7. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Buduje dataset (4 coiny × pullback backtest × features)...")
    dataset = build_dataset(SYMBOLS)
    dataset = dataset.sort_values("entry").reset_index(drop=True)

    n_total = len(dataset)
    n_wins  = int(dataset["label"].sum())
    winrate = n_wins / n_total * 100
    baseline = 100 - winrate  # accuracy ak vždy tipuješ "0" (prehra)

    print(f"\nDataset: {n_total} obchodov")
    print(f"  Výhry (1): {n_wins} ({winrate:.1f}%)")
    print(f"  Prehry (0): {n_total - n_wins} ({100-winrate:.1f}%)")
    print(f"  Baseline (vždy tipuj prehra): {baseline:.1f}%")

    n = len(dataset)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)

    train = dataset.iloc[:train_end]
    val   = dataset.iloc[train_end:val_end]
    fwd   = dataset.iloc[val_end:]
    print(f"\nSplit: train={len(train)} | val={len(val)} | forward={len(fwd)}")

    X_train = train[FEATURE_COLS].values.astype(float)
    y_train = train["label"].values.astype(float)
    X_val   = val[FEATURE_COLS].values.astype(float)
    y_val   = val["label"].values.astype(float)
    X_fwd   = fwd[FEATURE_COLS].values.astype(float)
    y_fwd   = fwd["label"].values.astype(float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_fwd   = scaler.transform(X_fwd)

    print("\n── Logistická Regresia ──")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_val = accuracy_score(y_val, lr.predict(X_val)) * 100
    lr_fwd = accuracy_score(y_fwd, lr.predict(X_fwd)) * 100
    print(f"  Val: {lr_val:.1f}%  |  Forward: {lr_fwd:.1f}%")

    print("\n── Neurónová Sieť (PyTorch) ──")
    model = train_model(X_train, y_train, input_size=len(FEATURE_COLS))

    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_fwd_t = torch.tensor(X_fwd, dtype=torch.float32)
        nn_val = accuracy_score(y_val, (torch.sigmoid(model(X_val_t)).numpy() > 0.5).astype(int)) * 100
        nn_fwd = accuracy_score(y_fwd, (torch.sigmoid(model(X_fwd_t)).numpy() > 0.5).astype(int)) * 100
    print(f"  Val: {nn_val:.1f}%  |  Forward: {nn_fwd:.1f}%")

    lr_fwd_pred = lr.predict(X_fwd)
    lr_prec = precision_score(y_fwd, lr_fwd_pred, zero_division=0) * 100
    lr_rec  = recall_score(y_fwd, lr_fwd_pred, zero_division=0) * 100

    with torch.no_grad():
        nn_fwd_pred = (torch.sigmoid(model(X_fwd_t)).numpy() > 0.5).astype(int)
    nn_prec = precision_score(y_fwd, nn_fwd_pred, zero_division=0) * 100
    nn_rec  = recall_score(y_fwd, nn_fwd_pred, zero_division=0) * 100

    print(f"\n{'Model':<20} {'Acc':>6} {'Prec(win)':>10} {'Recall(win)':>12}")
    print("─" * 52)
    print(f"{'Logistická Regresia':<20} {lr_fwd:>5.1f}% {lr_prec:>9.1f}% {lr_rec:>11.1f}%")
    print(f"{'Neurónová Sieť':<20} {nn_fwd:>5.1f}% {nn_prec:>9.1f}% {nn_rec:>11.1f}%")
    print(f"\n  Baseline accuracy: {baseline:.1f}%")
    print(f"  Precision(win): keď model povie výhra, ako často má pravdu")
    print(f"  Recall(win):    z reálnych výhier, koľko model našiel")

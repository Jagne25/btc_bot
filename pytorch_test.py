"""
PyTorch Test — jednoduchá neurónová sieť na BTC 4h dáta
Otázka: dokáže sieť predpovedať či cena porastie za 40 sviečok?
Porovnanie s LR z backtest_sltp.py
"""

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from features import build_features

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
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("open_time").drop_duplicates(subset="open_time").reset_index(drop=True)
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df

# ── 2. LABEL ─────────────────────────────────────────────────────────────────

def make_label(df, horizon=40):
    # Label = 1 ak cena za 40 sviečok vyššia ako teraz, inak 0
    # shift(-40) = pozrieme sa 40 sviečok dopredu
    future_close = df["close"].shift(-horizon)
    label = (future_close > df["close"]).astype(int)
    return label

# ── 3. FEATURES ───────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "ret1", "ret5", "vol10",
    "rsi14", "zscore20", "atr_pct",
    "trend_slope", "above_slow",
    "pos_in_range_50", "vol_rel_20",
    "macd_hist", "adx14", "plus_di", "minus_di"
]

# ── 4. NEURÓNOVÁ SIEŤ ─────────────────────────────────────────────────────────

class BTCNet(nn.Module):
    """
    Jednoduchá sieť: vstup → skrytá vrstva 1 → skrytá vrstva 2 → výstup
    Výstup = pravdepodobnosť 0-1 (cena porastie)
    """
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),   # vrstva 1: input → 32 neurónov (menšia sieť)
            nn.ReLU(),
            nn.Dropout(0.3),             # 30% neurónov vypnutých → menej overfitting
            nn.Linear(32, 16),           # vrstva 2: 32 → 16 neurónov
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),            # výstup: 16 → 1 číslo
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# ── 5. TRÉNING ────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, input_size, epochs=300):
    model = BTCNet(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()  # Binary Cross Entropy — štandard pre 0/1 predikciu

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {loss.item():.4f}")

    return model

# ── 6. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Sťahujem dáta BTC 4h...")
    df = fetch_data()
    df = build_features(df)
    df["label"] = make_label(df, horizon=40)
    df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)

    print(f"Dáta: {len(df)} sviečok | od {df['open_time'].iloc[0].date()} do {df['open_time'].iloc[-1].date()}")
    print(f"Label rozdelenie: 1={df['label'].sum()} ({df['label'].mean()*100:.1f}%) | 0={len(df)-df['label'].sum()}")

    # ── TRAIN / VAL / FORWARD SPLIT ──
    # Rovnaký princíp ako v backtest_sltp.py — chronologické delenie
    n = len(df)
    train_end = int(n * 0.60)   # 60% train
    val_end   = int(n * 0.80)   # 20% val
    # zvyšok    = 20% forward

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    fwd   = df.iloc[val_end:]

    print(f"\nSplit: train={len(train)} | val={len(val)} | forward={len(fwd)}")

    X_train = train[FEATURE_COLS].values
    y_train = train["label"].values
    X_val   = val[FEATURE_COLS].values
    y_val   = val["label"].values
    X_fwd   = fwd[FEATURE_COLS].values
    y_fwd   = fwd["label"].values

    # Normalizácia — rovnako ako StandardScaler v backtest_sltp.py
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_fwd   = scaler.transform(X_fwd)

    # ── LOGISTICKÁ REGRESIA (porovnanie) ──
    print("\n── Logistická Regresia ──")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_val_acc = accuracy_score(y_val, lr.predict(X_val)) * 100
    lr_fwd_acc = accuracy_score(y_fwd, lr.predict(X_fwd)) * 100
    print(f"  Val accuracy:     {lr_val_acc:.1f}%")
    print(f"  Forward accuracy: {lr_fwd_acc:.1f}%")

    # ── NEURÓNOVÁ SIEŤ ──
    print("\n── Neurónová Sieť (PyTorch) ──")
    model = train_model(X_train, y_train, input_size=len(FEATURE_COLS))

    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_fwd_t = torch.tensor(X_fwd, dtype=torch.float32)
        val_pred  = (model(X_val_t).numpy() > 0.5).astype(int)
        fwd_pred  = (model(X_fwd_t).numpy() > 0.5).astype(int)

    nn_val_acc = accuracy_score(y_val, val_pred) * 100
    nn_fwd_acc = accuracy_score(y_fwd, fwd_pred) * 100
    print(f"  Val accuracy:     {nn_val_acc:.1f}%")
    print(f"  Forward accuracy: {nn_fwd_acc:.1f}%")

    # ── POROVNANIE ──
    print("\n── Porovnanie ──")
    print(f"{'Model':<20} {'Val':>8} {'Forward':>10}")
    print("─" * 40)
    print(f"{'Logistická Regresia':<20} {lr_val_acc:>7.1f}% {lr_fwd_acc:>9.1f}%")
    print(f"{'Neurónová Sieť':<20} {nn_val_acc:>7.1f}% {nn_fwd_acc:>9.1f}%")
    print()
    print("  Baseline (náhoda): 50.0%")
    print("  Ak forward > 52% → možný edge")

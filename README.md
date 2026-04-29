# BTC Algorithmic Trading Bot

## Overview
Automated cryptocurrency trading system built from scratch over 7 months. Implements a trend-following pullback strategy with machine learning experiments on BTC, ETH, SOL and BNB.

## Strategy
Rule-based pullback system:
- Entry when price is above MA200 (uptrend)
- ADX > 25 (strong trend filter)
- Price 1-8% below MA20 (pullback zone)
- Breakout above previous candle high (entry trigger)
- Exit via 3xATR trailing stop

## Backtest Results (4h timeframe, 5 years)

| Coin | EV | Return | Max Drawdown |
|------|----|--------|--------------|
| BTC | +1.49% | +58.1% | -22.5% |
| ETH | +0.83% | +39.1% | -32.1% |
| SOL | +2.91% | +136.6% | -48.4% |
| Combined | +1.22% | +223.8% | -66.4% |

*EV = average return per trade*

## Tech Stack
Python, pandas, numpy, scikit-learn, PyTorch, PostgreSQL, Binance API, VPS (Hetzner)

## Architecture
- `bot.py` — signal detection (runs every 4h via cron on VPS)
- `bot_trade.py` — order execution on Binance Futures
- `features.py` — feature engineering (RSI, MACD, ADX, ATR, volume)
- `trend_pullback_test.py` — multi-coin backtest engine
- `backtest_sltp.py` — ML-based backtest with LR/RF

## ML Experiments
- Logistic Regression and Random Forest for signal prediction
- PyTorch neural network with B1 labeling (trade outcome prediction)
- Regime analysis (trend/chop/neutral via ADX)
- Result: rule-based system outperformed ML — edge is in RR ratio (2.3x), not prediction

## Live System
- Runs on Hetzner VPS (Helsinki)
- Monitors BTC, ETH, SOL, BNB every 4 hours
- Telegram notifications for every signal
- Binance Futures with trailing stop management

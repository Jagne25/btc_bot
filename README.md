# BTC Algorithmic Trading Bot

Automated cryptocurrency trading system built from scratch. Implements a trend-following pullback strategy with machine learning experiments and a news sentiment pipeline, running live on Binance Futures.

## Backtest Results (4h timeframe, 5 years)

| Coin | EV per trade | Total Return | Max Drawdown |
|------|-------------|--------------|--------------|
| BTC  | +1.49%      | +58.1%       | -22.5%       |
| ETH  | +0.83%      | +39.1%       | -32.1%       |
| SOL  | +2.91%      | +136.6%      | -48.4%       |
| Combined | +1.22% | +223.8%      | -66.4%       |

*EV = average return per trade*

## Strategy

Rule-based pullback system:
- Entry when price is above MA200 (uptrend confirmed)
- ADX > 25 (strong trend filter)
- Price 1–8% below MA20 (pullback zone)
- Breakout above previous candle high (entry trigger)
- Exit via 3×ATR trailing stop

## Architecture

```
News Scraper → PostgreSQL → Feature Engineering → Signal Model → Binance Futures
```

- `bot.py` — main signal loop (runs every 4h via cron on VPS)
- `bot_trade.py` — order execution on Binance Futures with trailing stop management
- `features.py` — feature engineering (RSI, MACD, ADX, ATR, volume, sentiment z-score)
- `scraper_news.py` — RSS news scraper → stores headlines + sentiment tags to PostgreSQL
- `db_mirror.py` — syncs VPS PostgreSQL to local for analysis
- `walk_forward.py` — walk-forward validation engine

## ML Experiments

- Logistic Regression and Random Forest for signal prediction
- PyTorch neural network with B1 labeling (trade outcome prediction)
- Regime analysis: trend / chop / neutral via ADX
- Social sentiment pipeline: RSS scraping → keyword bucketization → z-score vs rolling baseline

**Result:** rule-based system outperformed ML — edge is in risk/reward ratio (2.3×), not in prediction accuracy. ML experiments documented and included for reference.

## Analysis Scripts

- `regime_analysis.py` — market regime classification
- `mean_reversion_test.py` — mean-reversion property testing
- `trend_pullback_test.py` — multi-coin backtest engine
- `observe_fomo_fud.py` — FOMO/FUD sentiment spikes vs price reaction
- `social_zscore_count.py`, `social_zscore_sent.py` — sentiment z-score tracking

## Live System

- Hetzner VPS (Helsinki)
- Monitors BTC, ETH, SOL, BNB every 4 hours
- Telegram notifications for every signal
- Scheduled tasks via cron (Linux VPS) and `.bat` runners (Windows dev machine)

## Stack

Python, pandas, NumPy, PyTorch, scikit-learn, psycopg2, feedparser, Binance Futures API, PostgreSQL

## Security

All API keys and secrets are loaded via environment variables (`os.getenv`). No credentials are stored in the repository. Environment files are excluded via `.gitignore`.

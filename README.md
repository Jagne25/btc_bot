# BTC Order Flow Research

Research and tooling for **BTC/USDT market microstructure** — live order-flow capture, hypothesis testing, and reproducible backtests. This repository documents an ongoing data analysis project: measuring aggressive trade flow, resting liquidity imbalance, absorption, and large-trade events, then evaluating them against explicit baselines.

> **Not trading advice.** Results below are observational findings from logged and historical data. Sample sizes, market regimes, and overlapping signals matter; treat every win-rate as provisional until re-validated on larger independent windows.

---

## Key concepts

| Concept | What it measures | Data source |
|--------|------------------|-------------|
| **CVD (Cumulative Volume Delta)** | Running sum of aggressive buy volume minus aggressive sell volume (taker flow) | Binance `aggTrades` (`buyer_is_maker` → aggressor side) |
| **Queue imbalance (IMB)** | Relative depth at the top of book: `(bid_vol − ask_vol) / (bid_vol + ask_vol)` | Binance depth / `@depth10@100ms` |
| **Absorption** | Large CVD move over a short window while price barely moves — aggressive flow absorbed by resting liquidity | CVD + mid/last price over ~20s |
| **Large Order Detection** | Trades that are outliers vs recent activity (relative threshold, not a fixed BTC size) | Rolling average of recent trade sizes |

**Context rule (exploratory):** CVD direction (print-to-print) aligned with IMB sign → “strong” BUY/SELL context; opposing signs → caution. Used for logging and testing, not as an automated strategy.

---

## Methodology

Hypotheses are tested with the same discipline across scripts:

1. **Define the event** (e.g. CVD–price divergence, strong BUY/SELL context transition, absorption entry).
2. **Forward horizon** — typically price after 60s / 5–10 minutes (first available timestamp within a tolerance).
3. **Baseline** — same horizon on every valid row (“always long” / “always short”), so signal win-rate is compared in percentage points, not in isolation.
4. **Independence** — consecutive identical context rows are **not** counted as separate signals; only **transitions** (first row where context differs from the previous) enter the event set.
5. **Regime check** — day-by-day (or period) splits to see whether apparent edge is just market drift (e.g. mostly down days in the live sample).
6. **Threshold calibration** — absorption and large-order cutoffs are set from empirical distributions on logged data (e.g. ~p90 \|ΔCVD\|), analogous to using an extreme IMB alert level.

Live series are stored on external SSD storage and optionally in **PostgreSQL** (`order_flow_context_log`); chart/backtest artefacts stay off the Mac internal disk when `BASE_SSD_DIR` / WORK_SSD is mounted.

---

## Selected findings (data results)

Findings are from scripts in this repo on Binance BTCUSDT; numbers will change as more live data accumulates.

### CVD divergence (historical klines)

- Strong **bullish** divergence (price down, CVD up) on a 90-day 1m sample: overall success ~**59%** over a 10-minute forward window vs baseline “always long” ~**49%** (~**+10 pp** edge in that sample).
- Edge varied by sub-period (weaker in quieter/down regimes, stronger in others) — regime matters.

### Live context log (CVD + IMB transitions)

- On transition-based events (~60s forward), **SELL** context often printed higher success than **BUY** in early live logs.
- Day-level breakdown showed most collection days were **net down**; on those days SELL looked stronger and BUY weaker. On a scarce up day the pattern flipped. **Interpretation:** a large part of the BUY/SELL gap was market drift in the sample, not a proven permanent asymmetry.

### Absorption (reconstructed from logged CVD + price)

- At **+1 minute**, post-event \|Δprice\| was close to baseline.
- At **+5 minutes**, bullish absorption (strong CVD up, price held) was more often followed by **lower** prices; bearish absorption more often by **higher** prices — consistent with a fade-of-absorbed-pressure story, but **sample size remains small**.

These are research checkpoints, not production signals.

---

## Tech stack

- **Python 3** — capture, analytics, backtests  
- **pandas / NumPy** — time-aligned merges, statistics  
- **PostgreSQL** (`psycopg2`) — optional structured log (`order_flow_context_log`, klines helpers)  
- **Binance** — REST (aggTrades, depth, klines) + WebSocket combined streams  
- **matplotlib** — CVD/price charts (SSD export)  
- Related repo experiments also use **PyTorch** / scikit-learn (separate from the current OF live loop)

---

## Repository structure (order-flow focus)

```
order_flow_week1.py              # aggTrades CVD, snapshot/live, divergence, relative large-order alert
order_flow_queue.py              # order-book imbalance (snapshot + live depth)
order_flow_context.py            # live CVD + IMB + absorption + large orders → CSV / PostgreSQL
order_flow_context_backtest.py   # BUY/SELL context transitions vs 60s baseline
order_flow_context_by_day.py     # day-level market drift vs BUY/SELL success
order_flow_absorption_backtest.py# absorption entries vs 60s / 5m baseline
order_flow_backtest.py           # historical CVD divergence (klines) + period split
```

Other files in the repo (`bot.py`, sentiment scrapers, earlier ML/backtest utilities) belong to a broader BTC research/trading codebase; the modules above are the dedicated **order-flow research track**.

**Typical live collector:**

```bash
python3 order_flow_context.py
```

**Typical analyses:**

```bash
python3 order_flow_context_backtest.py
python3 order_flow_context_by_day.py
python3 order_flow_absorption_backtest.py
python3 order_flow_backtest.py
```

Secrets (API keys, DB passwords) load from environment variables / `.env`. They are **not** committed — see `.gitignore`.

---

## Security

- `.env`, `.env.*`, credential/secret filenames, and common key material patterns are gitignored.
- Prefer `os.getenv` / env files for `PG*` and exchange credentials.
- Do not commit database dumps, private keys, or filled `.env` files.

---

## Future work

This project is **in active development**, not a finished product. Planned directions:

1. **LLM context summary** — natural-language digest of recent CVD / IMB / absorption / large-order state for research notes (not automated execution).
2. **PyTorch / ML fusion** — supervised or ranking models that combine CVD, imbalance, absorption flags, and large-order features, with walk-forward validation against the same baselines used above.
3. Richer logging (persist absorption and large-order events as first-class table rows), longer continuous live samples, and stricter out-of-sample periods.

---

## Disclaimer

Educational / research use. Cryptocurrency markets are risky. Past backtest or live-log statistics do not guarantee future results. No warranty; use at your own risk.

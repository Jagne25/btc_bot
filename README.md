# BTC Order Flow Research

Research and tooling for **BTC/USDT market microstructure** — live order-flow capture, hypothesis testing, and reproducible backtests. This repository documents an ongoing data analysis project: measuring aggressive trade flow, resting liquidity imbalance, absorption, and large-trade events, then evaluating them against explicit baselines.

> **Not trading advice.** Results below are observational findings from logged and historical data. Sample sizes, market regimes, overlapping signals, and short horizons matter; treat every win-rate as provisional until re-validated on larger independent windows.

---

## Key concepts

| Concept | What it measures | Data source |
|--------|------------------|-------------|
| **CVD (Cumulative Volume Delta)** | Running sum of aggressive buy volume minus aggressive sell volume (taker flow) | Binance `aggTrades` (`buyer_is_maker` → aggressor side) |
| **Queue imbalance (IMB)** | Relative depth at the top of book: `(bid_vol − ask_vol) / (bid_vol + ask_vol)` | Binance depth / `@depth10@100ms` |
| **Absorption** | Large CVD move over a short window while price barely moves — aggressive flow absorbed by resting liquidity | CVD + mid/last price over ~20s |
| **Large Order Detection** | Trades that are outliers vs recent activity (**relative** threshold, e.g. ≥10× rolling average of the last ~100 trades — not a fixed BTC size) | Rolling average of recent trade sizes |

**Context rule (exploratory):** CVD direction (print-to-print) aligned with IMB sign → “strong” BUY/SELL context; opposing signs → caution. Used for logging and testing, not as an automated strategy.

---

## Methodology

Hypotheses are tested with the same discipline across scripts:

1. **Define the event** (e.g. IMB sign transition, strong BUY/SELL context transition, absorption entry, large-order print, or a defined combo).
2. **Forward horizon** — typically price after **+1 … +5 minutes** (and 60s for some context tests), using the first available timestamp within a tolerance.
3. **Baseline** — same horizon on every valid row (“always long” / “always short”), so signal win-rate is compared in **percentage points (p.b. / pp)**, not in isolation.
4. **Independence** — consecutive identical context/IMB rows are **not** counted as separate signals; only **transitions** (or combo *entries*) enter the event set where applicable.
5. **Regime check** — day-by-day splits to see whether apparent edge is just market drift (e.g. mostly down days in the live sample).
6. **Threshold calibration** — absorption and large-order cutoffs are set from empirical distributions on logged data (e.g. ~p90 \|ΔCVD\|); strong IMB uses \|IMB\| ≥ 0.5 at the transition.
7. **Combo hygiene** — when combining two signals, compare the combo to the **better solo** of its legs. Combos that merely restate the same information (highly dependent sources) are expected to add less.

Live series are stored on external SSD storage and optionally in **PostgreSQL** (`order_flow_context_log`, `order_flow_large_orders`); chart/backtest artefacts stay off the Mac internal disk when `BASE_SSD_DIR` / WORK_SSD is mounted.

---

## Selected findings (data results)

Findings are from scripts in this repo on Binance BTCUSDT using the live context log (~2 weeks of collection in the current sample, ~350+ hours of rows). **Numbers will change** as more data accumulates; figures below are research checkpoints, not production signals.

### Strongest simple signal: IMB alone

- **Order-book imbalance (IMB)** — without CVD or large orders — is currently the strongest *single* short-horizon signal in this sample.
- **Strong negative IMB** (IMB → negative with \|IMB\| ≥ 0.5): success predicting a subsequent **down** move at **+1 min** around **~60%**, edge about **+10 pp** vs short baseline.
- Plain IMB− (any sign flip) also beats baseline, but the **strong** filter adds meaningful lift.
- Edge tends to **fade** over longer horizons (+3 … +5 min) — short windows matter.

### Strongest overall finding: IMB− + bullish absorption

- **Combo:** strong **IMB−** together with **bullish absorption** (CVD up hard, price held) → expect subsequent **down**.
- At **+1 min** in the current sample: success ~**66%**, edge about **+16 pp** vs short baseline — the largest combo edge observed so far (on the order of **+16 p.b.**; exact print moves slightly as the log grows).
- The same combo **weakens by +5 min** and often no longer beats the best solo leg — treat the +1 min result as short-horizon and sample-sensitive.
- Sample of combo *entries* is still **modest** (low hundreds of events); re-validation is required.

### Large Order Detection

- Relative large-order alerts (outlier vs recent trade size) are **implemented, logged, and backtested** across +1 … +5 min.
- Solo large orders show **modest positive edge** at short horizons in this sample; they are useful as a second, trade-flow-based feature, not as a standalone “holy grail.”

### Live context (CVD + IMB) and absorption alone

- On transition-based CVD+IMB context (~60s), **SELL** often printed higher success than **BUY** in early live logs.
- Day-level breakdown showed most collection days were **net down**; on those days SELL looked stronger and BUY weaker. On scarce up days the pattern flipped. **Interpretation:** a large part of the BUY/SELL gap was **market drift in the sample**, not a proven permanent asymmetry.
- Absorption alone: at +5 minutes, bullish absorption was more often followed by lower prices (fade story), but sample size and dependence on CVD definition remain caveats.

### CVD divergence (historical klines)

- Strong **bullish** divergence (price down, CVD up) on a 90-day 1m sample: overall success ~**59%** over a 10-minute forward window vs baseline “always long” ~**49%** (~**+10 pp** edge in that sample).
- Edge varied by sub-period — regime matters.

### Combinations: when they help, when they don’t

Combined signals **sometimes strengthen and sometimes dilute** prediction. A useful rule of thumb from this project:

| Combo | Typical reading in this sample |
|-------|--------------------------------|
| **IMB + Absorption** | Often **additive** — book pressure (IMB) and absorbed aggressive flow (CVD vs price) are **more independent** sources |
| **IMB + Large Order** | Can strengthen at **+1 min**; edge often **fades by +5 min** |
| **CVD/context + Large Order** | More **overlapping** — both lean on aggressive trade flow; stacking them adds less new information than IMB+absorption |

**Takeaway:** prefer combining features that come from **different microstructure channels** (resting book vs absorbed flow) over stacking two views of the same taker tape.

---

## Dashboard & batch runner

Two tooling scripts keep the research loop reproducible:

| Script | Role |
|--------|------|
| **`order_flow_run_all.py`** | Runs all `order_flow_*backtest*.py` (and related) scripts **one-by-one**, continues on failure, writes summaries to SSD; optional `--no-dashboard` |
| **`order_flow_dashboard.py`** | **Streamlit** UI that reads SSD `*summary*.txt` files, groups results by experiment type (IMB, Large Order, Absorption, CVD+IMB context, combos), and highlights win-rate / edge with green/red cards |

```bash
# Re-run every backtest, then open the dashboard
python3 order_flow_run_all.py

# Backtests only
python3 order_flow_run_all.py --no-dashboard

# Dashboard alone (SSD must be mounted)
streamlit run order_flow_dashboard.py
```

---

## Tech stack

- **Python 3** — capture, analytics, backtests  
- **pandas / NumPy** — time-aligned merges, statistics  
- **PostgreSQL** (`psycopg2`) — optional structured log (`order_flow_context_log`, `order_flow_large_orders`)  
- **Binance** — REST (aggTrades, depth, klines) + WebSocket combined streams  
- **Streamlit** — research dashboard over SSD summaries  
- **matplotlib** — CVD/price charts (SSD export)  
- Related repo experiments also use **PyTorch** / scikit-learn (separate from the current OF live loop)

---

## Repository structure (order-flow focus)

```
# Live capture
order_flow_week1.py                 # aggTrades CVD, snapshot/live, divergence, relative large-order alert
order_flow_queue.py                 # order-book imbalance (snapshot + live depth)
order_flow_context.py               # live CVD + IMB + absorption + large orders → CSV / PostgreSQL

# Solo backtests
order_flow_context_backtest.py      # BUY/SELL context transitions vs 60s baseline
order_flow_context_by_day.py        # day-level market drift vs BUY/SELL success
order_flow_absorption_backtest.py   # absorption entries vs short horizons
order_flow_large_order_backtest.py  # relative large orders, +1…+5 min + trend table
order_flow_imb_backtest.py          # IMB sign transitions (± strong |IMB|≥0.5), +1…+5 min
order_flow_backtest.py              # historical CVD divergence (klines) + period split

# Combo backtests
order_flow_combo_backtest.py              # context + absorption agreement
order_flow_combo_large_backtest.py        # context + large order
order_flow_combo_imb_large_backtest.py    # IMB + large order
order_flow_combo_imb_absorption_backtest.py  # strong IMB + absorption (strongest combo so far)

# Research UX
order_flow_run_all.py               # run all backtests → SSD summaries (+ optional dashboard)
order_flow_dashboard.py             # Streamlit overview of all summaries
order_flow_llm_summary.py           # optional Claude digest of SSD summaries (ANTHROPIC_API_KEY)
```

Other files in the repo (`bot.py`, sentiment scrapers, earlier ML/backtest utilities) belong to a broader BTC research/trading codebase; the modules above are the dedicated **order-flow research track**.

**Typical live collector:**

```bash
python3 order_flow_context.py
```

**Typical analyses (or use the runner):**

```bash
python3 order_flow_imb_backtest.py
python3 order_flow_combo_imb_absorption_backtest.py
python3 order_flow_run_all.py --no-dashboard
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

1. Longer continuous live samples and **stricter out-of-sample** periods for IMB− and IMB+absorption.
2. Richer first-class persistence of absorption / combo events; walk-forward checks that edge is not regime-only.
3. **ML fusion** (optional) — models that combine IMB, absorption, and large-order features with the same baseline discipline as the hand-crafted tests above.
4. LLM digests remain for **research notes**, not automated execution.

---

## Disclaimer

Educational / research use. Cryptocurrency markets are risky. Past backtest or live-log statistics do not guarantee future results. No warranty; use at your own risk.

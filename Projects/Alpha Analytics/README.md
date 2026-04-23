# Alpha Analytics — Earnings Event Study & Trading Strategies

End-to-end pipeline that pulls daily prices, EPS estimates/actuals, SEC XBRL
financials, and FMP technical indicators for the S&P 100 universe, builds
event windows around each earnings print, and back-tests event-driven
trading strategies with regime filters and realistic position sizing.

LSE Data Analytics Career Accelerator capstone. Repackaged from a single
notebook into a Python package with a thin orchestrator notebook, a
Streamlit dashboard, and a research/analysis module tree.

## Universe and coverage

- **Tickers:** S&P 100 constituents (103 with reported earnings in-sample)
- **Events:** 2,234 earnings announcements, 2020-01-24 → 2025-07-30
- **Prices:** daily OHLCV via yfinance, 2019-12-31 → 2025-07-31
- **Indicators:** FMP `/api/v3/technical_indicator/daily/{ticker}`, 2021-04-19
  → 2026-04-16 (77.5% event coverage). RSI-14, SMA-20/50/200, EMA-12/26
  served directly; MACD / signal / hist derived locally from EMA-12 − EMA-26
  and a 9-period EMA of the resulting line.
- **Fundamentals:** SEC XBRL via EDGAR, plus FRED macro series.

The original 3-ticker default (AAPL/NVDA/GOOG) in `src/config.py` is kept
for debugging; the S&P 100 panel is driven by `src/pull_sp100.py` and
`src/run_technical_regime.py`.

## Layout

```
Alpha Analytics/
├── src/
│   ├── config.py                     # Universe, CIK map, env-var loaders
│   ├── pull_sp100.py                 # Orchestrated S&P 100 extraction with atomic writes
│   ├── analyze_sp100.py              # Summary dataframes + ranking helpers
│   ├── run_technical_regime.py       # Regime × strategy interaction grid builder
│   ├── monitor.py                    # Forward-looking watchlist + rolling edge metrics
│   ├── data/
│   │   ├── price_extractor.py        # yfinance
│   │   ├── eps_extractor.py          # FMP earnings-surprises
│   │   ├── earnings_calendar_extractor.py  # FMP historical calendar (bmo/amc/dmh timing)
│   │   ├── sec_extractor.py          # XBRL facts + 8-K Item 2.02
│   │   └── technical_extractor.py    # FMP technical indicators (+ local MACD derivation)
│   ├── cleaning/
│   │   ├── clean_dates.py
│   │   └── sec_cleaner.py            # True Q4 = FY − (Q1+Q2+Q3)
│   ├── analysis/
│   │   ├── technical.py              # SMAs, RSI, MACD, vol
│   │   ├── event_study.py            # ±N-day windows
│   │   ├── cross_section.py          # Returns × QoQ × correlations
│   │   ├── filtered_hold_cut.py      # Regime-filtered Hold/Cut backtests + buy-and-hold benchmark
│   │   ├── position_sizing.py        # Realistic portfolio simulator (3% base / 30% cap / pyramid)
│   │   └── verify_eps_dates.py       # FMP calendar ↔ SEC 8-K ↔ AMC entry-shift data-quality audit
│   └── strategies/
│       ├── earnings_strategies.py    # Momentum, contrarian, agnostic, Hold/Cut
│       └── pre_earnings.py           # Pre-earnings run-up (three variants, see caveats)
├── dashboard.py                      # Streamlit app (7 pages incl. Live Monitor, Regime Filters)
├── notebooks/
│   └── Capstone.ipynb                # Thin orchestrator
├── data/                             # Extracted panels, summaries, equity curves
├── Report.docx                       # Original capstone write-up
├── Report_RegimeFilters.docx         # Regime-filter findings memo (4 pp)
├── Report_PositionSizing.docx        # Realistic position-sizing memo
├── Report_DataQuality.docx           # FMP / SEC / AMC data-quality annex
├── Pitch - Alpha Analytics.pptx
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with FMP key, FRED key, and a SEC user-agent email
```

## Usage

Extract and refresh the S&P 100 panel (respects FMP Starter-tier 300 req/min;
uses 0.25s spacing by default):

```bash
python -m src.pull_sp100
```

Build the regime × strategy interaction grids for Hold/Cut, Momentum, and
Pre-Runup:

```bash
python -m src.run_technical_regime
```

Re-run the filtered Hold/Cut backtests + buy-and-hold benchmark:

```bash
python -m src.analysis.filtered_hold_cut
```

Run the realistic-sizing simulator (4 variants: unfiltered / MACD-bullish,
flat / pyramid):

```bash
python -m src.analysis.position_sizing
```

Verify the `eps_date` column used by the Hold/Cut backtest — pulls the FMP
historical earnings calendar (with `bmo`/`amc`/`dmh` timing), cross-checks
against SEC 8-K Item 2.02 acceptance timestamps, and reports whether shifting
AMC event exits by one trading day materially changes the headline:

```bash
python -m src.data.earnings_calendar_extractor     # → data/earnings_calendar_sp100.csv
python -m src.analysis.verify_eps_dates            # → 3 verify_eps_*.csv artefacts
```

Build the live monitoring artefacts (watchlist of upcoming events with
MACD-filter state at the day-−10 entry, rolling-30-event edge metrics on
realised trades, pipeline-health snapshot):

```bash
python -m src.monitor                              # → watchlist.csv + monitoring_summary.csv + monitoring_health.csv
```

Interactive exploration:

```bash
streamlit run dashboard.py
```

## Strategies

Per-event strategies return a DataFrame of net returns; `summarise_strategy`
gives n / mean / median / hit-rate / t-stat.

- **Post-earnings momentum** (`+1..+H`) — long beats, short misses
- **Contrarian / agnostic contrarian** — fade the announcement move
- **Pre-earnings run-up** (`−P..−1`) — long the run-up into the print
- **Hold/Cut** (threshold + extension) — enter at announcement close; if
  cumulative return breaches the threshold on day +1, hold `+ext` more
  trading days, else exit at end of standard window. This is the headline
  strategy.

## Pre-earnings variants (credibility note)

`src/strategies/pre_earnings.py` ships three implementations with different
credibility profiles:

1. **Naive pre-earnings run-up** — tradable. Enter at −P, exit at −1, no
   selection.
2. **Post-event hold/cut** — tradable. The surprise is observed before the
   hold/cut decision, not before entry.
3. **⚠️ Entry-time surprise filter** — **look-ahead bias, research only**.
   Filters events with the post-announcement surprise before entry. The
   function logs a `WARNING` at runtime.

When quoting numbers, state which variant is being used. Do not quote
variant 3.

## Headline results (S&P 100, 2020–2025)

| Variant                                      | n     | Mean / event | Hit-rate  | t-stat    |
| -------------------------------------------- | ----- | ------------ | --------- | --------- |
| Hold/Cut (thr = 5%, ext = +10), unfiltered   | 2,234 | +2.18%       | 59.4%     | 11.27     |
| Hold/Cut + MACD-bullish filter at close      | 921   | **+6.07%**   | **81.0%** | **22.31** |
| Buy-and-hold benchmark (same dates, +1..+10) | 2,234 | +1.32%       | 57.7%     | 8.31      |

Under realistic portfolio sizing (3% of equity per event, 30% gross cap,
pyramid +3% on day +1 close > entry × 1.01), the MACD-bullish variant
returns **CAGR 16.6% / daily-Sharpe 3.90 / max drawdown −1.1%** across
2020–2025 with an average of 2.9 concurrent positions. After correcting
for the AMC exit-shift (see `Report_DataQuality.docx`), the same variant
lands at **CAGR 18.1% / Sharpe 3.63 / max drawdown −3.4%** — strategy
thesis intact, margin of imprecision is roughly ±30 bps on per-event
mean and ±0.3 on Sharpe.

Full methodology, sensitivity analysis, and caveats are in the three memos:

- `Report_RegimeFilters.docx` — how the MACD filter was chosen and why it
  works, Bonferroni correction, grid of 21 regime × strategy tests.
- `Report_PositionSizing.docx` — the realistic portfolio simulator: base
  size, pyramid trigger, gross cap, 4 variants, sensitivity sweep on
  (base, cap).
- `Report_DataQuality.docx` — FMP calendar coverage, SEC 8-K cross-check,
  and the AMC entry-shift audit referenced above.

## Key data artefacts (under `data/`)

- `indicators_sp100_v2.csv` — 104-ticker technical panel (130k rows)
- `event_regime_features_<strategy>_sp100.csv` — per-event regime snapshots
- `regime_grid_<strategy>_sp100.csv`, `regime_uplift_<strategy>_sp100.csv`
  — interaction grids and sorted uplift tables
- `filtered_hold_cut_summary.csv` — unfiltered + 3 filtered variants +
  buy-and-hold benchmark
- `equity_curve_hold_cut_*.csv`, `equity_curve_sized_*.csv` — date-keyed
  equity series for plotting
- `position_sizing_summary.csv` — sized backtest headline metrics
- `earnings_calendar_sp100.csv` — FMP historical calendar with bmo/amc/dmh
  timing, used to validate `eps_date` and identify look-ahead exposure
- `verify_eps_coverage_summary.csv`, `verify_eps_sec_crosscheck.csv`,
  `verify_eps_amc_shift_summary.csv`, `verify_eps_amc_shift_events.csv` —
  outputs of `src.analysis.verify_eps_dates`
- `watchlist.csv` — upcoming and recent S&P 100 earnings events tagged with
  MACD state at the strategy's day-−10 entry (`entry-imminent` / `entered` /
  `pre-watch` / `post-event` / `closed`). Refreshed by `src.monitor`.
- `monitoring_summary.csv` — rolling-30-event edge metrics (mean, hit-rate,
  t-stat, event Sharpe) on realised MACD-bullish trades with alarm flags.
- `monitoring_health.csv` — one-row pipeline-health snapshot (calendar
  match rate, SEC cross-check rate, AMC-shift delta, indicator freshness).

## Caveats

- yfinance returns split/dividend-adjusted closes; corporate-action dates
  are not separately filtered. For high-precision event studies, switch to
  a point-in-time vendor.
- The 8-K Item 2.02 acceptance timestamp is used as the earnings-release
  date — usually correct, occasionally lags the press-release wire by
  minutes.
- The "true Q4 = FY − (Q1+Q2+Q3)" derivation assumes reported quarterly
  values reconcile to the FY total. Restatements can break this.
- Transaction costs default to 0 bps in all backtests. A 5-bps round-trip
  overlay shaves ~10 bps from each per-event mean without changing the
  qualitative ranking.
- Compounded-equity curves in the unsized backtest assume 100% sequential
  reinvestment, which is not achievable because earnings events cluster
  heavily. Max drawdown in those curves is a comparative risk metric only
  — the `position_sizing.py` simulator is the honest version.
- FMP technical indicator history starts 2021-04-19, so 22.5% of events
  (pre-2021) sit in the "NaN" regime bucket and are excluded from the
  filter analysis.
- Regime thresholds (RSI 30/50/70, SMA crosses, MACD signal) were chosen ex
  ante from standard technical-analysis conventions, not fitted. Out-of-
  sample stability should be checked by splitting 2020–2023 vs 2024–2025.

## License

MIT

# CLAUDE.md — Congressional Trading Signal Project

## Project Charter

**Objective:** Build and validate a systematic ETF trading strategy that aggregates Congressional trading disclosures into sector-level signals, tests whether those signals contain exploitable alpha, and evaluates strategy performance against standard factor models.

**Success Criteria:**
1. H1 event study produces statistically testable CAR results (reject or fail to reject)
2. H2 committee analysis produces a comparison of committee-relevant vs. non-relevant trade CARs
3. A backtest produces a return time series with Sharpe ratio, max drawdown, and factor regression alpha
4. All outputs are reproducible from raw data to final analytics

**Users:** This is an academic research project for a Quantitative Portfolio Management course at Chicago Booth.

**Constraints:**
- No lookahead bias — all signals use `ReportDate` (disclosure date), never `TransactionDate`
- STOCK Act filings have up to 45-day disclosure lag — this is a feature, not a bug
- Quiver Quant API is the primary data source for congressional trades
- The `qpm` library from the course is available for factor regression analytics
- The `Amount` field is a lower-bound range, not an exact value — treatment must be documented

---

## Directory Structure

```
congressional-trading/
├── CLAUDE.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── settings.py                  # Global parameters, API keys, date ranges
│   └── sector_config.py             # SECTOR_CONFIG dict, committee lists, ETF mappings
├── data/
│   ├── raw/                         # Untouched API pulls, downloaded files
│   │   └── congress_trading.parquet
│   ├── processed/                   # Cleaned, enriched, analysis-ready
│   │   ├── trades_enriched.parquet
│   │   ├── committee_roster.parquet
│   │   └── ticker_sector_map.parquet
│   ├── external/                    # Third-party reference data
│   │   ├── ff5_mom_factors.csv
│   │   ├── etf_returns.parquet
│   │   └── sp500_returns.csv
│   └── snapshots/                   # Versioned data snapshots for reproducibility
├── src/
│   ├── __init__.py
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── quiver_client.py         # Module 1: Quiver API fetch + sync
│   │   ├── sector_mapper.py         # Module 2: Ticker → GICS sector
│   │   ├── committee_mapper.py      # Module 3: Legislator → committee roster
│   │   ├── stock_returns.py         # Module 4: Daily stock + market returns
│   │   ├── etf_returns.py           # Module 5: Sector ETF daily returns
│   │   ├── factor_data.py           # Module 6: FF5 + Momentum factors
│   │   └── enrichment.py            # Module 7: Master join + validation
│   ├── event_study/
│   │   ├── __init__.py
│   │   ├── market_model.py          # Module 8: Estimation window + expected returns
│   │   ├── abnormal_returns.py      # Module 9: AR and CAR computation
│   │   └── h1_test.py               # Module 10: Cross-sectional tests for H1
│   ├── committee_analysis/
│   │   ├── __init__.py
│   │   └── h2_test.py               # Module 11: Committee vs non-committee CAR comparison
│   ├── signal_engine/
│   │   ├── __init__.py
│   │   └── signal_generator.py      # Module 12: Refactored signal generation
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── portfolio.py             # Module 13: Portfolio construction + rebalancing
│   │   ├── performance.py           # Module 14: Performance metrics
│   │   └── factor_regression.py     # Module 15: FF5+Mom alpha regression
│   └── utils/
│       ├── __init__.py
│       ├── validation.py            # Schema validation, dedup, type checks
│       └── constants.py             # Enums, column name constants, amount range mappings
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_h1_event_study.ipynb
│   ├── 03_h2_committee_test.ipynb
│   ├── 04_signal_dashboard.ipynb
│   ├── 05_backtest.ipynb
│   └── 06_factor_regression.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_event_study.py
│   ├── test_signal_engine.py
│   └── test_backtest.py
└── logs/
    ├── assumptions.md               # Every assumption, mapping decision, threshold choice
    └── data_transformations.md      # Every cleaning step, filter, join, with row counts
```

---

## Implementation Sequence

**CRITICAL: Follow this order. Each phase depends on the prior phase's outputs.**

```
Phase 1: Foundation (config + data pipeline)
    → Phase 2: H1 Event Study (validate the signal exists)
        → Phase 3: H2 Committee Analysis (validate the information channel)
            → Phase 4: Signal Engine (refactor existing code + fix bugs)
                → Phase 5: Backtest (historical portfolio simulation)
                    → Phase 6: Factor Regression (alpha evaluation)
```

---

## Phase 1: Foundation — Config & Data Pipeline

### Charter Alignment
Serves: Data Requirements (Proposal §5), Global Instructions §8 (Data Discipline)

### Definition of Done
- All six data sources are fetchable, validated, and persisted to `data/processed/`
- Master enriched trade dataset joins trades → sectors → committees → stock returns
- Schema validation passes with zero nulls in critical columns
- `logs/assumptions.md` and `logs/data_transformations.md` are initialized

---

### Module 1: `config/settings.py`

```python
"""
All tunable parameters in one place. No magic numbers elsewhere.
"""
# --- API ---
QUIVER_API_KEY = ""  # Set via environment variable QUIVER_API_KEY
QUIVER_BASE_URL = "https://api.quiverquant.com/beta"

# --- SAMPLE PERIOD ---
SAMPLE_START = "2016-01-01"    # Start of backtest period
SAMPLE_END = "2025-12-31"      # End of backtest period
ESTIMATION_WINDOW_START = "2015-01-01"  # Extra year for market model estimation

# --- EVENT STUDY (H1) ---
ESTIMATION_WINDOW = (-250, -30)    # Trading days relative to event
EVENT_WINDOW_IMMEDIATE = (-1, 1)   # CAR for immediate reaction
EVENT_WINDOW_DRIFT = (2, 20)       # CAR for post-disclosure drift
MIN_ESTIMATION_DAYS = 120          # Minimum obs in estimation window

# --- SIGNAL ENGINE ---
LOOKBACK_DAYS = 45                 # Rolling window for signal generation
CONVICTION_THRESHOLD = 0.80        # Net volume / total volume threshold
BACKTEST_LOOKBACK_DAYS = 90        # Proposal specifies 90 for backtest

# --- BACKTEST ---
REBALANCE_FREQUENCY = "M"         # Monthly rebalancing
TRANSACTION_COST_BPS = 10         # One-way transaction cost assumption (basis points)
# PRIMARY: Threshold-based (Variant A) — any sector passing gates + conviction acts independently
# ROBUSTNESS: Tercile ranking (Variant B) — relative ranking, requires 3+ valid sectors
BACKTEST_VARIANT = "threshold"    # "threshold" (primary) or "tercile" (robustness)

# --- AMOUNT RANGE MAPPING ---
# Quiver reports lower bound of STOCK Act ranges. This maps to midpoints.
# ASSUMPTION: We use midpoint of each range as the dollar estimate.
# Document this in logs/assumptions.md
AMOUNT_MIDPOINTS = {
    1001: 8000,        # $1,001 - $15,000 → midpoint $8,000
    15001: 57500,      # $15,001 - $50,000 → midpoint $32,500 (VERIFY RANGE)
    50001: 125000,     # $50,001 - $100,000 → midpoint $75,000
    100001: 300000,    # $100,001 - $250,000 → midpoint $175,000
    250001: 375000,    # $250,001 - $500,000 → midpoint $375,000
    500001: 750000,    # $500,001 - $1,000,000 → midpoint $750,000
    1000001: 5000000,  # $1,000,001 - $5,000,000 → midpoint $3,000,000
    5000001: 25000000, # $5,000,001 - $25,000,000 → midpoint $15,000,000
    25000001: 37500000,# $25,000,001 - $50,000,000 → midpoint $37,500,000
    50000001: 50000001 # $50,000,001+ → USE LOWER BOUND (no upper bound)
}
# NOTE: Verify these ranges against actual Quiver schema. The exact breakpoints
# may differ. Pull a sample and confirm before using.
```

### Module 2: `config/sector_config.py`

This is the existing `SECTOR_CONFIG` dict from the Gemini codebase, but with one addition: a `gics_codes` field per sector that maps GICS sub-industry codes to the sector. This bridges the gap between the ticker→GICS mapping and the signal engine.

```python
"""
Sector configuration: committees, ETFs, thresholds, and GICS mappings.

ASSUMPTION: We restrict to 7 sectors with high government intervention.
This is a design choice, not a data-driven result. Document in assumptions.md.

ASSUMPTION: Committee lists are manually curated based on jurisdiction analysis.
Each mapping decision should be recorded in assumptions.md with rationale.
"""
SECTOR_CONFIG = {
    "Defense & Aerospace": {
        "target_etf": "XAR",
        "committees": [
            "House Armed Services",
            "Senate Armed Services",
            "House Appropriations",  # Defense subcommittee
            "Senate Appropriations",  # Defense subcommittee
            "House Foreign Affairs",
            "Senate Foreign Relations"
        ],
        "gics_sector": "Industrials",
        "gics_industry_keywords": ["Aerospace", "Defense"],
        "tau_member": 2,
        "tau_ticker": 3,
        "min_transactions": 8
    },
    # ... (include all 7 sectors from existing code, each with gics_sector
    #      and gics_industry_keywords added)
}

# Reverse lookup: ETF ticker → sector name
ETF_TO_SECTOR = {v["target_etf"]: k for k, v in SECTOR_CONFIG.items()}
```

**Implementation note on `gics_industry_keywords`:** The sector mapper (Module 3) will use these keywords to match GICS sub-industry descriptions to project sectors. For example, a stock classified as GICS sub-industry "Aerospace & Defense" matches `["Aerospace", "Defense"]` for the "Defense & Aerospace" project sector. This is fuzzy by design — document every edge case in `assumptions.md`.

---

### Module 3: `src/data_pipeline/quiver_client.py`

**Purpose:** Fetch congressional trades from Quiver Quant API, sync with historical parquet, validate and deduplicate.

```python
"""
Quiver Quant API client for STOCK Act congressional trading data.

Functions:
    fetch_live_trades(api_key: str) -> pd.DataFrame
    load_historical(parquet_path: str) -> pd.DataFrame
    sync_and_deduplicate(df_hist: pd.DataFrame, df_live: pd.DataFrame) -> pd.DataFrame

API SCHEMA NORMALIZATION (CRITICAL):
    The Quiver API V2 endpoint returns 'Filed' and 'Traded', NOT 'ReportDate'
    and 'TransactionDate'. The historical parquet may use either convention.
    Immediately upon fetch, rename:
        'Filed' → 'ReportDate'
        'Traded' → 'TransactionDate'
    DEFENSIVE CHECK: If the raw response contains NEITHER 'Filed' NOR
    'ReportDate', raise a ValueError with the actual column names found.
    Do not silently proceed with missing date columns.

Output Schema (after sync):
    - ReportDate: datetime64       (disclosure date — THE date for signal timing)
    - TransactionDate: datetime64  (actual trade date — for analysis only, never signal timing)
    - Ticker: str                  (stock ticker)
    - Name: str                    (legislator name as reported)
    - Transaction: str             ("Purchase" or "Sale")
    - Amount: float                (lower bound USD of reported range)
    - Chamber: str                 ("House" or "Senate", if available)
    - Direction: int               (+1 for Purchase, -1 for Sale)
    - AmountMidpoint: float        (estimated midpoint from AMOUNT_MIDPOINTS mapping)
    - DisclosureLag: int           (ReportDate - TransactionDate in calendar days)

Deduplication Strategy (in priority order):
    1. PREFERRED: If Quiver API returns a unique transaction ID field,
       use it as the sole dedup key. Check the raw schema for fields like
       'id', 'transaction_id', or similar.
    2. FALLBACK: Use composite key (Name, Ticker, TransactionDate, Transaction, Amount).
       This includes Amount to preserve multiple same-day trades of different sizes
       (bug fix from prior code which used only 4 fields).
    3. KNOWN LIMITATION: If a legislator makes two truly identical trades
       (same ticker, same day, same direction, same amount bucket), the composite
       key cannot distinguish them. In this case, keep all rows and document
       that trade counts may be slightly inflated. Log the count of such
       ambiguous duplicates in data_transformations.md.

Validation checks:
    - No nulls in (ReportDate, Ticker, Name, Transaction)
    - Transaction ∈ {"Purchase", "Sale"} (drop "Exchange", "Receive" if present)
    - DisclosureLag ∈ [0, 365] (flag outliers beyond 45 days)
    - Amount > 0
"""
```

**Key fixes from prior code:**
1. The deduplication composite key must include `Amount` (or use a native transaction ID if available).
2. API column names must be normalized immediately on fetch — `Filed`→`ReportDate`, `Traded`→`TransactionDate`.
Log both changes in `data_transformations.md`.

---

### Module 4: `src/data_pipeline/sector_mapper.py`

**Purpose:** Map each stock ticker to a GICS sector and to one of the 7 project sectors.

```python
"""
Ticker → GICS → Project Sector mapping.

Approach:
    1. Use yfinance (or a static GICS classification file) to get GICS sector
       and GICS industry for each unique ticker in the trade dataset.
    2. Match GICS industry descriptions against gics_industry_keywords in SECTOR_CONFIG.
    3. Tickers that don't match any of the 7 project sectors → labeled "Other" and
       excluded from the signal engine (but retained for H1 event study).

Functions:
    get_gics_for_tickers(tickers: list[str]) -> pd.DataFrame
        Returns: ticker, gics_sector, gics_industry, gics_sub_industry
    map_to_project_sector(gics_df: pd.DataFrame, sector_config: dict) -> pd.DataFrame
        Returns: ticker, gics_sector, gics_industry, project_sector

Output saved to: data/processed/ticker_sector_map.parquet

KNOWN ISSUES:
    - yfinance may return None for delisted tickers → log these, attempt manual lookup
    - Some tickers are ETFs/options traded by legislators → flag and decide treatment
    - GICS reclassifications happen over time → use point-in-time if possible,
      otherwise use current classification and document in assumptions.md

Caching: Cache yfinance lookups to avoid repeated API calls. Store in
data/raw/yfinance_gics_cache.json with {ticker: {sector, industry, sub_industry, fetched_date}}.
"""
```

---

### Module 5: `src/data_pipeline/committee_mapper.py`

**Purpose:** Map each legislator to their committee assignments. This is the hardest data engineering module.

```python
"""
Legislator → Committee membership mapping.

Data Sources (try in order):
    1. congress.gov API (https://api.congress.gov/) — free, requires API key
    2. ProPublica Congress API — may be deprecated, check availability
    3. Manually curated fallback CSV for the ~535 active members

Approach:
    1. For each Congress session in the sample period (e.g., 114th–119th Congress),
       pull full committee rosters.
    2. Normalize legislator names to match Quiver Quant's Name field.
       This is the hardest part — Quiver uses display names like "Nancy Pelosi"
       while official records use "Pelosi, Nancy" or include middle initials.
       Strategy: fuzzy match using (last_name, first_name, state, chamber).
    3. Build a temporal mapping: (legislator, committee, congress_session, start_date, end_date).
    4. For each trade, look up the legislator's committee assignments AT THE TIME OF THE TRADE
       (not current assignments).

Functions:
    fetch_committee_rosters(congress_sessions: list[int]) -> pd.DataFrame
        Returns: member_name, committee_name, congress_session, chamber
    normalize_names(quiver_names: pd.Series, roster_names: pd.Series) -> pd.DataFrame
        Returns: quiver_name, matched_roster_name, match_confidence
    assign_committees_to_trades(trades_df: pd.DataFrame, roster_df: pd.DataFrame) -> pd.DataFrame
        Returns: trades_df with added columns [Committee_List, Is_Committee_Relevant]

Output saved to: data/processed/committee_roster.parquet

NAME MATCHING STRATEGY:
    1. Exact match on lowercase(last_name + first_name)
    2. If no exact match: fuzzy match using fuzzywuzzy with threshold >= 90
    3. If still no match: log to assumptions.md and flag for manual review
    4. Maintain a manual overrides dict for known mismatches

COMMITTEE NAME NORMALIZATION:
    Committee names in SECTOR_CONFIG use abbreviated forms (e.g., "Senate HELP").
    Official names are longer (e.g., "Senate Health, Education, Labor, and Pensions").
    Build a mapping dict from official names → config names.
    Document every mapping in assumptions.md.

IS_COMMITTEE_RELEVANT LOGIC:
    For a given trade: look up the legislator's committees at that time,
    look up the trade's project_sector, check if any of the legislator's
    committees appear in SECTOR_CONFIG[project_sector]["committees"].
    If yes → Is_Committee_Relevant = True.
"""
```

---

### Module 6: `src/data_pipeline/stock_returns.py`

**Purpose:** Fetch daily stock returns and market returns for the event study.

```python
"""
Daily stock returns for event study computation.

Source: yfinance (free) or CRSP via WRDS (if available through course access).

Functions:
    fetch_stock_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame
        Returns: date, ticker, ret (daily return), price, volume
    fetch_market_returns(start: str, end: str) -> pd.DataFrame
        Returns: date, mkt_ret (S&P 500 daily return)

Output Schema:
    - date: datetime64
    - ticker: str
    - ret: float (daily simple return, e.g., 0.02 for +2%)
    - mkt_ret: float (S&P 500 daily return, joined from market data)

NOTE: yfinance returns adjusted close prices. Compute returns as:
    ret_t = (adj_close_t / adj_close_{t-1}) - 1

IMPORTANT: For the event study, we need returns for EVERY ticker that appears
in the congressional trade dataset. Some tickers may be delisted — handle gracefully
by fetching what's available and logging gaps.

Caching: Store fetched returns in data/raw/stock_returns_cache.parquet.
Only fetch incremental data for new date ranges.
"""
```

---

### Module 7: `src/data_pipeline/etf_returns.py`

**Purpose:** Fetch daily returns for sector ETFs used in the backtest.

```python
"""
Sector ETF daily returns for backtest portfolio construction.

ETFs (from SECTOR_CONFIG): XAR, XHS, XPH, XLF, XLE, XLU, XLI

Functions:
    fetch_etf_returns(etf_tickers: list[str], start: str, end: str) -> pd.DataFrame
        Returns: date, etf_ticker, ret, price

Output saved to: data/external/etf_returns.parquet

Also fetch SPY as benchmark for relative performance.
"""
```

---

### Module 8: `src/data_pipeline/factor_data.py`

**Purpose:** Download Fama-French 5 factors + Momentum.

```python
"""
Fama-French factor returns from Kenneth French Data Library.

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    Use pandas_datareader or direct CSV download.

Functions:
    fetch_ff5_mom(start: str, end: str) -> pd.DataFrame
        Returns: date, Mkt_RF, SMB, HML, RMW, CMA, Mom, RF

Output saved to: data/external/ff5_mom_factors.csv

NOTE: French library reports monthly factors. For the backtest (monthly rebalancing),
this is the correct frequency. For daily event study regressions, use daily factors
(available as "Fama/French 5 Factors (2x3) [Daily]" from the same source).

COLUMN NAMES: Standardize to Mkt_RF, SMB, HML, RMW, CMA, Mom, RF.
French library uses "Mkt-RF" — rename on load.
Values are in percentage points (e.g., 1.5 = 1.5%) — divide by 100 for decimal returns.
"""
```

---

### Module 9: `src/data_pipeline/enrichment.py`

**Purpose:** Master join that produces the analysis-ready dataset.

```python
"""
Master enrichment pipeline. Joins all data sources into a single
analysis-ready DataFrame.

Pipeline:
    1. Load raw trades from quiver_client
    2. Map tickers to sectors via sector_mapper
    3. Map legislators to committees via committee_mapper
    4. Tag each trade as Is_Committee_Relevant
    5. Join stock returns (for H1 event study)
    6. Validate final schema
    7. Save to data/processed/trades_enriched.parquet

Final Schema for trades_enriched.parquet:
    - ReportDate: datetime64
    - TransactionDate: datetime64
    - Ticker: str
    - Name: str
    - Transaction: str ("Purchase" or "Sale")
    - Direction: int (+1 or -1)
    - Amount: float (lower bound)
    - AmountMidpoint: float
    - DisclosureLag: int (calendar days)
    - Chamber: str
    - GICS_Sector: str
    - GICS_Industry: str
    - Project_Sector: str (one of 7 sectors or "Other")
    - Committee_List: list[str] (committees at time of trade)
    - Is_Committee_Relevant: bool
    - target_etf: str (from SECTOR_CONFIG, null if "Other")

Validation (run and log all counts):
    - Total trades loaded: N
    - After dedup: N
    - Matched to GICS sector: N (% of total)
    - Matched to project sector: N (% of total)
    - Matched to committee roster: N (% of total)
    - Committee-relevant trades: N (% of matched)
    - Trades with "Other" sector (excluded from signal): N
    - Final enriched dataset rows: N

Functions:
    run_enrichment_pipeline(api_key: str, parquet_path: str) -> pd.DataFrame
    validate_enriched(df: pd.DataFrame) -> dict  # Returns validation counts
"""
```

---

## Phase 2: H1 — Market Underreaction Event Study

### Charter Alignment
Serves: Proposal §3 H1, Global Instructions §3 (Scientific Method)

### Definition of Done
- CAR[-1,+1] and CAR[+2,+20] computed for all trade disclosure events
- Cross-sectional t-test results for buys and sells separately
- Clear accept/reject conclusion for H1
- Robustness checks: winsorized, micro-cap excluded, subperiod splits

---

### Module 10: `src/event_study/market_model.py`

```python
"""
Market model estimation for computing expected returns.

For each disclosure event (unique trade identified by ReportDate + Ticker):
    1. Define estimation window: [ReportDate - 250 trading days, ReportDate - 30 trading days]
    2. Require at least MIN_ESTIMATION_DAYS (120) observations in the window
    3. Run OLS: stock_ret_t = alpha + beta * mkt_ret_t + epsilon_t
    4. Store (alpha, beta, sigma_epsilon) per event

Functions:
    fit_market_model(event_date: date, ticker: str,
                     stock_returns: pd.DataFrame,
                     market_returns: pd.DataFrame,
                     estimation_window: tuple = (-250, -30),
                     min_obs: int = 120) -> dict
        Returns: {"alpha": float, "beta": float, "sigma": float,
                  "n_obs": int, "r_squared": float}

    fit_all_events(events_df: pd.DataFrame,
                   stock_returns: pd.DataFrame,
                   market_returns: pd.DataFrame) -> pd.DataFrame
        Returns: events_df with added columns [mm_alpha, mm_beta, mm_sigma, mm_n_obs, mm_r2]

NOTE: Events where market model cannot be estimated (insufficient data,
delisted stock, etc.) should be flagged and excluded from CAR analysis
with counts logged in data_transformations.md.

TRADING DAY CONVERSION: Use the stock returns DataFrame's date index
to count trading days, not calendar days.
"""
```

---

### Module 11: `src/event_study/abnormal_returns.py`

```python
"""
Abnormal return and cumulative abnormal return computation.

For each event with a fitted market model:
    AR_t = actual_ret_t - (alpha + beta * mkt_ret_t)
    CAR[t1, t2] = sum(AR_t for t in [t1, t2])

Functions:
    compute_abnormal_returns(event_date: date, ticker: str,
                             alpha: float, beta: float,
                             stock_returns: pd.DataFrame,
                             market_returns: pd.DataFrame,
                             window: tuple) -> pd.Series
        Returns: Series of AR indexed by trading day offset

    compute_car(ar_series: pd.Series, window: tuple) -> float
        Returns: CAR for the specified window

    compute_all_cars(events_df: pd.DataFrame,
                     stock_returns: pd.DataFrame,
                     market_returns: pd.DataFrame) -> pd.DataFrame
        Returns: events_df with added columns:
            - CAR_immediate (CAR[-1,+1])
            - CAR_drift (CAR[+2,+20])
            - AR_day0 (abnormal return on disclosure date itself)
            - n_days_immediate (actual trading days in window)
            - n_days_drift (actual trading days in window)

EDGE CASES:
    - If a stock doesn't trade on a day in the event window (holiday, halt),
      that day contributes AR=0. Log how often this happens.
    - If a stock is delisted during the event window, truncate the CAR
      and flag the event. Do NOT forward-fill returns.
"""
```

---

### Module 12: `src/event_study/h1_test.py`

```python
"""
Statistical tests for H1: Market Underreaction.

Hypothesis (as stated in proposal):
    If CAR[-1,+1] ≈ 0 but CAR[+2,+20] > 0 for purchases,
    the market underreacts to congressional trade disclosures.

Tests:
    1. Cross-sectional t-test on mean CAR[-1,+1] for purchases
       H0: mean CAR[-1,+1] = 0
    2. Cross-sectional t-test on mean CAR[+2,+20] for purchases
       H0: mean CAR[+2,+20] = 0
    3. Same for sales (expect opposite sign)
    4. Patell (1976) standardized test as robustness
       (standardize each CAR by its estimation-period sigma)

Functions:
    run_h1_tests(cars_df: pd.DataFrame) -> dict
        Input: DataFrame with CAR_immediate, CAR_drift, Direction
        Returns: {
            "buys_immediate": {"mean_car": float, "t_stat": float, "p_value": float, "n": int},
            "buys_drift":     {"mean_car": float, "t_stat": float, "p_value": float, "n": int},
            "sells_immediate": {...},
            "sells_drift":     {...},
        }

    run_robustness(cars_df: pd.DataFrame) -> dict
        Runs:
        - Winsorized at 1%/99%
        - Excluding micro-caps (bottom decile by market cap)
        - Subperiod split (first half vs second half of sample)
        - Excluding trades with DisclosureLag > 45 days
        Returns: dict of test results per robustness variant

Interpretation Guide (print in notebook output):
    - If buys_drift p < 0.05 and buys_immediate p > 0.10 → UNDERREACTION CONFIRMED
    - If buys_drift p > 0.10 → NO EVIDENCE of exploitable signal
    - If buys_immediate p < 0.05 → Market reacts at disclosure, window is smaller
    - Document result in assumptions.md regardless of outcome
"""
```

---

## Phase 3: H2 — Committee Information Advantage

### Charter Alignment
Serves: Proposal §3 H2, Global Instructions §3 (Scientific Method)

### Definition of Done
- CAR comparison: committee-relevant trades vs. non-relevant trades (within-member)
- CAR comparison: committee-relevant trades vs. same-sector trades by non-committee members
- Statistical test results with clustered standard errors
- Clear finding documented in assumptions.md

---

### Module 13: `src/committee_analysis/h2_test.py`

```python
"""
H2: Committee Information Advantage test.

Two comparisons (both required per proposal):

Comparison A — Within-member:
    For legislators who trade BOTH in and out of their committee sectors,
    compare mean CAR[+2,+20] for committee-relevant vs. non-relevant trades.
    This controls for individual trading skill.

Comparison B — Cross-member:
    For a given sector, compare mean CAR[+2,+20] of trades by committee members
    vs. trades by non-committee members.
    This isolates the committee information channel.

Statistical approach:
    - Difference-in-means with clustered standard errors
    - Cluster on (member) and (date) to handle correlation
    - Use statsmodels OLS with robust covariance:
        CAR_drift = alpha + beta * Is_Committee_Relevant + controls + epsilon
        cluster on member_id

Functions:
    run_h2_within_member(cars_df: pd.DataFrame) -> dict
    run_h2_cross_member(cars_df: pd.DataFrame) -> dict
    run_h2_regression(cars_df: pd.DataFrame) -> statsmodels.RegressionResults

Interpretation:
    - If beta on Is_Committee_Relevant is positive and significant →
      Committee channel validated, overweight these trades in signal engine
    - If not significant →
      H2 rejected, proceed with equal weighting (document null result)
"""
```

---

## Phase 4: Signal Engine Refactor

### Charter Alignment
Serves: Proposal §4 (Strategy Application), Global Instructions §5 (Reproducibility)

### Definition of Done
- Signal generator fixed (committee filter enforced, dedup key corrected, Amount treatment documented)
- Can be called for any historical date to produce sector directives
- Unit tests pass for known-state inputs

---

### Module 14: `src/signal_engine/signal_generator.py`

**This refactors the existing Gemini codebase. Key changes:**

```python
"""
Signal generation engine. Produces sector-level BUY/SELL/FLAT directives
based on aggregated congressional trading activity.

CHANGES FROM PRIOR CODE:
    1. BUG FIX: Actually filter trades to committee-relevant members only.
       The prior code read thresholds from SECTOR_CONFIG but never filtered
       the groupby to committee members. Now: for each sector, only count
       trades where Is_Committee_Relevant == True.

    2. BUG FIX: Use AmountMidpoint instead of raw Amount for volume calculations.
       Raw Amount is a lower bound and introduces systematic downward bias
       for small trades and massive bias from $50M+ trades.
       ASSUMPTION: AmountMidpoint from settings.py. Document in assumptions.md.

    3. ENHANCEMENT: Accept current_date parameter to enable historical backtesting.
       When current_date is provided, the lookback window ends at current_date.

    4. ENHANCEMENT: Return conviction_score as a continuous signal (not just
       BUY/SELL/FLAT). The backtest uses this for portfolio construction.

    5. FIX: The conviction threshold for directive strings (0.80) must be
       read from the conviction_thresh parameter, NOT hardcoded in the function
       body. The backtest may sweep over different thresholds.

Functions:
    generate_signals(df_enriched: pd.DataFrame,
                     config: dict,
                     lookback_days: int,
                     conviction_thresh: float,
                     current_date: pd.Timestamp = None,
                     committee_filter: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]
        Returns: (sector_summary_df, detailed_trades_df)

    sector_summary_df columns:
        - Sector: str
        - target_etf: str
        - directive: str ("BUY LONG {ETF}", "SELL SHORT {ETF}", "FLAT: ...", "INSUFFICIENT DATA: ...")
        - conviction_score: float [-1, 1]  (continuous signal for backtest)
        - total_transactions: int
        - net_volume: float
        - total_volume: float
        - n_members: int
        - n_tickers: int
        - diversity_pass: bool
        - volume_pass: bool
"""
```

---

## Phase 5: Backtest

### Charter Alignment
Serves: Proposal §4 (Strategy Application — Backtest), Global Instructions §9 (Evaluation)

### Definition of Done
- Historical portfolio returns computed monthly from `SAMPLE_START` to `SAMPLE_END`
- Portfolio weights based on signal engine directives at each rebalance date
- Returns time series saved for factor regression
- Performance summary: annualized return, Sharpe, max drawdown, turnover

---

### Module 15: `src/backtest/portfolio.py`

```python
"""
Backtest portfolio construction and rebalancing.

PRIMARY STRATEGY (Variant A — Absolute Threshold):
    The alpha hypothesis is absolute, not relative: "when enough committee members
    are buying heavily in a sector, that sector outperforms." A single sector with
    a strong enough signal is tradeable on its own. This means:

    1. At each month-end rebalance date:
       a. Call generate_signals() with current_date = rebalance_date
          and lookback_days = BACKTEST_LOOKBACK_DAYS (90 days per proposal)
       b. For each sector that passes ALL gating checks (diversity + volume):
          - conviction_score >= conviction_thresh → LONG that sector's ETF
          - conviction_score <= -conviction_thresh → SHORT that sector's ETF
          - Otherwise → NO POSITION in that sector
       c. Equal-weight across all triggered sectors on each side (long/short)
       d. If zero sectors trigger, go to cash (return = 0 for the month)
    2. Hold positions until next rebalance date
    3. Compute portfolio return for the month using sector ETF returns

    This design means the portfolio is concentrated when signals are sparse
    (potentially just 1 sector) and diversified when signals cluster. The
    strategy will carry directional market exposure — it will be net-long in
    months when buying signals dominate and net-short when selling dominates.
    The factor regression in Phase 6 will capture this as MktRF loading.

ROBUSTNESS CHECK (Variant B — Relative Tercile Ranking):
    For the notebook comparison only. Rank sectors with valid signals by
    conviction_score:
        n_long = floor(n_valid / 3), n_short = floor(n_valid / 3)
        remainder = NO POSITION
    Minimum n_valid = 3, otherwise go to cash.
    This variant is market-neutral by construction but fails when few
    sectors have signals (which is the common case).

Functions:
    run_backtest(df_enriched: pd.DataFrame,
                 etf_returns: pd.DataFrame,
                 config: dict,
                 start_date: str,
                 end_date: str,
                 lookback_days: int = 90,
                 conviction_thresh: float = 0.80,
                 rebalance_freq: str = "M",
                 tc_bps: int = 10,
                 variant: str = "threshold") -> pd.DataFrame
        variant: "threshold" (primary) or "tercile" (robustness)
        Returns: DataFrame with columns:
            - date: month-end date
            - portfolio_ret_gross: float
            - portfolio_ret_net: float (after transaction costs)
            - long_sectors: list[str]
            - short_sectors: list[str]
            - n_valid_signals: int
            - turnover: float

    The output DataFrame is the strategy return time series that feeds
    directly into the factor regression in Phase 6.

EDGE CASES:
    - Months with zero triggered sectors → return = 0 (all cash)
    - Months where only long or only short signals exist →
      run as long-only or short-only (this is expected behavior for the
      threshold variant, not an error). Document in assumptions.md.
    - Single sector triggered → 100% weight in that sector's ETF.
      This is by design — the gating logic ensures the signal is robust
      enough to act on individually.
    - ETF data gaps → log and exclude month

NOTE ON DIRECTIONAL EXPOSURE:
    Congressional buying tends to cluster in bullish environments, so the
    threshold strategy will be net-long most of the time. This is an inherent
    property of the signal, not a bug. The factor regression alpha is what
    matters — if alpha is significant after controlling for MktRF, the signal
    adds value beyond the directional bet.
"""
```

---

### Module 16: `src/backtest/performance.py`

```python
"""
Portfolio performance analytics.

Functions:
    compute_performance(returns: pd.Series, rf: pd.Series = None) -> dict
        Returns: {
            "annualized_return": float,
            "annualized_vol": float,
            "sharpe_ratio": float,
            "max_drawdown": float,
            "max_drawdown_start": date,
            "max_drawdown_end": date,
            "calmar_ratio": float,
            "avg_monthly_turnover": float,
            "hit_rate": float,  # % of months with positive returns
            "best_month": float,
            "worst_month": float,
            "n_months": int,
            "pct_months_invested": float  # % of months not all-cash
        }

    plot_cumulative_returns(returns: pd.Series, benchmark: pd.Series = None,
                           title: str = "") -> matplotlib.Figure

    plot_drawdown(returns: pd.Series) -> matplotlib.Figure
"""
```

---

## Phase 6: Factor Regression — Alpha Evaluation

### Charter Alignment
Serves: Proposal §4 (Sharpe ratio, Fama-French 5 + Momentum regression), Global Instructions §10 (Separation of Concerns)

### Definition of Done
- Regression of portfolio excess returns on FF5 + Momentum
- Alpha (intercept) with t-stat and p-value reported
- Factor loadings interpreted
- Comparison with course `qpm.analyze_strategy` output if feasible

---

### Module 17: `src/backtest/factor_regression.py`

```python
"""
Fama-French 5 Factor + Momentum regression to isolate alpha.

Model:
    R_portfolio_t - RF_t = alpha + b1*MktRF_t + b2*SMB_t + b3*HML_t
                           + b4*RMW_t + b5*CMA_t + b6*Mom_t + epsilon_t

Functions:
    run_factor_regression(portfolio_returns: pd.Series,
                          factor_data: pd.DataFrame) -> statsmodels.RegressionResults
        NOTE: portfolio_returns should be EXCESS returns (subtract RF).
        factor_data must contain: Mkt_RF, SMB, HML, RMW, CMA, Mom
        Both must be at the same frequency (monthly).

    format_regression_table(results: statsmodels.RegressionResults) -> pd.DataFrame
        Returns: formatted coefficient table with t-stats, p-values, significance stars

    interpret_results(results: statsmodels.RegressionResults) -> str
        Returns: plain-English interpretation of alpha and factor loadings

Interpretation Guide:
    - alpha > 0 and significant (p < 0.05): Strategy generates genuine alpha
      beyond known risk factors. This is the goal.
    - alpha ≈ 0: Returns are explained by factor exposure. The signal is not
      adding value beyond what you could get from a factor portfolio.
    - Large HML loading: Strategy is essentially a value tilt
    - Large MktRF loading: Strategy is just leveraged beta
    - Large Mom loading: Strategy is momentum in disguise

CONNECTION TO COURSE NOTEBOOK:
    The class notebook uses qpm.analyze_strategy(df_rets, analysis_type='Factor Regression').
    If you can format the backtest returns into the shape expected by qpm, use it directly.
    Otherwise, this module replicates the same regression using statsmodels.
"""
```

---

## Notebooks

Each notebook should follow this structure:

```python
# Cell 1: Header
"""
Notebook: XX_name.ipynb
Charter Objective: [which objective this serves]
Current Milestone: [what we're computing]
Definition of Done: [what success looks like for this notebook]
"""

# Cell 2: Imports and config
# Cell 3-N: Analysis
# Final Cell: Summary
"""
Decisions Made: [list]
Unresolved Questions: [list]
Next Required Artifacts: [list]
"""
```

### `01_data_exploration.ipynb`
- Load raw Quiver data, inspect schema, check date ranges
- Verify Amount distribution, flag outliers
- Count unique legislators, tickers, sectors
- Visualize trade volume over time (are there seasonal patterns?)
- Check disclosure lag distribution
- Output: Validation report, data quality assessment

### `02_h1_event_study.ipynb`
- Load enriched dataset + stock returns
- Run market model estimation, show example for 3-5 events
- Compute CARs for full sample
- Run H1 tests, display results table
- Run robustness checks
- Plot average CAR path [-5, +30] for buys vs. sells
- Output: H1 accept/reject with confidence level

### `03_h2_committee_test.ipynb`
- Load enriched dataset with committee flags
- Split into committee-relevant vs. non-relevant
- Run within-member and cross-member comparisons
- Display regression results
- Output: H2 accept/reject, decision on committee weighting

### `04_signal_dashboard.ipynb`
- Run signal engine for current date
- Display sector summary table
- Display triggered sector ledger
- Show historical signal heatmap (which sectors triggered over time)
- Output: Current actionable signals

### `05_backtest.ipynb`
- Run full backtest using primary strategy (threshold-based, Variant A)
- Plot cumulative returns vs. SPY
- Plot drawdown
- Display monthly returns heatmap
- Show portfolio composition over time (which sectors triggered each month)
- Show distribution of number of sectors held per month
- Run robustness check: Variant B (tercile ranking) for comparison
- Document expected directional (net-long) exposure
- Output: Backtest return time series saved to file

### `06_factor_regression.ipynb`
- Load backtest returns + factor data
- Run FF5+Mom regression
- Display formatted coefficient table
- Interpret alpha and factor loadings
- If possible, run through qpm.analyze_strategy for validation
- Output: Final alpha assessment, project conclusion

---

## Tests

### `tests/test_data_pipeline.py`
```python
"""
- test_dedup_preserves_different_amounts: Two trades by same member, same ticker,
  same day, same direction, different amounts → both kept
- test_dedup_removes_true_duplicates: Identical rows → one kept
- test_dedup_ambiguous_identical_trades: Two rows identical in ALL fields →
  both kept (conservative), count logged as ambiguous
- test_api_schema_rename: Raw API with 'Filed'/'Traded' columns →
  renamed to 'ReportDate'/'TransactionDate' after fetch
- test_api_schema_missing_dates: Raw API missing both 'Filed' and 'ReportDate' →
  raises ValueError with actual column names
- test_direction_encoding: Purchase → +1, Sale → -1
- test_amount_midpoint_mapping: Each Amount lower bound → correct midpoint
- test_disclosure_lag_calculation: Known (ReportDate, TransactionDate) → correct lag
- test_no_lookahead: Signal window ends at current_date, never after
"""
```

### `tests/test_event_study.py`
```python
"""
- test_market_model_known_values: Synthetic data with known alpha/beta → correct estimates
- test_car_computation: Synthetic AR series → correct CAR over window
- test_insufficient_estimation_data: < MIN_ESTIMATION_DAYS → event excluded
- test_event_window_trading_days: Uses trading days not calendar days
"""
```

### `tests/test_signal_engine.py`
```python
"""
- test_committee_filter_applied: Only Is_Committee_Relevant=True trades counted
- test_conviction_score_range: Always in [-1, 1]
- test_diversity_threshold: Below tau_member or tau_ticker → FLAT
- test_volume_threshold: Below min_transactions → INSUFFICIENT DATA
- test_no_future_data: Signal at date T uses only ReportDate <= T
"""
```

### `tests/test_backtest.py`
```python
"""
- test_monthly_rebalance: Portfolio changes only at month-end
- test_all_cash_when_no_signals: Zero triggered sectors → return = 0
- test_single_sector_long: One sector passes gating with conviction >= 0.80 →
  100% weight in that sector's ETF, portfolio return = ETF return
- test_single_sector_short: One sector with conviction <= -0.80 →
  100% short weight, portfolio return = -1 * ETF return
- test_multiple_sectors_equal_weight: 3 sectors trigger LONG →
  each gets 1/3 weight
- test_long_only_month: Only LONG signals, no SHORT → net-long portfolio
  (this is expected behavior for threshold variant, not an error)
- test_transaction_costs: Known turnover → correct cost deduction
- test_returns_match_etf: Single-sector portfolio → matches ETF return
- test_tercile_variant_minimum: Tercile variant with < 3 valid signals → cash
"""
```

---

## Assumptions Log (Initial Entries)

Start `logs/assumptions.md` with these documented decisions:

```markdown
# Assumptions Log

| ID | Date | Assumption | Rationale | Impact if Wrong |
|----|------|-----------|-----------|-----------------|
| A1 | YYYY-MM-DD | Amount field treated as midpoint of STOCK Act range | Lower bound biases volume calculations downward | Conviction scores shift; threshold may need recalibration |
| A2 | YYYY-MM-DD | Restrict to 7 sectors with high government intervention | Reduce noise from sectors where committee edge is unlikely | May miss alpha in other sectors |
| A3 | YYYY-MM-DD | GICS classification is current, not point-in-time | Historical GICS unavailable without CRSP | Some tickers misclassified in historical periods |
| A4 | YYYY-MM-DD | Committee assignments matched at Congress session level, not exact date | Mid-session changes are rare | Some trades tagged to wrong committee |
| A5 | YYYY-MM-DD | Lookback window uses ReportDate only | Avoids lookahead bias from TransactionDate | Signal may be stale for trades with long disclosure lag |
| A6 | YYYY-MM-DD | Deduplication prefers native transaction ID; falls back to composite key including Amount | Preserves multiple same-day trades of different sizes; truly identical trades kept conservatively | Trade counts may be slightly inflated for identical-in-all-fields trades |
| A7 | YYYY-MM-DD | Transaction cost assumption: 10 bps one-way | Sector ETFs are highly liquid | If costs are higher, net returns decrease proportionally |
| A8 | YYYY-MM-DD | Threshold-based portfolio (Variant A) is the primary strategy, not tercile ranking | The alpha hypothesis is absolute ("sector outperforms when committee members buy") not relative ("sector X outperforms sector Y"). A single sector with a strong signal is tradeable. Tercile ranking fails with sparse signals (common case with 7 sectors and strict gating). | Strategy carries directional market exposure; factor regression captures this as MktRF loading |
| A9 | YYYY-MM-DD | Strategy will be net-long most months due to congressional buying clustering in bullish environments | This is an inherent property of the signal, not a design flaw | High MktRF beta in factor regression; alpha after controlling for market is what matters |
| A10 | YYYY-MM-DD | API column names normalized immediately on fetch: Filed→ReportDate, Traded→TransactionDate | Quiver API V2 uses different field names than internal schema | Pipeline crashes if rename is missed; defensive check raises error on schema mismatch |
```

---

## Requirements

```
# requirements.txt
pandas>=2.0
numpy>=1.24
requests>=2.28
pyarrow>=12.0           # parquet support
yfinance>=0.2.18        # stock data
statsmodels>=0.14       # regressions, event study stats
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
fuzzywuzzy>=0.18        # name matching
python-Levenshtein>=0.21 # speeds up fuzzywuzzy
jupyter>=1.0
pytest>=7.0
pandas_datareader>=0.10 # Fama-French data
tqdm>=4.65              # progress bars for long loops
```

---

## Execution Instructions for Claude Code

1. **Start by creating the directory structure exactly as specified above.**
2. **Implement Phase 1 modules in order (Module 1 through Module 9).** After each module, run the corresponding test.
3. **After Phase 1 is complete, run the enrichment pipeline end-to-end** and verify the validation counts in `data_transformations.md`.
4. **Implement Phase 2 (Modules 10-12).** Run `02_h1_event_study.ipynb` and record results.
5. **The outcome of H1 determines whether to proceed.** If H1 shows no signal, stop and document. If H1 confirms underreaction, continue.
6. **Implement Phase 3 (Module 13).** Run `03_h2_committee_test.ipynb`. H2 outcome determines whether to use committee weighting in signal engine.
7. **Implement Phase 4 (Module 14)** — the signal engine refactor.
8. **Implement Phase 5 (Modules 15-16).** Run `05_backtest.ipynb`.
9. **Implement Phase 6 (Module 17).** Run `06_factor_regression.ipynb`. This produces the final deliverable.

**At every phase boundary, update `logs/assumptions.md` and `logs/data_transformations.md`.**

---

## Binding Rules

- No module may use `TransactionDate` for signal timing or lookback windows. Only `ReportDate`.
- No module may introduce a new threshold, parameter, or mapping without adding it to `config/settings.py` or `config/sector_config.py` and documenting it in `logs/assumptions.md`.
- All DataFrames passed between modules must conform to the documented schema. If a schema changes, update the docstring AND the downstream consumers.
- Every statistical test must report: test statistic, p-value, sample size, and degrees of freedom.
- The backtest must never use information from after the rebalance date. This is the single most important correctness requirement.

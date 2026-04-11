# DRL Stock Trading Agent Sandbox

Welcome to the FinRL-based Proximal Policy Optimization (PPO) agent sandbox! This project establishes an environment to train a Deep Reinforcement Learning agent to trade 5 diverse Indian assets autonomously.

## The Environment (Markov Decision Process)

To ensure this model mimics the real-life constraints of a stock exchange, the DRL agent interacts with a strictly defined Markov Decision Process (MDP):

### 1. State Space
The agent observes the environment through a multidimensional state vector at each timestep:
- **Available Balance**: Current cash in the portfolio.
- **Asset Prices**: The closing prices of the 5 assets (NIFTYETF, Reliance, Tata Motors, GoldETF, Infosys).
- **Shares Owned**: The current number of shares held for each asset.
- **Technical Indicators**: MACD and RSI (30-day window) to gauge momentum and overbought/oversold conditions.

### 2. Action Space
The agent outputs an action containing continuous values between `-1` and `1` for each of the 5 assets simultaneously:
- **-1**: Sell all shares.
- **0**: Hold position.
- **1**: Buy maximum possible shares with available cash.
Values in between represent fractional/percentage buys and sells.

### 3. Reward Function
The reward function drives the agent's behavior:
- **Base Reward**: The change in portfolio value (Total Value at $t$ - Total Value at $t-1$).
- **Volatility Penalty**: A massive penalty applied if the portfolio experiences high downside volatility. The agent is trained to survive market crashes, not just to make highly leveraged, risky bets.

### Real-life constraints modeled into the environment:
- **Transaction Fees**: 0.1% fee on every buy/sell simulating exchange and broker fees.
- **Slippage**: Slight penalty upon execution to account for spread.
- **Absolute Cash Boundary**: Agent cannot spend more cash than available (no naked shorting).

---

## Technical Stack Overview

- **FinRL**: Used for standardized environment building (`StockTradingEnv`). Note: We bypass `FeatureEngineer` and `YahooDownloader` to ensure data schema stability.
- **Stable Baselines 3**: Provides the highly robust Proximal Policy Optimization (**PPO**) implementation. PPO uses clipped surrogate objectives, balancing sample efficiency and ease of tuning.
- **Stockstats**: Utilized for native, deterministic technical indicator generation (MACD, RSI), replacing the unstable FinRL internal preprocessors.
- **Optuna**: Handles automated hyperparameter tuning (Learning Rate, Batch Size, etc.) using the Tree-structured Parzen Estimator (TPE) algorithm.
- **MLflow**: Acts as our indestructible "flight recorder." It tracks every trial parameter and optimization metric using nested runs for trial isolation.
- **SHAP (SHapley Additive exPlanations)**: Post-training explainability framework to understand exactly *why* the agent makes certain trades.

## Dependency Management & Mocks

This project includes two "shim" files in the root directory: `alpaca_trade_api.py` and `ccxt.py`.

### Why are they here?
The `finrl` library contains hard-coded imports for these modules deep within its core logic. Even if you are trading Indian assets (via `yfinance`) and not using Alpaca or Crypto brokers, the library will crash with a `ModuleNotFoundError` during initialization if these are missing.

### Why mocks instead of real packages?
Installing the real `alpaca-trade-api` package forces a downgrade of `urllib3` to a legacy version (v1.24) which is incompatible with modern ML tools like `mlflow`. To maintain a high-performance environment, we use these lightweight **mocks** to satisfy FinRL's import requirements while keeping the core environment stable and up-to-date.

---

## Architecture Decision Records (ADRs)

### Pandas vs. Polars for Data Preprocessing
While we originally architected the preprocessing pipeline to utilize **Polars** to achieve lightning-fast data cleaning, we discovered a severe compatibility bottleneck when interfacing with the RL backend.

The `stockstats` library (which calculates MACD and RSI) relies on traditional Pandas indexing built on pure Python `object` strings. However, passing dataframes through Polars and exporting them via `.to_pandas()` triggers the default `PyArrow` backend. This string conversion fundamentally fractured `stockstats` ticker lookups, throwing massive `KeyError` crashes in the pipeline.

**Resolution**: To guarantee deterministic indicator generation, the data phase was reverted to native, pure **Pandas**. Given the current volume of the sandbox dataset (10 years, 5 assets = ~12,000 rows), Pandas processes it instantly. If the project ever scales to ultra-high-frequency tick aggregation, Polars must be reintroduced with strict custom cast safeguards deployed immediately prior to metric calculation.

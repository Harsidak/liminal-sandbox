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

- **FinRL**: Used for standardized environment building (`StockTradingEnv`) and automated technical indicator generation (`FeatureEngineer`).
- **Stable Baselines 3**: Provides the highly robust Proximal Policy Optimization (**PPO**) implementation. PPO uses clipped surrogate objectives, balancing sample efficiency and ease of tuning.
- **Polars**: Utilized instead of pandas for ultra-fast dataframe operations during preprocessing.
- **Optuna**: Handles automated hyperparameter tuning (Learning Rate, Batch Size, Gamma, etc.) using Tree-structured Parzen Estimator (TPE) algorithm.
- **MLflow**: Acts as our indestructible "flight recorder." It tracks every trial parameter, optimization metrics, and automatically logs the best model weights.
- **SHAP (SHapley Additive exPlanations)**: Post-training explainability framework to understand exactly *why* the agent makes certain trades based on indicators like MACD or RSI.

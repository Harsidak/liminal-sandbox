import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


class StrategistTradingEnv(StockTradingEnv):
    """
    A long-horizon portfolio allocation environment that subclasses FinRL's StockTradingEnv.

    Key modifications from the base day-trading environment:

    1. Reward Function: Rolling Sharpe Ratio + Continuous Quadratic Drawdown Penalty
       instead of raw daily portfolio ₹ change. This forces the agent to optimize for
       risk-adjusted returns rather than chasing raw growth.

    2. Inflation Mechanic: Daily cash decay (8% annual default) simulating real
       purchasing power erosion. If the agent holds idle cash, it bleeds money —
       quantifying the "Cost of Inaction."

    3. Quadratic Drawdown Penalty: A continuous, non-linear penalty that barely
       registers for small dips (5%) but becomes devastating for large crashes (40%).
       This teaches the agent to hedge rather than rely on a binary safety net.
    """

    def __init__(
        self,
        *,
        inflation_rate: float = 0.08,
        sharpe_lookback: int = 60,
        drawdown_penalty_scale: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --- Inflation Mechanic ---
        # 8% annual inflation → daily decay factor ≈ 0.999669
        # Uninvested cash bleeds purchasing power every step, forcing capital deployment.
        self.inflation_rate = inflation_rate
        self.daily_decay_factor = (1 - inflation_rate) ** (1 / 252)

        # --- Sharpe Ratio Reward ---
        # Rolling window for computing the annualized Sharpe Ratio.
        # 60 trading days ≈ 3 months — balances reactivity with stability.
        self.sharpe_lookback = sharpe_lookback

        # --- Continuous Quadratic Drawdown Penalty ---
        # penalty = -(drawdown_fraction²) × scale
        # 5% dd → -0.025 (barely registers)
        # 15% dd → -0.225 (moderate drag)
        # 40% dd → -1.600 (devastating)
        self.drawdown_penalty_scale = drawdown_penalty_scale

    def step(self, actions):
        # =====================================================================
        # Phase 1: INFLATE AWAY IDLE CASH (before any trades execute)
        # This is the "Cost of Inaction" — money sitting uninvested loses value.
        # Applied every non-terminal step, eroding self.state[0] (cash balance).
        # =====================================================================
        if not self.terminal:
            self.state[0] *= self.daily_decay_factor

        # =====================================================================
        # Phase 2: EXECUTE PARENT'S FULL TRADE CYCLE
        # Handles: action scaling by hmax, sell/buy execution with transaction
        # costs, day advancement, state vector update, and raw portfolio value
        # change stored in asset_memory.
        # =====================================================================
        state, reward, terminal, truncated, info = super().step(actions)

        # =====================================================================
        # Phase 3: OVERRIDE REWARD WITH RISK-ADJUSTED METRIC
        # Replace the parent's raw ₹-change reward with:
        #   reward = Rolling Sharpe Ratio + Quadratic Drawdown Penalty
        # Only recalculate for non-terminal steps with sufficient price history.
        # =====================================================================
        if not terminal and len(self.asset_memory) >= 2:
            # -- Rolling Sharpe Ratio Component --
            # Uses min(lookback, available_history) so early steps still get
            # a Sharpe estimate that gradually stabilizes.
            lookback = min(self.sharpe_lookback, len(self.asset_memory) - 1)
            recent_values = self.asset_memory[-(lookback + 1):]
            daily_returns = np.diff(recent_values) / np.array(recent_values[:-1])

            if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10:
                sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
            else:
                sharpe = 0.0

            # -- Continuous Quadratic Drawdown Penalty --
            # Peak-to-trough drawdown fraction, penalized as dd².
            # Small drawdowns are tolerated; catastrophic ones are devastating.
            peak = max(self.asset_memory)
            current_value = self.asset_memory[-1]
            drawdown_frac = (peak - current_value) / peak if peak > 0 else 0.0
            drawdown_penalty = -(drawdown_frac ** 2) * self.drawdown_penalty_scale

            # -- Combined Risk-Adjusted Reward --
            new_reward = sharpe + drawdown_penalty

            # Overwrite the parent's raw ₹-change reward in memory
            self.rewards_memory[-1] = new_reward
            self.reward = new_reward * self.reward_scaling
            reward = self.reward

        return state, reward, terminal, truncated, info


def build_environment(df, inflation_rate=0.08, sharpe_lookback=60, drawdown_penalty_scale=10.0):
    """
    Builds the Strategist MDP with real-world constraints:
    - Risk-adjusted reward (Sharpe + Quadratic Drawdown Penalty)
    - Inflation-decaying cash pool (8% annual)
    - Higher position limits for portfolio-scale allocation (hmax=1000)
    - Expanded state space: momentum indicators + macroeconomic proxies
    """
    # CRITICAL: FinRL's StockTradingEnv iterates through time using `self.df.loc[self.day]`.
    # To return a dataframe patch of all 5 tickers concurrently per step, the index must match the unique dates.
    # We must enforce this here in case `df` was loaded from a CSV (which drops the index).
    df = df.copy()
    if "date" in df.columns:
         df.index = df.date.factorize()[0]

    stock_dimension = len(df.tic.unique())

    # Expanded indicator list: momentum (MACD, RSI) + macroeconomic (VIX, TNX, Gold)
    tech_indicator_list = ["macd", "rsi_30", "vix_close", "tnx_close", "gold_close"]

    state_space = 1 + 2 * stock_dimension + len(tech_indicator_list) * stock_dimension
    print(f"State Space Dimension: {state_space}")
    
    env_kwargs = {
        "hmax": 1000,                                   # Portfolio-scale position sizing (was 100)
        "initial_amount": 100000,                        # ₹1,00,000 INR
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,       # 0.1% Broker/Exchange Fee
        "sell_cost_pct": [0.001] * stock_dimension,      # 0.1% Broker/Exchange Fee
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StrategistTradingEnv(
        df=df,
        inflation_rate=inflation_rate,
        sharpe_lookback=sharpe_lookback,
        drawdown_penalty_scale=drawdown_penalty_scale,
        **env_kwargs,
    )
    return e_train_gym, env_kwargs

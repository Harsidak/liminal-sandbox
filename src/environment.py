import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from gymnasium import spaces


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
        
        # --- MultiDiscrete Support for MaskablePPO ---
        # 21 bins means 10 is 'hold', 20 is 'max buy', 0 is 'max sell'
        # This discrete space ensures sb3-contrib MaskablePPO functions without crashing
        self.action_space = spaces.MultiDiscrete([21] * self.stock_dim)

    def action_masks(self):
        """
        Hard constraint mask: If an asset currently holds > 40% of the portfolio's total value,
        mask (disable) further BUY actions for that specific asset.
        """
        prices = np.array(self.state[1 : 1 + self.stock_dim])
        holdings = np.array(self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim])
        cash = self.state[0]
        
        portfolio_value = cash + np.sum(prices * holdings)
        masks = np.ones((self.stock_dim, 21), dtype=bool)
        
        if portfolio_value > 0:
            asset_weights = (prices * holdings) / portfolio_value
            for i in range(self.stock_dim):
                if asset_weights[i] > 0.40:
                    # Disable all BUY bins (11 to 20)
                    masks[i, 11:] = False
                    
        # sb3-contrib requires a 1D flat array for MultiDiscrete masks
        return masks.flatten()

    def step(self, actions):
        # =====================================================================
        # Phase 1: INFLATE AWAY IDLE CASH (before any trades execute)
        # This is the "Cost of Inaction" — money sitting uninvested loses value.
        # Applied every non-terminal step, eroding self.state[0] (cash balance).
        # =====================================================================
        if not self.terminal:
            self.state[0] *= self.daily_decay_factor

        # Decode MultiDiscrete (0-20 bins) to Continuous [-1, 1] for FinRL backend
        continuous_actions = (actions - 10) / 10.0

        if not self.terminal:
            # Snapshot state for slippage derivation
            old_holdings = np.array(self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim])
            
            # Execute trade cycle
            state, reward, terminal, truncated, info = super().step(continuous_actions)
            
            # --- Dynamic Slippage ---
            # Cost = Base_fee + K * price * shares * sqrt(shares / volume)
            new_holdings = np.array(self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim])
            prices = np.array(self.state[1 : 1 + self.stock_dim])
            
            # The environment pulls tech_indicators into self.data sequentially
            # Volume is at index 5 in the tech_indicator list (MACD, RSI, VIX, TNX, Gold, Volume...)
            if hasattr(self, "data") and "volume" in self.data:
                volumes = self.data["volume"].values
                volumes_safe = np.maximum(volumes, 1.0)
                shares_traded = np.abs(new_holdings - old_holdings)
                
                # Dynamic slip penalty constant
                K = 0.05
                dynamic_slip = np.sum(0.001 * prices * shares_traded + K * prices * shares_traded * np.sqrt(shares_traded / volumes_safe))
                
                self.state[0] -= dynamic_slip
                self.asset_memory[-1] -= dynamic_slip
        else:
            state, reward, terminal, truncated, info = super().step(continuous_actions)

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
    # Add volume & covariance tracking (dynamic search)
    cov_cols = [c for c in df.columns if c.startswith("cov_")]
    tech_indicator_list = ["macd", "rsi_30", "vix_close", "tnx_close", "gold_close", "volume"] + cov_cols

    state_space = 1 + 2 * stock_dimension + len(tech_indicator_list) * stock_dimension
    print(f"State Space Dimension: {state_space}")
    
    env_kwargs = {
        "hmax": 1000,                                   # Portfolio-scale position sizing (was 100)
        "initial_amount": 100000,                        # ₹1,00,000 INR
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.0] * stock_dimension,         # 0% base (moved to dynamic slippage)
        "sell_cost_pct": [0.0] * stock_dimension,        # 0% base (moved to dynamic slippage)
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

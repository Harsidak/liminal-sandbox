from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

def build_environment(df):
    """
    Builds the MDP with real-world stock exchange constraints.
    """
    # CRITICAL: FinRL's StockTradingEnv iterates through time using `self.df.loc[self.day]`.
    # To return a dataframe patch of all 5 tickers concurrently per step, the index must match the unique dates.
    # We must enforce this here in case `df` was loaded from a CSV (which drops the index).
    df = df.copy()
    if "date" in df.columns:
         df.index = df.date.factorize()[0]

    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2*stock_dimension + len(["macd", "rsi_30"])*stock_dimension
    print(f"State Space Dimension: {state_space}")
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000, # INR
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,  # 0.1% Broker/Exchange Fee
        "sell_cost_pct": [0.001] * stock_dimension, # 0.1% Broker/Exchange Fee
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": ["macd", "rsi_30"],
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    
    e_train_gym = StockTradingEnv(df=df, **env_kwargs)
    return e_train_gym, env_kwargs

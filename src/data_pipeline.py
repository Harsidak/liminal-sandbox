import pandas as pd
import yfinance as yf
import stockstats

# --- Macroeconomic Proxy Tickers ---
# These are exogenous state variables that let the agent "see" the broader economy.
# Without these, the agent is blind to regime changes and can only react to price momentum.
#
# ^VIX  - CBOE Volatility Index (fear gauge — spikes during market crashes)
# ^TNX  - 10-Year US Treasury Yield (interest rate proxy — drives capital rotation)
# GC=F  - Gold Futures (safe-haven proxy — inversely correlated with equities in crisis)
MACRO_TICKERS = {"^VIX": "vix_close", "^TNX": "tnx_close", "GC=F": "gold_close"}


def _download_macro_proxies(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads macroeconomic proxy data and returns a single-row-per-date DataFrame.
    Forward-fills to handle US/India trading calendar mismatches (NSE may be open
    when US markets are closed, so we carry the last known macro value forward).
    """
    print(f"Downloading macroeconomic proxies: {list(MACRO_TICKERS.keys())}")

    macro_frames = []
    for ticker, col_name in MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if raw.empty:
                print(f"  WARNING: No data for {ticker}, skipping.")
                continue

            # Handle MultiIndex columns that yfinance sometimes returns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            series = raw[["Close"]].copy()
            series.columns = [col_name]
            series.index = pd.to_datetime(series.index).strftime('%Y-%m-%d')
            macro_frames.append(series)
        except Exception as e:
            print(f"  WARNING: Failed to download {ticker}: {e}")

    if not macro_frames:
        raise ValueError("Failed to download any macroeconomic proxy data.")

    # Merge all macro series on date, using outer join to capture all trading days
    macro_df = macro_frames[0]
    for frame in macro_frames[1:]:
        macro_df = macro_df.join(frame, how="outer")

    # Forward-fill calendar gaps, then back-fill edges (first few rows)
    macro_df = macro_df.ffill().bfill()
    macro_df.index.name = "date"
    macro_df = macro_df.reset_index()

    print(f"  Macro data shape: {macro_df.shape}")
    return macro_df


def fetch_and_process_data(start_date: str, end_date: str, tickers: list) -> pd.DataFrame:
    """
    Downloads historical data via YFinance, engineers technical indicators,
    and merges macroeconomic proxy data for the RL environment.

    Pipeline:
        1. Download OHLCV data for Indian assets
        2. Engineer momentum indicators (MACD, RSI-30) via stockstats
        3. Download & merge macroeconomic proxies (VIX, TNX, Gold)
        4. Validate alignment (every date must have exactly stock_dim tickers)
        5. Set FinRL-compatible index
    """
    print(f"Downloading data for: {tickers} from {start_date} to {end_date}")

    # =========================================================================
    # Step 1: Download Indian asset OHLCV data
    # We bypass FinRL's broken YahooDownloader by calling modern yfinance natively
    # =========================================================================
    df_raw = yf.download(tickers, start=start_date, end=end_date)

    # Restructure safely without deprecated kwargs
    data_list = []
    for tic in tickers:
        temp_df = df_raw.xs(tic, level=1, axis=1).copy() if isinstance(df_raw.columns, pd.MultiIndex) else df_raw.copy()
        if "Close" in temp_df:
            temp_df["tic"] = tic
            temp_df["date"] = temp_df.index
            temp_df = temp_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            data_list.append(temp_df[["date", "tic", "open", "high", "low", "close", "volume"]])

    if not data_list:
        raise ValueError("No valid data downloaded from yfinance for the requested tickers.")

    final_df = pd.concat(data_list, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"]).dt.strftime('%Y-%m-%d')
    final_df = final_df.sort_values(["date", "tic"]).reset_index(drop=True)

    # Clean nulls natively using Pandas without triggering PyArrow string casting from Polars
    # which notoriously fractures FinRL's stockstats backend when computing technicals
    clean_df = final_df.dropna().reset_index(drop=True)

    print("Asset data processed. Shape:", clean_df.shape)

    # =========================================================================
    # Step 2: Engineer Momentum Indicators (MACD + RSI-30) NATIVELY
    # We bypass FinRL's broken loop concatenation FeatureEngineer and process
    # indicators group-by-group so the pandas index doesn't explode.
    # =========================================================================
    print("Engineering MACD and RSI (30 days) natively...")

    processed_tables = []
    unique_tickers = clean_df["tic"].unique()

    for tic in unique_tickers:
        tic_df = clean_df[clean_df["tic"] == tic].copy()
        # Stockstats mutates the dataframe it works on natively
        stock = stockstats.StockDataFrame.retype(tic_df.copy())

        # Pull generated series straight into target frames
        tic_df["macd"] = stock["macd"].values
        tic_df["rsi_30"] = stock["rsi_30"].values
        processed_tables.append(tic_df)

    processed_df = pd.concat(processed_tables, ignore_index=True)

    # Sort strictly by date then tic because gym environments expect chronological ordering
    processed_df = processed_df.sort_values(["date", "tic"]).reset_index(drop=True)

    # Final null clearance for lagging indicators (MACD requires 26 bars, RSI requires 30 bars)
    processed_df = processed_df.dropna().reset_index(drop=True)

    # =========================================================================
    # Step 3: Download & Merge Macroeconomic Proxies
    # VIX, TNX, and Gold Futures are global indicators merged into every ticker
    # row by date. Since US/India calendars differ, forward-fill handles gaps.
    # =========================================================================
    macro_df = _download_macro_proxies(start_date, end_date)
    processed_df = processed_df.merge(macro_df, on="date", how="left")

    # Forward-fill any remaining NaN in macro columns (NSE dates without US data)
    for col in MACRO_TICKERS.values():
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].ffill().bfill()

    # Drop rows where macro data is still NaN (edge case at data boundaries)
    processed_df = processed_df.dropna().reset_index(drop=True)

    # =========================================================================
    # Step 4: Gym Alignment — Every step must contain exactly stock_dim tickers
    # Filter out dates where a ticker might have halted trading and dropped out.
    # =========================================================================
    counts = processed_df.groupby("date").size()
    valid_dates = counts[counts == len(unique_tickers)].index
    processed_df = processed_df[processed_df["date"].isin(valid_dates)].reset_index(drop=True)

    # =========================================================================
    # Step 5: Set FinRL-compatible index
    # FinRL's StockTradingEnv iterates through time using `self.df.loc[self.day]`.
    # To return a dataframe patch of all 5 tickers concurrently per step, the
    # index must match the unique dates.
    # =========================================================================
    processed_df.index = processed_df.date.factorize()[0]

    print("Feature Engineering complete. Final Shape:", processed_df.shape)
    print(f"Columns: {list(processed_df.columns)}")
    return processed_df

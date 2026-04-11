import pandas as pd
import yfinance as yf
import polars as pl
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

def fetch_and_process_data(start_date: str, end_date: str, tickers: list) -> pd.DataFrame:
    """
    Downloads historical data via YFinance, processes it safely bypassing deprecated methods,
    and formats for FinRL's FeatureEngineer.
    """
    print(f"Downloading data for: {tickers} from {start_date} to {end_date}")
    
    # Let's bypass FinRL's broken YahooDownloader by calling modern yfinance natively
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
    
    # Log some stats
    print("Data processed safely natively. Shape:", clean_df.shape)
    
    # Setup technical indicators NATIVELY bypass FinRL's broken loop concatenation FeatureEngineer
    print("Engineering MACD and RSI (30 days) natively...")
    import stockstats
    
    # Process indicators group by group securely so pandas index doesn't explode
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
    
    # CRITICAL: Gym environments require EVERY step to contain exactly `stock_dimension` tickers.
    # We must filter out any dates where a ticker might have halted trading and dropped out.
    counts = processed_df.groupby("date").size()
    valid_dates = counts[counts == len(unique_tickers)].index
    processed_df = processed_df[processed_df["date"].isin(valid_dates)].reset_index(drop=True)
    
    # CRITICAL: FinRL's StockTradingEnv iterates through time using `self.df.loc[self.day]`.
    # To return a dataframe patch of all 5 tickers concurrently per step, the index must match the unique dates.
    processed_df.index = processed_df.date.factorize()[0]
    
    print("Feature Engineering complete. Final Shape:", processed_df.shape)
    return processed_df

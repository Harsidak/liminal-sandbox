import os
import pandas as pd
import optuna
import mlflow
import shap

from src.data_pipeline import fetch_and_process_data
from src.training import make_objective, train_final_model

def run():
    # Top 5 diverse Indian Assets + macroeconomic proxies (VIX, TNX, Gold) as exogenous state
    TICKERS = ["NIFTYBEES.NS", "RELIANCE.NS", "HDFCBANK.NS", "GOLDBEES.NS", "INFY.NS"]
    START_DATE = "2014-01-01"
    END_DATE = "2024-01-01"

    # Create root level required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Step 1: Prep Data (Downloads 10 years of history + macroeconomic proxies)
    # NOTE: Delete data/processed_indian_assets.csv to force re-download after pipeline changes
    processed_data_path = "data/processed_indian_assets.csv"
    if not os.path.exists(processed_data_path):
        df = fetch_and_process_data(START_DATE, END_DATE, TICKERS)
        df.to_csv(processed_data_path, index=False)
    else:
        print("Loading cached processed data...")
        df = pd.read_csv(processed_data_path)
    
    # Step 2: Optuna search & MLflow setup
    mlflow.set_experiment("Indian_Assets_DRL_PPO")
    
    print("\nStarting Optuna Hyperparameter Optimization...")
    with mlflow.start_run(run_name="Optuna_Optimization"):
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(df), n_trials=3) # set n_trials larger for actual tuning
        
        print("\nBest Hyperparameters:", study.best_params)
        mlflow.log_params(study.best_params)
        
        # Step 3: Train final robust model with best params
        final_model = train_final_model(df, study.best_params)

    # Step 4: SHAP Feature Importance Mock output for Explainability Strategy
    print("\nSHAP initialization (Explainability)")
    print("Simulated SHAP Output: MACD features contribute most heavily during periods of high market volatility.")
    
    print("\n[SUCCESS] Entire pipeline completed.")

if __name__ == "__main__":
    run()

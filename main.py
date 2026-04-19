import os
import pandas as pd
import optuna
import mlflow

from src.data_pipeline import fetch_and_process_data
from src.training import make_objective, train_final_model

# =====================================================================
# Production Configuration
# =====================================================================
TICKERS = ["NIFTYBEES.NS", "RELIANCE.NS", "HDFCBANK.NS", "GOLDBEES.NS", "INFY.NS"]
START_DATE = "2014-01-01"
END_DATE = "2024-01-01"

# Optuna trials: 100 for production-grade hyperparameter search
N_TRIALS = 100

# SubprocVecEnv worker count (sweet spot for i5-13th Gen: 8 workers)
N_ENVS = 8

# Total timesteps for final training
TOTAL_TIMESTEPS = 10_000_000


def run():
    # Create root level required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # -----------------------------------------------------------------
    # Step 1: Prep Data (Downloads 10 years of history + macro proxies)
    # -----------------------------------------------------------------
    # NOTE: Delete data/processed_indian_assets.csv to force re-download
    # after pipeline changes (e.g., new indicators or covariance features).
    processed_data_path = "data/processed_indian_assets.csv"
    if not os.path.exists(processed_data_path):
        df = fetch_and_process_data(START_DATE, END_DATE, TICKERS)
        df.to_csv(processed_data_path, index=False)
        print(f"[✓] Processed data saved to {processed_data_path}")
    else:
        print(f"[*] Loading cached processed data from {processed_data_path}")
        print("    (Delete this file to force a fresh download)")
        df = pd.read_csv(processed_data_path)
    
    # -----------------------------------------------------------------
    # Step 2: Optuna search & MLflow setup
    # -----------------------------------------------------------------
    mlflow.set_experiment("Indian_Assets_DRL_PPO")
    
    print(f"\n[*] Starting Optuna Hyperparameter Optimization ({N_TRIALS} trials)...")
    with mlflow.start_run(run_name="Optuna_Optimization"):
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(df), n_trials=N_TRIALS)
        
        print(f"\n[✓] Best Hyperparameters: {study.best_params}")
        mlflow.log_params(study.best_params)
        
        # -----------------------------------------------------------------
        # Step 3: Train final robust model with best params (parallelized)
        # -----------------------------------------------------------------
        final_model = train_final_model(
            df,
            study.best_params,
            n_envs=N_ENVS,
            total_timesteps=TOTAL_TIMESTEPS,
        )

    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Trials completed : {N_TRIALS}")
    print(f"  Parallel workers : {N_ENVS}")
    print(f"  Total timesteps  : {TOTAL_TIMESTEPS:,}")
    print(f"  Model saved to   : models/liminal_ppo_agent.zip")
    print(f"  Checkpoints in   : models/checkpoints/")
    print("=" * 55)
    print("\n  Next steps:")
    print("    1. python src/stitch.py    → Promote best checkpoint to champion_model.zip")
    print("    2. python src/evaluate.py  → Generate audit report + SHAP analysis")
    print()

if __name__ == "__main__":
    run()

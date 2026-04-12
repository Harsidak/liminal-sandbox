"""
stitch.py — Champion Model Selector
====================================
Scans ./models/checkpoints/ for all PPO checkpoint .zip files,
evaluates each on a holdout validation environment (2023–2024),
and promotes the best performer to ./models/champion_model.zip.

Usage (from project root):
    python src/stitch.py
    python -m src.stitch
"""

import os
import sys
import glob
import shutil
import numpy as np
import pandas as pd

# =====================================================================
# PATH BOOTSTRAP — Must execute BEFORE any project/library imports.
# When running `python src/stitch.py`, sys.path[0] == "src/".
# The shim files (alpaca_trade_api.py, ccxt.py) live in the project
# root, and FinRL's internal imports require them. We inject the
# project root into sys.path so Python can find both `src.*` modules
# and the root-level shims.
# =====================================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import mlflow
from stable_baselines3 import PPO
from src.data_pipeline import fetch_and_process_data
from src.environment import build_environment


# =====================================================================
# Pipeline Configuration — Single source of truth for evaluation params
# =====================================================================
TICKERS = ["NIFTYBEES.NS", "RELIANCE.NS", "HDFCBANK.NS", "GOLDBEES.NS", "INFY.NS"]
VAL_START = "2023-01-01"
VAL_END = "2024-01-01"
NUM_EVAL_EPISODES = 5


def run_evaluation_episodes(model, env, num_episodes=NUM_EVAL_EPISODES):
    """
    Runs deterministic evaluation episodes and returns mean reward and mean Sharpe.

    The agent is run with deterministic=True (no exploration noise) to measure
    pure policy quality. Each episode produces:
      - total cumulative reward (sum of per-step rewards)
      - annualised Sharpe ratio (from daily portfolio returns)
    """
    vec_env = model.get_env()
    episode_rewards = []
    episode_sharpes = []

    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec_env.step(action)

            if dones[0]:
                done = True
                asset_memory = vec_env.get_attr("asset_memory")[0]
                rewards_memory = vec_env.get_attr("rewards_memory")[0]

                # Annualised Sharpe from daily portfolio returns
                daily_returns = np.diff(asset_memory) / np.array(asset_memory[:-1])
                if len(daily_returns) > 1 and np.std(daily_returns) > 1e-9:
                    sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
                else:
                    sharpe = 0.0

                episode_rewards.append(np.sum(rewards_memory))
                episode_sharpes.append(sharpe)

    return float(np.mean(episode_rewards)), float(np.mean(episode_sharpes))


def main():
    # -----------------------------------------------------------------
    # 1. Resolve paths relative to the project root (not CWD)
    # -----------------------------------------------------------------
    checkpoints_dir = os.path.join(_PROJECT_ROOT, "models", "checkpoints")
    champion_path = os.path.join(_PROJECT_ROOT, "models", "champion_model.zip")

    print(f"[*] Scanning for checkpoints in: {checkpoints_dir}")

    if not os.path.isdir(checkpoints_dir):
        print(f"[!] Error: Directory not found — {checkpoints_dir}")
        print("    Train a model first (python main.py) to generate checkpoints.")
        return

    checkpoint_files = sorted(glob.glob(os.path.join(checkpoints_dir, "*.zip")))
    if not checkpoint_files:
        print("[!] No .zip checkpoints found. Start training first.")
        return

    print(f"[*] Found {len(checkpoint_files)} checkpoint(s).\n")

    # -----------------------------------------------------------------
    # 2. Build the validation environment (holdout period)
    # -----------------------------------------------------------------
    print(f"[*] Fetching validation data ({VAL_START} to {VAL_END})...")
    df_val = fetch_and_process_data(VAL_START, VAL_END, TICKERS)
    env_val, _ = build_environment(df_val)

    # -----------------------------------------------------------------
    # 3. Evaluate every checkpoint deterministically
    # -----------------------------------------------------------------
    results = []
    mlflow.set_experiment("Champion_Selection")

    run_name = "Selection_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name):
        for idx, cp_path in enumerate(checkpoint_files, 1):
            cp_name = os.path.basename(cp_path)
            print(f"[{idx}/{len(checkpoint_files)}] Evaluating {cp_name}...")

            try:
                model = PPO.load(cp_path, env=env_val)
                reward, sharpe = run_evaluation_episodes(model, env_val)

                results.append({
                    "path": cp_path,
                    "name": cp_name,
                    "reward": reward,
                    "sharpe": sharpe,
                })
                print(f"    -> Reward: {reward:.2f}  |  Sharpe: {sharpe:.4f}")

            except Exception as e:
                # Dimension Guard: catches architecture mismatches from
                # stale checkpoints trained with a different state space.
                print(f"    [!] Skipping {cp_name}: {type(e).__name__}: {e}")

        # -------------------------------------------------------------
        # 4. Select and promote the champion
        # -------------------------------------------------------------
        if not results:
            print("\n[!] All checkpoints failed evaluation. Nothing to promote.")
            return

        results.sort(key=lambda r: r["sharpe"], reverse=True)
        champion = results[0]

        print("\n" + "=" * 55)
        print("  EVALUATION RESULTS (ranked by Sharpe)")
        print("=" * 55)
        for r in results:
            marker = " ← CHAMPION" if r is champion else ""
            print(f"  {r['name']:40s}  Sharpe={r['sharpe']:+.4f}  Reward={r['reward']:+.2f}{marker}")
        print("=" * 55)

        os.makedirs(os.path.dirname(champion_path), exist_ok=True)
        shutil.copy2(champion["path"], champion_path)
        print(f"\n[✓] Champion promoted to: {champion_path}")

        # Log for traceability
        mlflow.log_params({"champion_file": champion["name"]})
        mlflow.log_metrics({
            "best_sharpe": champion["sharpe"],
            "best_reward": champion["reward"],
        })


if __name__ == "__main__":
    main()

import os
import optuna
import mlflow
import numpy as np
from sb3_contrib import MaskablePPO as PPO
from src.environment import build_environment

def make_objective(df):
    """
    Returns an Optuna objective function bound to the specific market data.

    Expanded search space for long-horizon portfolio optimization:
    - learning_rate : Controls gradient step size
    - batch_size    : Mini-batch size for PPO updates
    - n_steps       : Rollout buffer length
    - ent_coef      : Entropy coefficient — forces exploration of diverse allocations
    - gamma         : Discount factor — extremely high values (0.999+) make the agent
                      prioritize long-term survival over immediate daily gains
    """
    def optimize_ppo(trial):
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            # --- Core PPO Hyperparameters ---
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            n_steps = trial.suggest_categorical("n_steps", [2048, 4096])

            # --- Exploration: Entropy Coefficient ---
            # Forces the agent to explore diverse asset allocations rather than
            # converging on a single repetitive strategy (e.g., always buying NIFTY).
            ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.01, log=True)

            # --- Long-Horizon: Discount Factor (Gamma) ---
            # Controls how much the agent values future rewards vs immediate ones.
            # For 30-year portfolio simulation, gamma must be very high (0.999+)
            # so the agent cares about compounding survival, not just today's P&L.
            gamma = trial.suggest_float("gamma", 0.99, 0.9999, log=True)

            # Build Env
            env, _ = build_environment(df)

            # Initialize PPO with full expanded parameter set
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_steps=n_steps,
                ent_coef=ent_coef,
                gamma=gamma,
                verbose=0,
            )

            # Train for 50k timesteps per trial — long enough to measure
            # whether a hyperparameter set actually enables meaningful learning.
            # (Previous 2k was too short to differentiate good vs bad configs.)
            model.learn(total_timesteps=50_000)

            # Pseudo-evaluation (In a real scenario, use a validation split environment)
            vec_env = model.get_env()
            obs = vec_env.reset()
            total_rewards = []
            for _ in range(500):
                action_masks = np.array(vec_env.env_method("action_masks"))
                action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, rewards, dones, info = vec_env.step(action)
                total_rewards.append(rewards[0])

            mean_reward = np.mean(total_rewards)

            # Log all parameters to MLflow for trial isolation
            mlflow.log_params({
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_steps": n_steps,
                "ent_coef": ent_coef,
                "gamma": gamma,
            })
            mlflow.log_metric("trial_mean_reward", mean_reward)

            return mean_reward

    return optimize_ppo

def train_final_model(df, best_params):
    """
    Trains the final PPO model with Optuna-optimized parameters and saves it.

    Safety strategy for i5-13th Gen / RTX 4050 hardware:
    1. Saves a checkpoint every 250k steps to models/checkpoints/
    2. Runs initial 500k steps first, saves a backup .zip
    3. Continues training to 3M total steps with ongoing checkpoints
    If thermal throttling kills the process, the latest checkpoint survives.
    """
    from stable_baselines3.common.callbacks import CheckpointCallback

    print("\nTraining Final PPO Agent with optimized parameters...")
    env, _ = build_environment(df)

    final_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        n_steps=best_params["n_steps"],
        ent_coef=best_params.get("ent_coef", 0.001),
        gamma=best_params.get("gamma", 0.999),
        verbose=1,
        tensorboard_log="./logs/"
    )

    # Setup checkpoint callback: saves a .zip every 250k steps
    os.makedirs("models/checkpoints", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=250_000,
        save_path="./models/checkpoints/",
        name_prefix="liminal_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # --- Phase 1: Initial 500k step safety run ---
    # Get a usable model on disk ASAP before committing to the full marathon.
    print("\n[Phase 1/2] Training initial 1M steps (safety backup)...")
    final_model.learn(total_timesteps=1_000_000, callback=checkpoint_cb)

    os.makedirs("models", exist_ok=True)
    backup_path = "models/liminal_ppo_agent_backup"
    final_model.save(backup_path)
    print(f"  Backup saved to {backup_path}.zip")
    mlflow.log_artifact(f"{backup_path}.zip")

    # --- Phase 2: Continue to 3M total steps ---
    # SB3's learn() with reset_num_timesteps=False continues from where Phase 1 left off.
    print("\n[Phase 2/2] Continuing training to 10M total steps...")
    final_model.learn(
        total_timesteps=10_000_000,
        callback=checkpoint_cb,
        reset_num_timesteps=False,
    )

    # Export final brain
    export_path = "models/liminal_ppo_agent"
    final_model.save(export_path)
    print(f"Model successfully saved to {export_path}.zip")

    # Log to MLflow
    mlflow.log_artifact(f"{export_path}.zip")

    return final_model

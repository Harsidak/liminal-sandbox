import os
import optuna
import mlflow
import numpy as np
from sb3_contrib import MaskablePPO as PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from src.environment import build_environment


# =====================================================================
# SubprocVecEnv Factory
# =====================================================================
# SubprocVecEnv requires a list of *callables* (not pre-built envs).
# Each callable must construct a completely independent environment
# inside its own subprocess — sharing a single env object across
# processes would cause memory corruption.
# =====================================================================

def _make_env_fn(df, env_kwargs_override=None):
    """
    Returns a zero-argument callable that builds a fresh StrategistTradingEnv.
    This is the factory pattern required by SubprocVecEnv.
    """
    def _init():
        env, _ = build_environment(df)
        return env
    return _init


def make_parallel_env(df, n_envs=8):
    """
    Creates a SubprocVecEnv with `n_envs` independent environment workers.

    Each worker gets its own copy of the dataframe and runs in a separate
    process. On an i5-13th Gen (12+ logical cores), n_envs=8 is the sweet
    spot — leaves headroom for the OS and the main training process.

    Falls back to DummyVecEnv (single-process) if SubprocVecEnv fails
    (e.g. on systems where fork/spawn is restricted).
    """
    env_fns = [_make_env_fn(df) for _ in range(n_envs)]
    try:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        print(f"  [✓] SubprocVecEnv created with {n_envs} parallel workers (spawn)")
    except Exception as e:
        print(f"  [!] SubprocVecEnv failed ({e}). Falling back to DummyVecEnv.")
        vec_env = DummyVecEnv(env_fns)
    return vec_env


class MLflowCallback(BaseCallback):
    """
    Custom callback to stream live episode rewards and training lengths
    directly to the MLflow UI so you don't have to rely solely on Tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log every 2000 steps to keep the UI snappy and avoid API limits
        if self.n_calls % 2000 == 0:
            if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
                ep_reward = self.model.ep_info_buffer[-1]["r"]
                ep_length = self.model.ep_info_buffer[-1]["l"]
                
                # Live single-episode metric
                mlflow.log_metric("train/ep_reward_last", ep_reward, step=self.num_timesteps)
                mlflow.log_metric("train/ep_length", ep_length, step=self.num_timesteps)
                
                # Smoothed 100-episode moving average (key proxy for actual learning!)
                ep_rews = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                mlflow.log_metric("train/ep_reward_mean100", np.mean(ep_rews), step=self.num_timesteps)
        return True


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
    - n_epochs      : Number of PPO update epochs per rollout (higher = more gradient steps)
    - clip_range    : PPO clipping parameter — controls policy update magnitude
    """
    def optimize_ppo(trial):
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            # --- Core PPO Hyperparameters ---
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
            n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])

            # --- Exploration: Entropy Coefficient ---
            # Forces the agent to explore diverse asset allocations rather than
            # converging on a single repetitive strategy (e.g., always buying NIFTY).
            ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)

            # --- Long-Horizon: Discount Factor (Gamma) ---
            # Controls how much the agent values future rewards vs immediate ones.
            # For 30-year portfolio simulation, gamma must be very high (0.999+)
            # so the agent cares about compounding survival, not just today's P&L.
            gamma = trial.suggest_float("gamma", 0.99, 0.9999, log=True)

            # --- PPO Update Depth ---
            n_epochs = trial.suggest_int("n_epochs", 5, 20)
            clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

            # Build single env for Optuna trials (parallel is overkill for 50k steps)
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
                n_epochs=n_epochs,
                clip_range=clip_range,
                verbose=0,
                device="auto",
            )

            # Train for 50k timesteps per trial with live MLflow streaming
            mlflow_cb = MLflowCallback()
            model.learn(total_timesteps=50_000, callback=mlflow_cb)

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
                "n_epochs": n_epochs,
                "clip_range": clip_range,
            })
            mlflow.log_metric("trial_mean_reward", mean_reward)

            return mean_reward

    return optimize_ppo

def train_final_model(df, best_params, n_envs=8, total_timesteps=10_000_000):
    """
    Trains the final PPO model with Optuna-optimized parameters and saves it.

    Production strategy for i5-13th Gen / RTX 4050 hardware:
    1. Creates a SubprocVecEnv with `n_envs` parallel environment workers
    2. Saves a checkpoint every 250k steps to models/checkpoints/
    3. Runs initial 1M steps first, saves a backup .zip
    4. Continues training to `total_timesteps` with ongoing checkpoints

    IMPORTANT: With SubprocVecEnv, effective throughput is multiplied by n_envs.
    SB3 counts total_timesteps as the sum across ALL environments, so the wall-
    clock time is roughly (total_timesteps / n_envs / steps_per_second).
    """
    print(f"\nTraining Final PPO Agent with optimized parameters...")
    print(f"  Parallel workers: {n_envs}")
    print(f"  Target timesteps: {total_timesteps:,}")

    # Build parallel environment
    vec_env = make_parallel_env(df, n_envs=n_envs)

    final_model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        n_steps=best_params["n_steps"],
        ent_coef=best_params.get("ent_coef", 0.001),
        gamma=best_params.get("gamma", 0.999),
        n_epochs=best_params.get("n_epochs", 10),
        clip_range=best_params.get("clip_range", 0.2),
        verbose=1,
        tensorboard_log="./logs/",
        device="auto",
    )

    # Setup checkpoint callback: saves a .zip every 250k steps
    os.makedirs("models/checkpoints", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(250_000 // n_envs, 1),   # SB3 counts per-env steps for callbacks
        save_path="./models/checkpoints/",
        name_prefix="liminal_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Combine standard checkpoint saving with our live MLflow streaming
    callback = CallbackList([checkpoint_cb, MLflowCallback()])

    # --- Phase 1: Initial 1M step safety run ---
    # Get a usable model on disk ASAP before committing to the full marathon.
    warmup_steps = min(1_000_000, total_timesteps)
    print(f"\n[Phase 1/2] Training initial {warmup_steps:,} steps (safety backup)...")
    final_model.learn(total_timesteps=warmup_steps, callback=callback)

    os.makedirs("models", exist_ok=True)
    backup_path = "models/liminal_ppo_agent_backup"
    final_model.save(backup_path)
    print(f"  Backup saved to {backup_path}.zip")
    mlflow.log_artifact(f"{backup_path}.zip")

    # --- Phase 2: Continue to total_timesteps ---
    # SB3's learn() with reset_num_timesteps=False continues from where Phase 1 left off.
    # The total_timesteps here is the GLOBAL target — SB3 will stop once num_timesteps reaches it.
    remaining = total_timesteps - warmup_steps
    if remaining > 0:
        print(f"\n[Phase 2/2] Continuing training to {total_timesteps:,} total steps...")
        final_model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )

    # Export final brain
    export_path = "models/liminal_ppo_agent"
    final_model.save(export_path)
    print(f"Model successfully saved to {export_path}.zip")

    # Log to MLflow
    mlflow.log_artifact(f"{export_path}.zip")

    # Clean up SubprocVecEnv workers
    vec_env.close()

    return final_model

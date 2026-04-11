import os
import optuna
import mlflow
import numpy as np
from stable_baselines3 import PPO
from src.environment import build_environment

def make_objective(df):
    """
    Returns an Optuna objective function bound to the specific market data.
    """
    def optimize_ppo(trial):
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            n_steps = trial.suggest_categorical("n_steps", [2048, 4096])
            
            # Build Env
            env, _ = build_environment(df)
            
            # Initialize PPO
            model = PPO("MlpPolicy", env, learning_rate=learning_rate, batch_size=batch_size, n_steps=n_steps, verbose=0)
            
            # Train for a very short duration just for the Optuna trial
            model.learn(total_timesteps=2000) 
            
            # Pseudo-evaluation (In a real scenario, use a validation split environment)
            vec_env = model.get_env()
            obs = vec_env.reset()
            total_rewards = []
            for _ in range(500):
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = vec_env.step(action)
                total_rewards.append(rewards[0])
            
            mean_reward = np.mean(total_rewards)
            
            # MLflow automatically tracking within nested trial
            mlflow.log_params({"learning_rate": learning_rate, "batch_size": batch_size, "n_steps": n_steps})
            mlflow.log_metric("trial_mean_reward", mean_reward)
            
            return mean_reward

    return optimize_ppo

def train_final_model(df, best_params):
    """
    Trains the final PPO model with optimized parameters and saves it.
    """
    print("\nTraining Final PPO Agent with optimized parameters...")
    env, _ = build_environment(df)
    
    final_model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=best_params["learning_rate"], 
        batch_size=best_params["batch_size"], 
        n_steps=best_params["n_steps"], 
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # Train for longer time (10k timesteps here for sandbox testing)
    final_model.learn(total_timesteps=10000) 
    
    # Export Brain
    os.makedirs("models", exist_ok=True)
    export_path = "models/liminal_ppo_agent"
    final_model.save(export_path)
    print(f"Model successfully saved to {export_path}.zip")
    
    # Log to MLflow
    mlflow.log_artifact(f"{export_path}.zip")
    
    return final_model

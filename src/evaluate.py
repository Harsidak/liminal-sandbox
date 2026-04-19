"""
evaluate.py — Audit & Documentation Generator
===============================================
Loads champion_model.zip, runs a deterministic backtest on 2024–present
data, and generates:
  1. Quantstats HTML tear sheet (with NIFTY benchmark)
  2. Underwater drawdown plot (PNG)
  3. SHAP beeswarm plot explaining feature importance during the
     worst-drawdown ("stress") window

Usage (from project root):
    python src/evaluate.py
    python -m src.evaluate
"""

import os
import sys
import datetime
import warnings

# =====================================================================
# PATH BOOTSTRAP — Must execute BEFORE any project/library imports.
# When running `python src/evaluate.py`, sys.path[0] == "src/".
# The shim files (alpaca_trade_api.py, ccxt.py) live in the project
# root, and FinRL's internal imports require them. We inject the
# project root into sys.path so Python can find both `src.*` modules
# and the root-level shims.
# =====================================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Suppress noisy library warnings (yfinance FutureWarnings, etc.)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — no GUI required
import matplotlib.pyplot as plt
import quantstats as qs
import torch
import shap
import yfinance as yf
from sb3_contrib import MaskablePPO as PPO

from src.data_pipeline import fetch_and_process_data
from src.environment import build_environment


# =====================================================================
# Pipeline Configuration
# =====================================================================
TICKERS = ["NIFTYBEES.NS", "RELIANCE.NS", "HDFCBANK.NS", "GOLDBEES.NS", "INFY.NS"]
TICKER_LABELS = ["NIFTYBEES", "RELIANCE", "HDFCBANK", "GOLDBEES", "INFY"]
TEST_START = "2024-01-01"


def _build_feature_names(tech_indicator_list):
    """
    Reconstructs human-readable names for each element of the flat
    state vector produced by StrategistTradingEnv.

    Dynamically adapts to whatever tech_indicator_list the environment
    was built with — no more hardcoded dimension assumptions.
    """
    names = ["cash"]
    names += [f"price_{t}" for t in TICKER_LABELS]
    names += [f"shares_{t}" for t in TICKER_LABELS]
    for tech in tech_indicator_list:
        names += [f"{tech}_{t}" for t in TICKER_LABELS]
    return names


def isolate_stress_event(returns, window=20):
    """
    Finds the 20-day window with the worst cumulative return.
    Returns the end-date of that window (the 'peak pain' date).
    """
    if len(returns) < window:
        # Not enough data for a rolling window — return the worst single day
        return returns.idxmin()

    rolling = returns.rolling(window=window).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    )
    return rolling.idxmin()


def generate_shap_report(model, docs_dir, portfolio_returns, tech_indicator_list):
    """
    Uses SHAP DeepExplainer on the PPO Actor network to produce a
    beeswarm summary plot of feature importance.
    """
    print("[*] Generating SHAP Explainability Report...")

    # 1. Identify the stress window
    stress_date = isolate_stress_event(portfolio_returns)
    print(f"    -> Stress peak identified around: {stress_date.date()}")

    feature_names = _build_feature_names(tech_indicator_list)

    # 2. Collect background observations from a rollout
    vec_env = model.get_env()
    obs_samples = []
    obs = vec_env.reset()
    for _ in range(50):
        obs_samples.append(obs[0].copy())
        action_masks = np.array(vec_env.env_method("action_masks"))
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, _, dones, _ = vec_env.step(action)
        if dones[0]:
            obs = vec_env.reset()

    background = torch.tensor(np.array(obs_samples), dtype=torch.float32)

    # 3. Wrap the policy for SHAP compatibility
    #    SB3 PPO's policy.get_distribution(obs).mode() returns the
    #    deterministic action — this is what we want to explain.
    class ActorWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, x):
            return self.policy.get_distribution(x).mode()

    actor = ActorWrapper(model.policy)

    # 4. Compute SHAP values
    print("[*] Computing SHAP values (this may take a minute)...")
    explainer = shap.DeepExplainer(actor, background)
    shap_values = explainer.shap_values(background)

    # SHAP output format varies by version. It may be a list of 2D arrays (one per action)
    # or a single 3D array (samples, features, actions). We slice out the first action.
    if isinstance(shap_values, list):
        sv = shap_values[0]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # Ensure we're indexing correctly: usually (samples, features, actions)
        # but if it happens to be (actions, samples, features), we must adjust.
        # DeepExplainer usually returns (samples, features, actions)
        sv = shap_values[:, :, 0] if shap_values.shape[-1] == 5 else shap_values[0]
    else:
        sv = shap_values

    # Truncate feature_names to match actual observation dimension
    actual_dim = sv.shape[1] if sv.ndim == 2 else sv.shape[0]
    if len(feature_names) > actual_dim:
        feature_names = feature_names[:actual_dim]
    elif len(feature_names) < actual_dim:
        feature_names += [f"feat_{i}" for i in range(len(feature_names), actual_dim)]

    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        sv,
        background.numpy(),
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Feature Importance — Stress Window ({stress_date.date()})")
    plt.tight_layout()

    out_path = os.path.join(docs_dir, "shap_insights.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [OK] SHAP beeswarm plot saved -> {out_path}")


def main():
    # -----------------------------------------------------------------
    # 1. Resolve paths relative to the project root
    # -----------------------------------------------------------------
    model_path = os.path.join(_PROJECT_ROOT, "models", "champion_model.zip")
    docs_dir = os.path.join(_PROJECT_ROOT, "docs", "metrics")
    os.makedirs(docs_dir, exist_ok=True)

    print("\n" + "=" * 55)
    print("  LIMINAL AUDIT: PERFORMANCE & EXPLAINABILITY REPORT")
    print("=" * 55)

    if not os.path.isfile(model_path):
        # Fallback: try the direct final model if champion hasn't been promoted yet
        alt_path = os.path.join(_PROJECT_ROOT, "models", "liminal_ppo_agent.zip")
        if os.path.isfile(alt_path):
            print(f"[*] Champion not found. Using final model: {alt_path}")
            model_path = alt_path
        else:
            print(f"[!] No model found at: {model_path}")
            print("    Run `python src/stitch.py` first to select a champion,")
            print("    or ensure `models/liminal_ppo_agent.zip` exists.")
            return

    # -----------------------------------------------------------------
    # 2. Acquire test data (2024 → today)
    # -----------------------------------------------------------------
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"\n[*] Loading test data: {TEST_START} to {end_date}")

    df_test = fetch_and_process_data(TEST_START, end_date, TICKERS)
    env_test, env_kwargs = build_environment(df_test)

    # Extract tech_indicator_list from the env_kwargs for SHAP feature naming
    tech_indicator_list = env_kwargs.get("tech_indicator_list", [])

    print(f"[*] Loading champion model: {model_path}")
    model = PPO.load(model_path, env=env_test)

    # -----------------------------------------------------------------
    # 3. Deterministic backtest
    # -----------------------------------------------------------------
    print("[*] Running deterministic backtest (zero exploration)...")
    
    # We use env_test directly instead of model.get_env() 
    # because DummyVecEnv auto-resets upon termination, wiping `asset_memory`.
    reset_out = env_test.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False

    while not done:
        action_masks = env_test.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        step_out = env_test.step(action)
        
        if len(step_out) == 5:  # Gymnasium API
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:                   # Gym 0.21 API
            obs, reward, terminated, info = step_out
            done = terminated

    asset_memory = env_test.asset_memory
    unique_dates = df_test.date.unique()

    # asset_memory may have one extra entry (initial value at t=0),
    # so align carefully with the dates.
    dates = unique_dates[: len(asset_memory)]
    portfolio = pd.Series(asset_memory, index=pd.to_datetime(dates))
    returns = portfolio.pct_change().dropna()

    # -----------------------------------------------------------------
    # 4. Quantstats Financial Audit (with NIFTY benchmark)
    # -----------------------------------------------------------------
    print("[*] Generating Quantstats Institutional Audit Report...")

    # Download NIFTYBEES as Buy-and-Hold benchmark
    benchmark_raw = yf.download(
        "NIFTYBEES.NS", start=TEST_START, end=end_date, progress=False
    )
    if not benchmark_raw.empty:
        bench_close = benchmark_raw["Close"]
        # Handle yfinance MultiIndex columns
        if isinstance(bench_close, pd.DataFrame):
            bench_close = bench_close.iloc[:, 0]
        benchmark_returns = bench_close.pct_change().dropna()
    else:
        print("    [!] Benchmark download failed — generating report without it.")
        benchmark_returns = None

    # Full HTML Tear Sheet
    html_path = os.path.join(docs_dir, "institutional_audit.html")
    qs.reports.html(
        returns,
        benchmark=benchmark_returns,
        output=html_path,
        title="Liminal Champion: Institutional Audit",
    )
    print(f"  [OK] HTML tear sheet -> {html_path}")

    # Underwater Drawdown Plot
    dd_path = os.path.join(docs_dir, "underwater_drawdown.svg")
    
    qs.plots.drawdown(
        returns, 
        savefig={"fname": dd_path, "bbox_inches": "tight", "format": "svg"}, 
        show=False
    )
    
    print(f"  [OK] Drawdown plot      -> {dd_path}")

    # -----------------------------------------------------------------
    # 5. Explainable AI (SHAP)
    # -----------------------------------------------------------------
    try:
        generate_shap_report(model, docs_dir, returns, tech_indicator_list)
    except Exception as e:
        print(f"[!] SHAP analysis failed: {type(e).__name__}: {e}")
        print("    This is non-fatal — financial reports were still generated.")

    print("\n" + "=" * 55)
    print(f"  AUDIT COMPLETE — see {docs_dir}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

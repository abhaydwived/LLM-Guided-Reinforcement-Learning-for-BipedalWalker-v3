# =============================================================================
# main_loop.py — Automated reward optimisation loop
# =============================================================================
# Usage:
#   python main_loop.py
#
# The loop runs MAX_ITERATIONS times:
#   1. Train PPO agent with current reward function
#   2. Evaluate locomotion performance
#   3. Build LLM prompt from metrics
#   4. Call LLM to generate improved reward
#   5. Save metrics to logs/metrics_history.json
#   6. Repeat
# =============================================================================

import json
import os
import sys

# Ensure project root is on the Python path for all submodule imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MAX_ITERATIONS,
    TRAINING_TIMESTEPS,
    EVALUATION_EPISODES,
    REWARD_FILE,
    METRICS_LOG,
    LOG_DIR,
    RENDER_DEMO,          # NEW: show simulation window after each iteration?
)
from rl.train       import train
from rl.test_policy import evaluate_policy
from llm.prompt_builder   import build_prompt
from llm.reward_generator import generate_reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metrics_history() -> list:
    """Load existing metrics log; return empty list if not found."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.isfile(METRICS_LOG):
        return []
    with open(METRICS_LOG, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_metrics_history(history: list) -> None:
    with open(METRICS_LOG, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _print_metrics(metrics: dict, iteration: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Iteration {iteration} — Evaluation Metrics")
    print(f"{'='*60}")
    for key, val in metrics.items():
        print(f"  {key:<30} {val:.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  LLM-Guided RL Reward Optimisation for BipedalWalker-v3")
    print("=" * 60 + "\n")

    history = _load_metrics_history()

    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'#'*60}")
        print(f"#  ITERATION {iteration + 1} / {MAX_ITERATIONS}")
        print(f"{'#'*60}\n")

        # ------------------------------------------------------------------
        # Step 1: Train
        # ------------------------------------------------------------------
        model = train(
            iteration   = iteration,
            timesteps   = TRAINING_TIMESTEPS,
            reward_file = REWARD_FILE,
        )

        # ------------------------------------------------------------------
        # Step 2: Evaluate
        # ------------------------------------------------------------------
        print(f"\n[main_loop] Evaluating policy over {EVALUATION_EPISODES} episodes …")
        # render_demo_episodes=1  → show 1 rendered episode in a pygame window
        # set RENDER_DEMO=False in config.py to disable (e.g. on a headless server)
        metrics = evaluate_policy(
            model                = model,
            n_episodes           = EVALUATION_EPISODES,
            reward_file          = REWARD_FILE,
            render_demo_episodes = 1 if RENDER_DEMO else 0,
            iteration            = iteration,
        )
        _print_metrics(metrics, iteration)

        # ------------------------------------------------------------------
        # Step 3: Log metrics
        # ------------------------------------------------------------------
        log_entry = {"iteration": iteration, "reward_version": iteration, **metrics}
        history.append(log_entry)
        _save_metrics_history(history)
        print(f"[main_loop] Metrics logged → {METRICS_LOG}")

        # ------------------------------------------------------------------
        # Step 4: Build prompt
        # ------------------------------------------------------------------
        prompt = build_prompt(metrics, iteration=iteration)

        # ------------------------------------------------------------------
        # Step 5: Generate improved reward via LLM
        # ------------------------------------------------------------------
        print("[main_loop] Requesting new reward function from LLM …")
        success = generate_reward(prompt, iteration=iteration)

        if success:
            print("[main_loop] Reward updated successfully.")
        else:
            print("[main_loop] Reward generation failed — continuing with existing reward.")

        # Free memory (the next iteration will retrain from scratch)
        del model

    print("\n" + "=" * 60)
    print(f"  Experiment complete after {MAX_ITERATIONS} iterations.")
    print(f"  Metrics log : {METRICS_LOG}")
    print(f"  Models      : models/")
    print(f"  Rewards     : rewards/archive/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

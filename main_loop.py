# =============================================================================
# main_loop.py — Automated reward optimisation loop
# =============================================================================
# Usage:
#   python main_loop.py
#
# The loop runs MAX_ITERATIONS times:
#   1. Train SAC agent for TRAINING_TIMESTEPS steps
#      (resumes from previous model — training is NEVER restarted)
#   2. Evaluate locomotion performance with rich gait metrics
#   3. Build LLM prompt from metrics
#   4. Call LLM to generate improved reward
#   5. Save metrics to logs/metrics_history.json
#   6. Repeat — new reward takes effect on next episode reset automatically
#
# Resume behaviour
# ----------------
#   If saved models already exist (e.g. from a previous run up to iter 9):
#     → The archived reward for that iteration is restored into
#       rewards/current_reward.py so training continues with the exact
#       reward the LLM produced at that checkpoint.
#     → Training resumes from iteration N+1 (skipping already-done iters).
#
#   If NO saved models exist (fresh start):
#     → rewards/current_reward.py is used as-is (your hand-tuned baseline).
#     → Training starts from iteration 0.
# =============================================================================

import argparse
import glob
import json
import os
import shutil
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
    RENDER_DEMO,
    USE_HARD_ENV,
    TERRAIN_DIFFICULTY_LEVEL,
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


def _find_latest_saved_iteration() -> int:
    """
    Scan the models/ directory for the highest SAC checkpoint index.
    Returns the iteration number (0-based) or -1 if no models exist.
    """
    pattern = os.path.join("models", "sac_bipedal_iter_*.zip")
    zips = glob.glob(pattern)
    if not zips:
        return -1
    indices = []
    for z in zips:
        basename = os.path.splitext(os.path.basename(z))[0]  # sac_bipedal_iter_N
        try:
            idx = int(basename.split("_iter_")[-1])
            indices.append(idx)
        except ValueError:
            pass
    return max(indices) if indices else -1


def _restore_reward_for_iteration(iteration: int) -> bool:
    """
    Copy rewards/archive/reward_iter_{iteration}.py → rewards/current_reward.py.
    Returns True if the archive file existed and was copied, False otherwise.
    """
    from config import REWARD_FILE, REWARD_ARCHIVE
    archive_path = os.path.join(REWARD_ARCHIVE, f"reward_iter_{iteration}.py")
    if os.path.isfile(archive_path):
        shutil.copy2(archive_path, REWARD_FILE)
        print(f"[main_loop] Restored reward from archive: {archive_path} → {REWARD_FILE}")
        return True
    print(f"[main_loop] WARNING: archived reward not found at {archive_path}. "
          f"Keeping current {REWARD_FILE} unchanged.")
    return False


def _save_metrics_history(history: list) -> None:
    with open(METRICS_LOG, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _print_metrics(metrics: dict, iteration: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Iteration {iteration + 1} — Evaluation Metrics")
    print(f"{'='*60}")
    for key, val in metrics.items():
        print(f"  {key:<35} {val:.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-Guided RL Reward Optimisation")
    parser.add_argument(
        "--start-iter", type=int, default=None,
        help="Iteration to start/resume from (auto-detected if omitted)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LLM-Guided RL Reward Optimisation for BipedalWalker-v3")
    print("  Training is CONTINUOUS — model is never reset between iterations.")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Auto-detect resume point
    # ------------------------------------------------------------------
    latest_iter = _find_latest_saved_iteration()

    if args.start_iter is not None:
        # User explicitly specified where to start — honour it
        start_iter = args.start_iter
        print(f"[main_loop] --start-iter={start_iter} supplied explicitly.")
    elif latest_iter >= 0:
        # Previous training exists → resume from the NEXT iteration
        start_iter = latest_iter + 1
        print(f"[main_loop] Found existing checkpoint up to iteration {latest_iter}.")
        print(f"[main_loop] AUTO-RESUMING from iteration {start_iter}.")
    else:
        # Completely fresh start
        start_iter = 0
        print("[main_loop] No prior checkpoints found — starting fresh from iteration 0.")

    # ------------------------------------------------------------------
    # Restore the correct reward function
    # ------------------------------------------------------------------
    if latest_iter >= 0 and args.start_iter is None:
        # Resumed run: restore the reward that corresponds to the last
        # iteration that was actually trained, so the LLM picks up from
        # the exact reward it produced at that checkpoint.
        print(f"[main_loop] Restoring reward for last completed iteration ({latest_iter}) …")
        _restore_reward_for_iteration(latest_iter)
    else:
        # Fresh start or explicit override: use current_reward.py as-is
        print(f"[main_loop] Using current rewards/current_reward.py as starting reward.")

    history = _load_metrics_history()

    # The model object is passed through iterations so training is truly
    # continuous.  On the very first step (or after a fresh restart) it
    # will be None and train() will create / load an appropriate checkpoint.
    current_model = None

    for iteration in range(start_iter, MAX_ITERATIONS):
        print(f"\n{'#'*60}")
        print(f"#  ITERATION {iteration + 1} / {MAX_ITERATIONS}")
        print(f"{'#'*60}\n")

        # ------------------------------------------------------------------
        # Step 1: Terrain difficulty 
        # ------------------------------------------------------------------
        # Use the difficulty level defined by the user in config.
        # This ensures the agent is trained and evaluated on the actual terrain we are testing.
        difficulty = TERRAIN_DIFFICULTY_LEVEL

        print(f"[main_loop] Terrain difficulty this iteration: {difficulty:.2f}")

        # ------------------------------------------------------------------
        # Step 2: Train (or continue training)
        #
        # We pass `model=current_model`.  On iteration 0  → None  → fresh start.
        # On subsequent iterations → the trained object is handed back in and
        # learning continues right where it left off (replay buffer preserved).
        # ------------------------------------------------------------------
        current_model = train(
            iteration        = iteration,
            timesteps        = TRAINING_TIMESTEPS,
            reward_file      = REWARD_FILE,
            difficulty_level = difficulty,
            model            = current_model,   # ← pass live model for continuity
        )

        # ------------------------------------------------------------------
        # Step 3: Evaluate
        # ------------------------------------------------------------------
        print(f"\n[main_loop] Evaluating policy over {EVALUATION_EPISODES} episodes …")
        metrics = evaluate_policy(
            model                = current_model,
            n_episodes           = EVALUATION_EPISODES,
            reward_file          = REWARD_FILE,
            render_demo_episodes = 1 if RENDER_DEMO else 0,
            iteration            = iteration,
            difficulty_level     = difficulty,
        )
        _print_metrics(metrics, iteration)

        # ------------------------------------------------------------------
        # Step 4: Log metrics
        # ------------------------------------------------------------------
        log_entry = {"iteration": iteration, "reward_version": iteration, **metrics}
        history.append(log_entry)
        _save_metrics_history(history)
        print(f"[main_loop] Metrics logged → {METRICS_LOG}")

        # ------------------------------------------------------------------
        # Step 5: Build prompt from rich metrics
        # ------------------------------------------------------------------
        prompt = build_prompt(metrics, iteration=iteration, history=history, difficulty=difficulty)

        # ------------------------------------------------------------------
        # Step 6: Generate improved reward via LLM
        # ------------------------------------------------------------------
        print("[main_loop] Requesting new reward function from LLM …")
        success = generate_reward(prompt, iteration=iteration)

        if success:
            print("[main_loop] Reward updated successfully.")
            print("[main_loop] New reward will take effect on the next episode reset "
                  "(training continues without interruption).")
        else:
            print("[main_loop] Reward generation failed — continuing with existing reward.")

        # NOTE: We do NOT delete current_model.
        # The reward wrapper reloads the reward file on every episode reset,
        # so the new reward takes effect automatically in the next training
        # segment with NO restart needed.

    print("\n" + "=" * 60)
    print(f"  Experiment complete after {MAX_ITERATIONS} iterations.")
    print(f"  Metrics log : {METRICS_LOG}")
    print(f"  Models      : models/")
    print(f"  Rewards     : rewards/archive/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

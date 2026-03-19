# =============================================================================
# rl/test_policy.py — Evaluate policy (headless) + optional live render demo
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from env.bipedal_env import make_env
from evaluation.metrics import compute_metrics
from config import EVALUATION_EPISODES, REWARD_FILE


# ---------------------------------------------------------------------------
# Live render helper
# ---------------------------------------------------------------------------

def render_demo(model: PPO,
                reward_file: str = REWARD_FILE,
                n_episodes: int = 1,
                episode_label: str = "") -> None:
    """
    Play n_episodes in "human" mode (pygame window) using the trained model.
    Call this after evaluate_policy() to visually inspect the learned behaviour.

    Args:
        model:         trained SB3 PPO model
        reward_file:   active reward file
        n_episodes:    how many rendered episodes to show
        episode_label: optional string shown in the console (e.g. "Iteration 2")
    """
    label = f" [{episode_label}]" if episode_label else ""
    print(f"\n[render_demo]{label} Opening BipedalWalker simulation window …")
    print("  Close the window or wait for the episode(s) to finish.\n")

    env = make_env(reward_file=reward_file, render_mode="human")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        total_r = 0.0
        steps   = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps   += 1
            done = terminated or truncated

        status = "FELL" if terminated else "time-limit"
        print(f"  Demo ep {ep + 1}/{n_episodes}: "
              f"{steps} steps | total reward = {total_r:.2f} | end = {status}")

    env.close()
    print("[render_demo] Window closed.\n")


# ---------------------------------------------------------------------------
# Headless metric evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(model: PPO,
                    n_episodes: int = EVALUATION_EPISODES,
                    reward_file: str = REWARD_FILE,
                    render_demo_episodes: int = 1,
                    iteration: int = 0) -> dict:
    """
    Roll out the trained policy for n_episodes (headless) to collect metrics,
    then open a render window for render_demo_episodes so you can watch the walker.

    Args:
        model:                trained SB3 PPO model
        n_episodes:           silent evaluation episodes (for metrics)
        reward_file:          active reward file (passed to wrapper)
        render_demo_episodes: episodes to show in the pygame window (0 = skip)
        iteration:            current loop iteration (used in console label)

    Returns:
        Dictionary of aggregated performance metrics
    """
    # --- Headless metric collection ---
    env = make_env(reward_file=reward_file, render_mode=None)
    episodes_data = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False

        obs_list    = []
        action_list = []
        rewards     = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_list.append(obs)
            action_list.append(action)
            rewards.append(reward)

            obs  = next_obs
            done = terminated or truncated

        episodes_data.append({
            "observations": np.array(obs_list),
            "actions":      np.array(action_list),
            "rewards":      np.array(rewards),
            "fell":         terminated,
        })

    env.close()
    metrics = compute_metrics(episodes_data)

    # --- Optional live render ---
    if render_demo_episodes > 0:
        render_demo(
            model         = model,
            reward_file   = reward_file,
            n_episodes    = render_demo_episodes,
            episode_label = f"Iteration {iteration}",
        )

    return metrics

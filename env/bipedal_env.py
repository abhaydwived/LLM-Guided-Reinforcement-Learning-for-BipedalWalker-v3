# =============================================================================
# env/bipedal_env.py — Factory that creates the wrapped BipedalWalker-v3 env
# =============================================================================

import sys
import os

# Allow imports from project root when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from rl.reward_wrapper import RewardWrapper
from config import ENV_ID, REWARD_FILE


def make_env(reward_file: str = REWARD_FILE, render_mode: str = None) -> gym.Env:
    """
    Create BipedalWalker-v3 and apply the custom reward wrapper.

    Args:
        reward_file: path to the active reward Python file
        render_mode:  None (headless) | "human" | "rgb_array"

    Returns:
        Wrapped Gymnasium environment
    """
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = RewardWrapper(env, reward_file=reward_file)
    return env

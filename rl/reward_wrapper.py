# =============================================================================
# rl/reward_wrapper.py — Gymnasium wrapper that injects a custom reward
# =============================================================================
# The active reward function is always loaded from:
#   rewards/current_reward.py  →  compute_reward(obs, action)
#
# Reloading happens automatically at the START of each episode so that a
# new reward generated mid-run takes effect on the next reset().
# =============================================================================

import importlib.util
import sys
import os

import gymnasium as gym
import numpy as np


def _load_reward_fn(reward_file: str):
    """
    Dynamically import compute_reward from the given .py file.
    Returns the callable, or raises ImportError with a clear message.
    """
    abs_path = os.path.abspath(reward_file)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Reward file not found: {abs_path}")

    spec   = importlib.util.spec_from_file_location("_current_reward", abs_path)
    module = importlib.util.module_from_spec(spec)

    # Remove cached version so we always get the freshest file
    sys.modules.pop("_current_reward", None)
    spec.loader.exec_module(module)

    if not hasattr(module, "compute_reward"):
        raise ImportError(
            f"'compute_reward(obs, action)' not found in {abs_path}"
        )
    return module.compute_reward


class RewardWrapper(gym.Wrapper):
    """
    Wraps BipedalWalker-v3 and replaces the native reward with
    compute_reward(obs, action) loaded from `reward_file`.

    The reward file is reloaded at every episode reset so that
    LLM-updated rewards take effect without restarting training.
    """

    def __init__(self, env: gym.Env, reward_file: str):
        super().__init__(env)
        self.reward_file  = reward_file
        self._reward_fn   = _load_reward_fn(reward_file)
        self._last_obs    = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reload reward in case the LLM updated the file since last episode
        try:
            self._reward_fn = _load_reward_fn(self.reward_file)
        except Exception as exc:
            print(f"[RewardWrapper] Warning: could not reload reward – {exc}")
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, _native_reward, terminated, truncated, info = self.env.step(action)

        # Compute custom reward
        try:
            reward = float(self._reward_fn(self._last_obs, action))
        except Exception as exc:
            print(f"[RewardWrapper] Error in compute_reward: {exc}. Falling back to 0.")
            reward = 0.0

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

# =============================================================================
# env/bipedal_env.py — Factory that creates the correct env variant
# =============================================================================
# When USE_HARD_ENV is True in config.py, this returns a HardBipedalEnv
# wrapped by a Gymnasium TimeLimit so it stays fully compatible with SB3.
# When False, it falls back to vanilla BipedalWalker-v3 + RewardWrapper.
# =============================================================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from rl.reward_wrapper import RewardWrapper
from config import (
    ENV_ID,
    REWARD_FILE,
    USE_HARD_ENV,
    ENABLE_TERRAIN_GENERATION,
    ENABLE_DISTURBANCE,
    TERRAIN_DIFFICULTY_LEVEL,
    CHUNK_LENGTH,
    MAX_HEIGHT_VARIATION,
    SLOPE_RANGE,
    STEP_HEIGHT,
    DISTURBANCE_FORCE_RANGE,
    DISTURBANCE_FREQUENCY,
    RANDOMISE_TERRAIN,
)


def make_env(
    reward_file: str = REWARD_FILE,
    render_mode: str = None,
    difficulty_level: float = TERRAIN_DIFFICULTY_LEVEL,
) -> gym.Env:
    """
    Create the training/evaluation environment.

    If USE_HARD_ENV is True:
        Returns HardBipedalEnv (procedural terrain + disturbances)
        wrapped with TimeLimit (1600 steps, matching vanilla default)
        and then RewardWrapper for the LLM-driven custom reward.

    If USE_HARD_ENV is False:
        Returns vanilla BipedalWalker-v3 + RewardWrapper (original behaviour).

    Parameters
    ----------
    reward_file      : path to the active reward .py file
    render_mode      : None | "human" | "rgb_array"
    difficulty_level : terrain difficulty override (0.0–1.0)
    """
    if USE_HARD_ENV:
        from env.hard_bipedal_env import HardBipedalEnv

        base_env = HardBipedalEnv(
            render_mode               = render_mode,
            enable_terrain_generation = ENABLE_TERRAIN_GENERATION,
            enable_disturbance        = ENABLE_DISTURBANCE,
            terrain_difficulty_level  = difficulty_level,
            chunk_length              = CHUNK_LENGTH,
            max_height_variation      = MAX_HEIGHT_VARIATION,
            slope_range               = tuple(SLOPE_RANGE),
            step_height               = STEP_HEIGHT,
            disturbance_force_range   = tuple(DISTURBANCE_FORCE_RANGE),
            disturbance_frequency     = DISTURBANCE_FREQUENCY,
            randomise_terrain         = RANDOMISE_TERRAIN,
        )

        # Wrap with TimeLimit — increased to 3200 steps for faster/longer walking
        env = TimeLimit(base_env, max_episode_steps=5000)
    else:
        env = gym.make(ENV_ID, render_mode=render_mode)

    # Apply LLM-driven reward wrapper on top
    env = RewardWrapper(env, reward_file=reward_file)
    return env


def make_hard_env_with_difficulty(
    difficulty_level: float,
    reward_file: str = REWARD_FILE,
    render_mode: str = None,
) -> gym.Env:
    """
    Convenience factory: create HardBipedalEnv at a specific difficulty level.
    Useful for curriculum training where difficulty ramps across iterations.
    """
    return make_env(
        reward_file      = reward_file,
        render_mode      = render_mode,
        difficulty_level = difficulty_level,
    )

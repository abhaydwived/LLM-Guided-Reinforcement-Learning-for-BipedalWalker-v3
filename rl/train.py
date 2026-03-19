# =============================================================================
# rl/train.py — Train a PPO agent on the wrapped BipedalWalker-v3
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.bipedal_env import make_env
from config import TRAINING_TIMESTEPS, MODEL_DIR, REWARD_FILE


def train(iteration: int = 0,
          timesteps: int = TRAINING_TIMESTEPS,
          reward_file: str = REWARD_FILE) -> PPO:
    """
    Train a PPO agent and save the model.

    Args:
        iteration:    current loop iteration index (used in the save name)
        timesteps:    total environment steps for this training run
        reward_file:  path to the active reward Python file

    Returns:
        Trained PPO model
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # SB3 expects a vectorised environment
    env = DummyVecEnv([lambda: make_env(reward_file=reward_file)])

    model = PPO(
        policy      = "MlpPolicy",
        env         = env,
        verbose     = 1,
        n_steps     = 2048,
        batch_size  = 64,
        n_epochs    = 10,
        gamma       = 0.99,
        gae_lambda  = 0.95,
        clip_range  = 0.2,
        ent_coef    = 0.0,
        learning_rate = 3e-4,
    )

    print(f"\n[train] Starting PPO training — iteration {iteration}, "
          f"{timesteps:,} timesteps …")
    model.learn(total_timesteps=timesteps)

    save_path = os.path.join(MODEL_DIR, f"ppo_bipedal_iter_{iteration}")
    model.save(save_path)
    print(f"[train] Model saved → {save_path}.zip")

    env.close()
    return model

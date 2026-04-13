# =============================================================================
# rl/train.py — Train a SAC agent on the wrapped BipedalWalker-v3
# =============================================================================
# Supports two modes:
#   • Fresh start (iteration 0):  creates a brand-new SAC model.
#   • Resume      (iteration > 0): loads the previous model and continues
#     learning with reset_num_timesteps=False so the replay buffer,
#     optimiser state, and total step count are all preserved.
#
# This means training is NEVER restarted between LLM reward updates —
# the agent keeps its accumulated knowledge and simply adapts to the
# new reward function.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env.bipedal_env import make_env
from config import (
    TRAINING_TIMESTEPS, MODEL_DIR, REWARD_FILE,
    TERRAIN_DIFFICULTY_LEVEL, RENDER_TRAINING, N_ENVS,
)


class RenderTrainingCallback(BaseCallback):
    """
    Callback for rendering the environment live during SAC training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.training_env.render()
        return True


def _build_env(reward_file: str, difficulty_level: float, render_mode):
    """Create a (possibly parallel) vectorised training environment."""

    def _make(rf, dl, rm):
        return lambda: make_env(reward_file=rf, difficulty_level=dl, render_mode=rm)

    env_fns = [_make(reward_file, difficulty_level, render_mode) for _ in range(N_ENVS)]

    if N_ENVS > 1:
        env = SubprocVecEnv(env_fns)
        print(f"[train] Using SubprocVecEnv with {N_ENVS} parallel workers.")
    else:
        env = DummyVecEnv(env_fns)
        print("[train] Using DummyVecEnv (single worker).")
    return env


def train(
    iteration: int = 0,
    timesteps: int = TRAINING_TIMESTEPS,
    reward_file: str = REWARD_FILE,
    difficulty_level: float = TERRAIN_DIFFICULTY_LEVEL,
    model: SAC = None,
) -> SAC:
    """
    Train (or continue training) a SAC agent and save the model.

    Args:
        iteration:        current loop iteration index (used in the save name)
        timesteps:        environment steps to run THIS segment
        reward_file:      path to the active reward Python file
        difficulty_level: terrain difficulty override (0.0–1.0)
        model:            an already-initialised SAC model to resume from.
                          If None AND a saved model for (iteration-1) exists,
                          that checkpoint is loaded automatically.
                          If None and no checkpoint exists, a fresh model is created.

    Returns:
        Trained SAC model (same object if resumed, new object if fresh)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"[train] Terrain difficulty level: {difficulty_level:.2f}")
    render_mode = "human" if RENDER_TRAINING else None

    env = _build_env(reward_file, difficulty_level, render_mode)

    # ── Resolve model: resume or fresh start ─────────────────────────────
    if model is not None:
        # Caller passed an already-loaded model — swap its env and keep going
        print(f"[train] Resuming training from caller-supplied model (iteration {iteration}).")
        model.set_env(env)
        reset_timesteps = False

    else:
        # Try to load the checkpoint saved in the previous iteration
        prev_checkpoint = os.path.join(MODEL_DIR, f"sac_bipedal_iter_{iteration - 1}.zip")
        if iteration > 0 and os.path.isfile(prev_checkpoint):
            print(f"[train] Loading checkpoint from previous iteration: {prev_checkpoint}")
            model = SAC.load(prev_checkpoint, env=env)
            # Reload replay buffer if it was saved alongside the model
            prev_buffer = os.path.join(MODEL_DIR, f"sac_bipedal_iter_{iteration - 1}_buffer.pkl")
            if os.path.isfile(prev_buffer):
                print(f"[train] Restoring replay buffer from: {prev_buffer}")
                model.load_replay_buffer(prev_buffer)
            reset_timesteps = False   # ← key: don't reset internal step counter
        else:
            print(f"[train] No checkpoint found — starting fresh SAC model.")
            model = SAC(
                policy          = "MlpPolicy",
                env             = env,
                verbose         = 1,
                learning_rate   = 3e-4,
                buffer_size     = 1_000_000,
                batch_size      = 256,
                ent_coef        = "auto",
                gamma           = 0.99,
                tau             = 0.005,
                train_freq      = 1,
                gradient_steps  = 1,
            )
            reset_timesteps = True

    # ── Learn ──────────────────────────────────────────────────────────────
    print(f"\n[train] SAC training — iteration {iteration}, "
          f"{timesteps:,} timesteps, resume={not reset_timesteps} …")

    callbacks = []
    if RENDER_TRAINING:
        print("[train] RENDER_TRAINING is ON. Live simulation will be shown.")
        callbacks.append(RenderTrainingCallback())

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,   # False = continue wall-clock counter
    )

    # ── Save checkpoint + replay buffer ────────────────────────────────────
    save_path = os.path.join(MODEL_DIR, f"sac_bipedal_iter_{iteration}")
    model.save(save_path)
    print(f"[train] Model saved → {save_path}.zip")

    # Save replay buffer so the next iteration can restore it
    buffer_path = os.path.join(MODEL_DIR, f"sac_bipedal_iter_{iteration}_buffer.pkl")
    try:
        model.save_replay_buffer(buffer_path)
        print(f"[train] Replay buffer saved → {buffer_path}")
    except Exception as exc:
        print(f"[train] Warning: could not save replay buffer — {exc}")

    env.close()
    return model

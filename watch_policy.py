# =============================================================================
# watch_policy.py — Load a saved model and watch it in the pygame window
# =============================================================================
# Usage:
#   python watch_policy.py                        # latest saved model
#   python watch_policy.py --iter 3               # specific iteration model
#   python watch_policy.py --model models/my.zip  # custom model path
#   python watch_policy.py --episodes 5           # show 5 episodes
# =============================================================================

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio
import numpy as np
from stable_baselines3 import PPO

from env.bipedal_env import make_env
from rl.test_policy import render_demo
from config import MODEL_DIR, REWARD_FILE


def find_latest_model() -> str:
    """Return the path of the most recently modified .zip in models/."""
    pattern = os.path.join(MODEL_DIR, "*.zip")
    zips = glob.glob(pattern)
    if not zips:
        raise FileNotFoundError(
            f"No .zip models found in '{MODEL_DIR}'. "
            "Run python main_loop.py first to train a model."
        )
    return max(zips, key=os.path.getmtime)


def save_demo_gif(model: PPO, reward_file: str, filename: str = "demo.gif", max_steps: int = 1500) -> None:
    """Run one episode in rgb_array mode and save the frames as a GIF."""
    print(f"[watch_policy] Running episode to save GIF to {filename} ...")
    env = make_env(reward_file=reward_file, render_mode="rgb_array")
    
    obs, _ = env.reset()
    done = False
    frames = []
    
    steps = 0
    while not done and steps < max_steps:
        # Render frame
        frame = env.render()
        frames.append(frame)
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    env.close()
    
    print(f"[watch_policy] Saving {len(frames)} frames to {filename} ...")
    imageio.mimsave(filename, frames, fps=50, loop=0) # BipedalWalker-v3 typically runs at 50fps, loop=0 makes it repeat infinitely
    print("[watch_policy] GIF saved successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained BipedalWalker PPO policy in simulation."
    )
    parser.add_argument(
        "--iter", type=int, default=None,
        help="Iteration index to load (e.g. --iter 2 loads models/ppo_bipedal_iter_2.zip)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Direct path to a .zip model file (overrides --iter)"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of rendered episodes to show (default: 3)"
    )
    parser.add_argument(
        "--gif", type=str, default=None,
        help="Path to save a GIF of the simulation instead of opening a window (e.g. --gif demo.gif)"
    )
    args = parser.parse_args()

    # --- Resolve model path ---
    if args.model:
        model_path = args.model
    elif args.iter is not None:
        model_path = os.path.join(MODEL_DIR, f"ppo_bipedal_iter_{args.iter}.zip")
    else:
        model_path = find_latest_model()
        print(f"[watch_policy] Auto-selected latest model: {model_path}")

    if not os.path.isfile(model_path):
        print(f"[watch_policy] ERROR: Model file not found: {model_path}")
        sys.exit(1)

    print(f"[watch_policy] Loading model from: {model_path}")
    model = PPO.load(model_path)

    # --- Run rendered episodes or save GIF ---
    if args.gif:
        save_demo_gif(model=model, reward_file=REWARD_FILE, filename=args.gif)
    else:
        render_demo(
            model         = model,
            reward_file   = REWARD_FILE,
            n_episodes    = args.episodes,
            episode_label = os.path.basename(model_path),
        )


if __name__ == "__main__":
    main()

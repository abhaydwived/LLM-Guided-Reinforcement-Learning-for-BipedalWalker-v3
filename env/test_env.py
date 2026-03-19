# =============================================================================
# env/test_env.py — Sanity-check: creates env, runs random actions WITH render
# =============================================================================
# Run with:   python env/test_env.py
#
# A pygame window will open showing the BipedalWalker taking random steps.
# Close the window or wait for the steps to finish to end the script.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.bipedal_env import make_env


def test_environment(n_steps: int = 500, render: bool = True):
    """
    1. Create BipedalWalker-v3 with the custom reward wrapper.
    2. Open a render window (render_mode='human') so you can watch the walker.
    3. Step through n_steps random actions.
    4. Print obs shape, reward and termination flags every 50 steps.

    Args:
        n_steps: how many environment steps to run
        render:  True = open pygame window, False = headless
    """
    print("=" * 60)
    print("  Environment sanity check — BipedalWalker-v3")
    print("=" * 60)

    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)

    obs, info = env.reset(seed=42)
    print(f"  Observation shape : {obs.shape}")
    print(f"  Action space      : {env.action_space}")

    if render:
        print("\n  >>> A pygame window has opened — watch the walker! <<<\n")
    else:
        print()

    total_reward = 0.0
    episode      = 1

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            print(f"  ep={episode}  step={step:>4d}  reward={reward:+.4f}  "
                  f"terminated={terminated}  truncated={truncated}")

        if terminated or truncated:
            print(f"  Episode {episode} ended at step {step}. "
                  f"Resetting …\n")
            episode += 1
            obs, info = env.reset()

    env.close()
    print()
    print(f"  Total accumulated reward over {n_steps} random steps: {total_reward:.2f}")
    print("  Environment check PASSED.")


if __name__ == "__main__":
    # render=True   → opens the pygame window (default)
    # render=False  → headless, useful for CI / servers without a display
    test_environment(n_steps=500, render=True)

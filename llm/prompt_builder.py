# =============================================================================
# llm/prompt_builder.py — Convert metrics + optional hint → LLM prompt string
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import HUMAN_HINT_FILE


# ---------------------------------------------------------------------------
# Observation reference included in every prompt so the LLM knows the schema
# ---------------------------------------------------------------------------
_OBS_REFERENCE = """
BipedalWalker-v3 observation vector (24 values):
  obs[0]  — hull angle (rad, 0 = upright)
  obs[1]  — hull angular velocity
  obs[2]  — forward (x) velocity  [primary locomotion signal]
  obs[3]  — vertical (y) velocity
  obs[4]  — hip joint 1 angle
  obs[5]  — hip joint 1 angular speed
  obs[6]  — knee joint 1 angle
  obs[7]  — knee joint 1 angular speed
  obs[8]  — leg 1 ground contact  (bool-like: 0 or 1)
  obs[9]  — hip joint 2 angle
  obs[10] — hip joint 2 angular speed
  obs[11] — knee joint 2 angle
  obs[12] — knee joint 2 angular speed
  obs[13] — leg 2 ground contact  (bool-like: 0 or 1)
  obs[14..23] — 10 lidar range readings

Action vector (4 values, each in [-1, 1]):
  action[0] — hip 1 torque
  action[1] — knee 1 torque
  action[2] — hip 2 torque
  action[3] — knee 2 torque
"""


def _load_human_hint() -> str:
    """Read the optional researcher hint file. Returns '' if absent or empty."""
    try:
        with open(HUMAN_HINT_FILE, "r", encoding="utf-8") as f:
            hint = f.read().strip()
        # Ignore comment-only lines (lines starting with #)
        non_comment = [l for l in hint.splitlines()
                       if l.strip() and not l.strip().startswith("#")]
        return "\n".join(non_comment)
    except FileNotFoundError:
        return ""


def build_prompt(metrics: dict, iteration: int = 0) -> str:
    """
    Build a structured prompt string for the LLM reward-generation call.

    Args:
        metrics:   dict from evaluation/metrics.py
        iteration: current loop iteration (informational)

    Returns:
        A plain-text prompt string ready to send as the user message.
    """
    m = metrics   # shorthand

    hint_text = _load_human_hint()
    hint_section = (
        f"\nResearcher hint:\n{hint_text}\n" if hint_text else ""
    )

    prompt = f"""\
You are an expert reinforcement learning researcher designing a reward function \
for a bipedal walking robot in the Gymnasium BipedalWalker-v3 environment.

=== ITERATION {iteration} PERFORMANCE METRICS ===

  Average forward distance : {m.get('average_forward_distance', 0):.3f}  (normalised units)
  Average speed            : {m.get('average_speed', 0):.4f}  (normalised m/s)
  Fall rate                : {m.get('fall_rate', 1):.2%}  (fraction of episodes that fell)
  Mean torso tilt          : {m.get('torso_tilt', 0):.4f}  (rad, 0 = perfectly upright)
  Energy consumption       : {m.get('energy_consumption', 0):.2f}  (sum |torques| per episode)

=== GOAL ===

Design a reward function that drives the agent towards:
  1. Fast, stable forward walking (maximise forward speed).
  2. Upright posture (minimise hull tilt, avoid falling).
  3. Energy efficiency (penalise unnecessary large torques).
  4. Smooth locomotion (avoid jerky, oscillating movements).
{hint_section}
=== OBSERVATION / ACTION SCHEMA ===
{_OBS_REFERENCE}
=== INSTRUCTIONS ===

Return ONLY a Python code block containing the function below — no prose, no markdown fences, \
no imports other than numpy (already available as `np`):

def compute_reward(obs, action):
    # your reward logic here
    return reward

Rules:
  - The function must be named exactly `compute_reward`.
  - `obs` is a numpy array of shape (24,).
  - `action` is a numpy array of shape (4,).
  - Return a single Python float.
  - Do NOT include any other top-level code, classes, or imports except numpy.
"""
    return prompt

# =============================================================================
# llm/prompt_builder.py — Convert metrics + optional hint → LLM prompt string
# =============================================================================
# v2: Richer prompt with:
#   • Gait-specific metric interpretation guidance
#   • Trend analysis from full metrics history
#   • Grouped metric sections matching human-like gait goals
#   • Concrete interpretation notes so LLM can read the numbers correctly
# =============================================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import HUMAN_HINT_FILE


# ---------------------------------------------------------------------------
# BipedalWalker observation / action schema (static reference)
# ---------------------------------------------------------------------------
_OBS_REFERENCE = """\
BipedalWalker-v3 observation vector (24 values):
  obs[0]  — hull angle (rad, 0 = upright, + = forward lean, - = backward lean)
  obs[1]  — hull angular velocity  (+ = tilting forward)
  obs[2]  — forward (x) velocity   [primary locomotion signal, normalised]
  obs[3]  — vertical (y) velocity
  obs[4]  — hip joint 1 angle      (right leg; + = leg forward)
  obs[5]  — hip joint 1 angular speed
  obs[6]  — knee joint 1 angle     (right leg; + = knee bent)
  obs[7]  — knee joint 1 angular speed
  obs[8]  — leg 1 ground contact   (0 = air, 1 = ground)  [RIGHT foot]
  obs[9]  — hip joint 2 angle      (left leg;  + = leg forward)
  obs[10] — hip joint 2 angular speed
  obs[11] — knee joint 2 angle     (left leg;  + = knee bent)
  obs[12] — knee joint 2 angular speed
  obs[13] — leg 2 ground contact   (0 = air, 1 = ground)  [LEFT foot]
  obs[14..23] — 10 LIDAR range readings (terrain ahead)

Action vector (4 values, each ∈ [-1, 1]):
  action[0] — hip 1 torque   (right leg)
  action[1] — knee 1 torque  (right leg)
  action[2] — hip 2 torque   (left leg)
  action[3] — knee 2 torque  (left leg)
"""

# ---------------------------------------------------------------------------
# Metric interpretation guide (helps LLM reason about the numbers)
# ---------------------------------------------------------------------------
_METRIC_GUIDE = """\
=== METRIC INTERPRETATION GUIDE ===

Ideal values for human-like bipedal walking:

  [Basic Locomotion]
    average_forward_distance  → higher is better; >30 is good, <10 means near-failure
    average_speed             → 0.05–0.15 m/s normalised is reasonable walking speed
    fall_rate                 → 0.0 is perfect; 1.0 = always falls

  [Posture & Stability]
    torso_tilt                → ideal ≈ 0.05–0.15 rad (slight forward lean); >0.4 = unstable
    torso_tilt_variance       → should be very low (<0.01); high = wobbling torso
    lateral_stability         → mean |angular velocity|; should be <0.15 for smooth walking

  [Energy & Smoothness]
    energy_consumption        → lower = more efficient; decreasing trend is good
    action_smoothness         → mean |Δaction| per step; <0.1 = smooth, >0.3 = jerky
    specific_resistance       → energy / (speed × steps); lower = more efficient gait

  [Gait Cycle — MOST IMPORTANT FOR HUMAN-LIKE WALKING]
    gait_symmetry_index       → 0.0 = perfect L/R symmetry; >0.3 = asymmetric limp
    right_leg_swing_freq      → swing cycles per 100 steps; ideal ≈ 15–30
    left_leg_swing_freq       → should match right_leg_swing_freq for symmetric gait
    mean_swing_duration       → steps per swing; ideal ≈ 10–20 steps
    double_support_fraction   → fraction of steps with BOTH feet on ground; 0.1–0.2 is realistic
    single_support_fraction   → fraction of steps with ONE foot on ground; should be ~0.6–0.7
    flight_fraction           → fraction with BOTH feet off ground; ideally ≈ 0.0 for walking (>0.1 = running/hopping)

  [Hip & Knee Coordination]
    hip_knee_coordination     → |correlation| between hip and knee of same leg; >0.5 = coordinated swing
    hip_angle_range           → peak-to-trough hip swing (rad); >0.8 = good propulsive ROM
    stance_leg_straightness   → mean |knee angle| during stance; <0.3 = near-straight knee (efficient)

  [Contact Quality]
    contact_per_metre         → foot contacts per metre; ≈4–6 is realistic; >10 = shuffle
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_human_hint() -> str:
    """Read the optional researcher hint file. Returns '' if absent or empty."""
    try:
        with open(HUMAN_HINT_FILE, "r", encoding="utf-8") as f:
            hint = f.read().strip()
        non_comment = [l for l in hint.splitlines()
                       if l.strip() and not l.strip().startswith("#")]
        return "\n".join(non_comment)
    except FileNotFoundError:
        return ""


def _trend_line(history: list, key: str, n: int = 3) -> str:
    """
    Build a short trend string showing the last `n` values for `key`.
    E.g.  "0.3200 → 0.2800 → 0.2400  (↓ improving)"
    """
    vals = [h.get(key) for h in history if h.get(key) is not None]
    if len(vals) < 2:
        return "  (no trend data yet)"

    vals = vals[-n:]
    arrow = "→"
    trend_str = f" {arrow} ".join(f"{v:.4f}" for v in vals)

    delta = vals[-1] - vals[0]
    # For fall_rate, tilt, energy, gait_sym → lower is better
    lower_is_better = {
        "fall_rate", "torso_tilt", "torso_tilt_variance", "lateral_stability",
        "energy_consumption", "action_smoothness", "specific_resistance",
        "gait_symmetry_index", "stance_leg_straightness", "flight_fraction",
        "contact_per_metre",
    }
    # Determine if the metric is improving
    key_base = key.replace("average_", "")
    improving = (delta < 0) if key in lower_is_better else (delta > 0)
    tag = "↑ improving" if improving else ("↓ worsening" if not improving else "→ stable")
    return f"{trend_str}  ({tag})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(metrics: dict, iteration: int = 0, history: list = None, difficulty: float = 0.0) -> str:
    """
    Build a structured, context-rich prompt string for the LLM reward-generation call.

    Args:
        metrics:   dict from evaluation/metrics.py
        iteration: current loop iteration (informational)
        history:   full list of past metric dicts (used for trend analysis)
        difficulty: the terrain difficulty level the agent is currently testing on.

    Returns:
        A plain-text prompt string ready to send as the user message.
    """
    m = metrics
    hist = history or []
    hint_text = _load_human_hint()
    hint_section = (
        f"\nResearcher hint:\n{hint_text}\n" if hint_text else ""
    )

    # ── Trend table (only rendered if we have history) ────────────────────
    trend_keys = [
        "average_forward_distance", "average_speed", "fall_rate",
        "torso_tilt", "gait_symmetry_index", "double_support_fraction",
        "single_support_fraction", "flight_fraction", "energy_consumption",
        "action_smoothness", "hip_angle_range", "stance_leg_straightness",
        "hip_knee_coordination",
    ]
    if len(hist) >= 2:
        trend_rows = "\n".join(
            f"  {k:<35} {_trend_line(hist, k)}"
            for k in trend_keys if k in m
        )
        trend_section = f"\n=== TREND ANALYSIS (last {min(3, len(hist))} iterations) ===\n{trend_rows}\n"
    else:
        trend_section = ""

    # ── Format current metrics grouped by category ────────────────────────
    def _fmt(key, fmt=".4f"):
        val = m.get(key, 0.0)
        return format(val, fmt)

    prompt = f"""\
You are an expert reinforcement-learning researcher specialising in \
locomotion control for 2-D bipedal robots.

Your task is to write an improved `compute_reward(obs, action)` function \
for a custom BipedalWalker environment. The agent is NOT walking on normal flat terrain. \
It is currently being tested on a custom procedural terrain with difficulty level {difficulty:.2f} \
(where 0.0 = flat, >0.3 = uneven ground, >0.55 = inclines/declines, >0.75 = steps/stairs). \
Your reward function must help the agent adapt to this actual terrain while maintaining \
a smooth, human-like gait: right-leg swing / left-leg stance, then left-leg swing / right-leg stance.

=== ITERATION {iteration + 1} — CURRENT PERFORMANCE METRICS ===

  ── Basic Locomotion ─────────────────────────────────────────────────────────
    average_forward_distance  : {_fmt('average_forward_distance')}  (normalised m)
    average_speed             : {_fmt('average_speed')}  (normalised m/s)
    fall_rate                 : {_fmt('fall_rate', '.2%')}  (0 = never falls)

  ── Posture & Stability ──────────────────────────────────────────────────────
    torso_tilt                : {_fmt('torso_tilt')}  rad  (0 = perfectly upright)
    torso_tilt_variance       : {_fmt('torso_tilt_variance')}  (low = steady posture)
    lateral_stability         : {_fmt('lateral_stability')}  (mean |angular velocity|)

  ── Energy & Smoothness ──────────────────────────────────────────────────────
    energy_consumption        : {_fmt('energy_consumption')}  (Σ|torques|)
    action_smoothness         : {_fmt('action_smoothness')}  (mean |Δaction|; lower = smoother)
    specific_resistance       : {_fmt('specific_resistance')}  (energy / dist×time; lower = better)

  ── Gait Cycle ───────────────────────────────────────────────────────────────
    gait_symmetry_index       : {_fmt('gait_symmetry_index')}  (0 = perfect bilateral symmetry)
    right_leg_swing_freq      : {_fmt('right_leg_swing_freq')}  swings / 100 steps
    left_leg_swing_freq       : {_fmt('left_leg_swing_freq')}  swings / 100 steps
    mean_swing_duration       : {_fmt('mean_swing_duration')}  steps per swing phase
    double_support_fraction   : {_fmt('double_support_fraction', '.3f')}  (both feet on ground)
    single_support_fraction   : {_fmt('single_support_fraction', '.3f')}  (one foot on ground)
    flight_fraction           : {_fmt('flight_fraction', '.3f')}  (both feet off ground; 0 = walking)

  ── Hip & Knee Coordination ──────────────────────────────────────────────────
    hip_knee_coordination     : {_fmt('hip_knee_coordination')}  (|corr|; >0.5 = well-coordinated)
    hip_angle_range           : {_fmt('hip_angle_range')}  rad  (propulsion ROM; >0.8 = good)
    stance_leg_straightness   : {_fmt('stance_leg_straightness')}  (mean knee angle in stance; <0.3 = straight)

  ── Contact Quality ──────────────────────────────────────────────────────────
    contact_per_metre         : {_fmt('contact_per_metre')}  foot contacts / m  (4–6 = realistic)
{trend_section}
=== GOALS (in order of priority) ===

  1. ELIMINATE FALLS ON THIS TERRAIN  (fall_rate → 0)
  2. ACHIEVE HUMAN-LIKE GAIT CYCLE over the custom terrain
       • Regular alternating swing/stance: one leg swings while the other is planted
       • gait_symmetry_index → 0  (both legs equally active)
       • single_support_fraction ≈ 0.65, double_support_fraction ≈ 0.15
       • flight_fraction ≈ 0 (walking, not hopping)
  3. FORWARD PROGRESS  (average_speed ↑)
  4. UPRIGHT POSTURE   (torso_tilt ↓, torso_tilt_variance ↓)
  5. ENERGY EFFICIENCY (energy_consumption ↓, specific_resistance ↓)
  6. SMOOTH ACTIONS    (action_smoothness ↓)
  7. GOOD HIP-KNEE COORDINATION  (hip_knee_coordination ↑, hip_angle_range ↑)
{hint_section}
=== OBSERVATION / ACTION SCHEMA ===
{_OBS_REFERENCE}

{_METRIC_GUIDE}

=== INSTRUCTIONS ===

Return ONLY the Python code block below, wrapped in ```python fences.
No conversational prose, no comments outside the function, no explanations.
You may ONLY use numpy (imported as `np`). Do not import anything else.

```python
def compute_reward(obs, action):
    # your reward logic here
    return reward
```

Rules:
  - The function must be named exactly `compute_reward`.
  - `obs` is a numpy array of shape (24,).  Use the index reference above.
  - `action` is a numpy array of shape (4,).
  - Return a single Python float (or numpy scalar that can be cast to float).
  - Do NOT include any other top-level code, classes, or imports except numpy.
  - Keep the function focused and readable; avoid magic constants without explanation.
"""
    return prompt

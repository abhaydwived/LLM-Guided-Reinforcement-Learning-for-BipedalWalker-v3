# =============================================================================
# evaluation/metrics.py — Compute locomotion metrics from episode rollout data
# =============================================================================
# Input:  list of episode dicts produced by rl/test_policy.py
#         Each dict has keys:
#           observations  np.ndarray  shape (T, 24)
#           actions       np.ndarray  shape (T, 4)
#           rewards       np.ndarray  shape (T,)
#           fell          bool         True if episode ended by fall
#
# BipedalWalker observation indices used here:
#   obs[0]  — hull angle (rad)     0 = perfectly upright
#   obs[2]  — horizontal speed     (normalised, scale ≈ 1 unit/step roughly)
#   obs[3]  — vertical speed
# =============================================================================

import numpy as np
from typing import List, Dict


# Approximate metres per environment unit for distance estimation.
# BipedalWalker's speed is normalised; 1 unit ≈ 0.3 m/s at 50 fps → tune if needed.
_SPEED_SCALE = 0.3   # m/s per normalised speed unit


def compute_metrics(episodes_data: List[Dict]) -> Dict[str, float]:
    """
    Aggregate locomotion metrics over all evaluation episodes.

    Args:
        episodes_data: list of episode rollout dicts (see module docstring)

    Returns:
        Dictionary with scalar metrics:
          average_forward_distance  — mean total x-distance per episode   (m)
          average_speed             — mean forward speed across all steps  (m/s)
          fall_rate                 — fraction of episodes that fell       (0–1)
          torso_tilt                — mean absolute hull angle             (rad)
          energy_consumption        — mean sum of |torque| per episode
    """
    if not episodes_data:
        return {
            "average_forward_distance": 0.0,
            "average_speed":            0.0,
            "fall_rate":                1.0,
            "torso_tilt":               0.0,
            "energy_consumption":       0.0,
        }

    forward_distances = []
    mean_speeds       = []
    falls             = []
    tilts             = []
    energies          = []

    for ep in episodes_data:
        obs     = ep["observations"]    # (T, 24)
        actions = ep["actions"]         # (T, 4)
        fell    = ep["fell"]

        T = len(obs)
        if T == 0:
            continue

        # --- Forward speed (normalised) ---
        speeds_x = obs[:, 2]            # horizontal speed column

        # Distance ≈ integral of speed × dt (dt = 1/50 s at 50 fps in gym)
        # We use normalised units here and apply a scale factor
        dt = 1.0 / 50.0
        forward_dist = float(np.sum(speeds_x) * dt * _SPEED_SCALE * 50)
        mean_spd     = float(np.mean(speeds_x) * _SPEED_SCALE)

        # --- Torso tilt ---
        hull_angles = obs[:, 0]         # hull angle column
        mean_tilt   = float(np.mean(np.abs(hull_angles)))

        # --- Energy consumption (sum of absolute joint torques over episode) ---
        energy = float(np.sum(np.abs(actions)))

        forward_distances.append(forward_dist)
        mean_speeds.append(mean_spd)
        falls.append(int(fell))
        tilts.append(mean_tilt)
        energies.append(energy)

    metrics = {
        "average_forward_distance": float(np.mean(forward_distances)),
        "average_speed":            float(np.mean(mean_speeds)),
        "fall_rate":                float(np.mean(falls)),
        "torso_tilt":               float(np.mean(tilts)),
        "energy_consumption":       float(np.mean(energies)),
    }
    return metrics

# =============================================================================
# rewards/current_reward.py — Active reward function used by the RL agent
# =============================================================================
# This file is OVERWRITTEN by the LLM each iteration.
# Archived copies are saved in rewards/archive/reward_iter_X.py
#
# Observation vector (24 dims) for BipedalWalker-v3:
#   [0]  hull angle
#   [1]  hull angular velocity
#   [2]  vel x
#   [3]  vel y
#   [4]  hip joint 1 angle
#   [5]  hip joint 1 speed
#   [6]  knee joint 1 angle
#   [7]  knee joint 1 speed
#   [8]  leg 1 ground contact
#   [9]  hip joint 2 angle
#   [10] hip joint 2 speed
#   [11] knee joint 2 angle
#   [12] knee joint 2 speed
#   [13] leg 2 ground contact
#   [14..23] 10 lidar readings
#
# Action vector (4 dims): torques for [hip1, knee1, hip2, knee2]
# =============================================================================

import numpy as np


def compute_reward(obs, action):
    """
    Baseline reward function for BipedalWalker-v3.

    Encourages:
      - Forward velocity (primary signal)
      - Upright posture (low hull tilt)
      - Low energy consumption (small torques)
    """
    vel_x        = float(obs[2])   # forward speed (normalised)
    hull_angle   = float(obs[0])   # hull tilt from upright
    energy       = float(np.sum(np.abs(action)))

    # Forward progress is strongly rewarded
    forward_reward = vel_x * 2.0

    # Penalise leaning (angle is ~0 when upright)
    tilt_penalty = abs(hull_angle) * 0.5

    # Small penalty for high joint torques (energy efficiency)
    energy_penalty = energy * 0.1

    reward = forward_reward - tilt_penalty - energy_penalty
    return reward

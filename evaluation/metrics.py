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
#   obs[0]  — hull angle (rad)        0 = perfectly upright
#   obs[1]  — hull angular velocity
#   obs[2]  — horizontal speed        (normalised)
#   obs[3]  — vertical speed
#   obs[4]  — hip joint 1 angle       (leg 1 = right leg, + = leg forward)
#   obs[5]  — hip joint 1 angular speed
#   obs[6]  — knee joint 1 angle      (right leg, + = knee bent)
#   obs[7]  — knee joint 1 angular speed
#   obs[8]  — leg 1 ground contact    (0 or 1)  [RIGHT foot]
#   obs[9]  — hip joint 2 angle       (leg 2 = left leg, + = leg forward)
#   obs[10] — hip joint 2 angular speed
#   obs[11] — knee joint 2 angle      (left leg, + = knee bent)
#   obs[12] — knee joint 2 angular speed
#   obs[13] — leg 2 ground contact    (0 or 1)  [LEFT foot]
#   obs[14..23] — 10 lidar range readings
#
# Action vector (4 values, each in [-1, 1]):
#   action[0] — hip 1 torque  (right leg)
#   action[1] — knee 1 torque (right leg)
#   action[2] — hip 2 torque  (left leg)
#   action[3] — knee 2 torque (left leg)
# =============================================================================

import numpy as np
from typing import List, Dict


# Approximate metres per environment unit for distance estimation.
_SPEED_SCALE = 0.3   # m/s per normalised speed unit

# Contact threshold
_CONTACT_THRESH = 0.5

# Minimum swing phase length in steps to count as a valid stride
_MIN_SWING_STEPS = 3


# ---------------------------------------------------------------------------
# Phase detection helpers
# ---------------------------------------------------------------------------

def _detect_phases(contact: np.ndarray):
    """
    Given a binary contact signal (1 = on ground, 0 = in air),
    return arrays of swing-phase lengths and stance-phase lengths (in steps).
    """
    swing_lens, stance_lens = [], []
    in_swing = False
    swing_count = 0
    stance_count = 0
    for c in contact:
        on_ground = c >= _CONTACT_THRESH
        if on_ground:
            if in_swing and swing_count >= _MIN_SWING_STEPS:
                swing_lens.append(swing_count)
            in_swing = False
            swing_count = 0
            stance_count += 1
        else:
            if not in_swing:
                if stance_count > 0:
                    stance_lens.append(stance_count)
                stance_count = 0
                in_swing = True
                swing_count = 0
            swing_count += 1
    # Edge: episode ended mid-swing
    if in_swing and swing_count >= _MIN_SWING_STEPS:
        swing_lens.append(swing_count)
    if not in_swing and stance_count > 0:
        stance_lens.append(stance_count)
    return np.array(swing_lens, dtype=float), np.array(stance_lens, dtype=float)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returns 0 if either signal is constant."""
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Main metrics function
# ---------------------------------------------------------------------------

def compute_metrics(episodes_data: List[Dict]) -> Dict[str, float]:
    """
    Aggregate locomotion metrics over all evaluation episodes.

    Returns
    -------
    Dictionary of scalar metrics critical for human-like bipedal gait:

    ── Basic locomotion ──────────────────────────────────────────────
      average_forward_distance   mean total x-distance per episode
      average_speed              mean forward speed across steps
      fall_rate                  fraction of episodes that ended in a fall

    ── Posture & stability ───────────────────────────────────────────
      torso_tilt                 mean |hull angle| (rad)
      torso_tilt_variance        variance of hull angle (wobble)
      lateral_stability          mean |hull angular velocity|

    ── Energy & smoothness ───────────────────────────────────────────
      energy_consumption         mean Σ|torques| per episode
      action_smoothness          mean |Δaction| step-to-step (jerk)
      specific_resistance        energy / (speed × episode_length)

    ── Gait cycle (alternating swing / stance) ───────────────────────
      gait_symmetry_index        |stride_R – stride_L| / mean  (0 = perfect)
      right_leg_swing_freq       swing cycles per 100 steps (right)
      left_leg_swing_freq        swing cycles per 100 steps (left)
      mean_swing_duration        mean swing phase length (steps)
      mean_stance_duration       mean stance phase length (steps)
      double_support_fraction    fraction with BOTH feet on ground
      single_support_fraction    fraction with exactly ONE foot on ground
      flight_fraction            fraction with BOTH feet off ground

    ── Phase alternation (right-swing / left-stance and vice versa) ──
      alternation_score          how well R-swing coincides with L-stance
                                   and L-swing with R-stance (0–1, 1=perfect)
      phase_lag_steps            mean step-lag between right and left swing
                                   onsets (ideal ≈ half stride period)

    ── Hip & knee coordination ────────────────────────────────────────
      hip_knee_coordination      |Pearson corr| hip↔knee same leg (mean)
      inter_leg_hip_anticorr     –(Pearson corr) R-hip vs L-hip  (1=antiphase)
      hip_angle_range            mean peak-to-trough hip swing (rad)
      hip_extension_at_pushoff   mean hip angle when foot leaves ground
                                   (negative = leg pushed behind, good)
      hip_flexion_at_swing_peak  mean hip angle at mid-swing (positive = good)
      knee_flexion_at_swing_peak mean knee angle at mid-swing (positive = good)

    ── Foot clearance & ground contact quality ───────────────────────
      stance_leg_straightness    mean |knee angle| during stance (lower = straight)
      contact_per_metre          foot contacts per forward metre
      step_length_symmetry       |mean_step_len_R – mean_step_len_L| /
                                   mean_step_len  (0 = symmetric steps)
    """
    if not episodes_data:
        return _empty_metrics()

    # Per-episode accumulators
    forward_distances      = []
    mean_speeds            = []
    falls                  = []
    tilts                  = []
    tilt_vars              = []
    lateral_stabs          = []
    energies               = []
    smoothnesses           = []
    spec_resistances       = []
    gait_sym_indices       = []
    r_swing_freqs          = []
    l_swing_freqs          = []
    mean_swing_durs        = []
    mean_stance_durs       = []
    double_sups            = []
    single_sups            = []
    flights                = []
    alternation_scores     = []
    phase_lag_steps_list   = []
    hip_knee_coords        = []
    inter_leg_anticorrs    = []
    hip_ranges             = []
    hip_ext_pushoffs       = []
    hip_flex_swings        = []
    knee_flex_swings       = []
    stance_straights       = []
    contact_per_metres     = []
    step_len_syms          = []

    for ep in episodes_data:
        obs     = ep["observations"]    # (T, 24)
        actions = ep["actions"]         # (T, 4)
        fell    = ep["fell"]

        T = len(obs)
        if T < 2:
            continue

        # ── Basic locomotion ─────────────────────────────────────────
        speeds_x     = obs[:, 2]
        dt           = 1.0 / 50.0
        forward_dist = float(np.sum(speeds_x) * dt * _SPEED_SCALE * 50)
        mean_spd     = float(np.mean(speeds_x) * _SPEED_SCALE)
        dist_safe    = max(abs(forward_dist), 1e-3)

        # ── Posture ───────────────────────────────────────────────────
        hull_angles  = obs[:, 0]
        mean_tilt    = float(np.mean(np.abs(hull_angles)))
        tilt_var     = float(np.var(hull_angles))
        lat_stab     = float(np.mean(np.abs(obs[:, 1])))

        # ── Energy ────────────────────────────────────────────────────
        energy       = float(np.sum(np.abs(actions)))
        act_diff     = np.diff(actions, axis=0)
        smoothness   = float(np.mean(np.abs(act_diff)))
        spec_res     = energy / (dist_safe * T) if (dist_safe > 0 and T > 0) else 999.0

        # ── Ground contact signals ────────────────────────────────────
        contact_r = obs[:, 8]    # right foot
        contact_l = obs[:, 13]   # left foot

        both_on  = (contact_r >= _CONTACT_THRESH) & (contact_l >= _CONTACT_THRESH)
        one_on   = (contact_r >= _CONTACT_THRESH) ^ (contact_l >= _CONTACT_THRESH)
        both_off = (contact_r < _CONTACT_THRESH)  & (contact_l < _CONTACT_THRESH)

        double_sup = float(np.mean(both_on))
        single_sup = float(np.mean(one_on))
        flight     = float(np.mean(both_off))

        # ── Gait cycle analysis ───────────────────────────────────────
        swing_r, stance_r = _detect_phases(contact_r)
        swing_l, stance_l = _detect_phases(contact_l)

        n_strides_r = len(swing_r)
        n_strides_l = len(swing_l)

        r_freq = (n_strides_r / T) * 100
        l_freq = (n_strides_l / T) * 100

        mean_strides = (n_strides_r + n_strides_l) / 2.0 + 1e-6
        gait_sym = abs(n_strides_r - n_strides_l) / mean_strides

        all_swings  = np.concatenate([swing_r, swing_l])  if len(swing_r) + len(swing_l) > 0 else np.array([0.0])
        all_stances = np.concatenate([stance_r, stance_l]) if len(stance_r) + len(stance_l) > 0 else np.array([0.0])
        mean_swing  = float(np.mean(all_swings))
        mean_stance = float(np.mean(all_stances))

        # ── Phase alternation score ───────────────────────────────────
        # A human gait has right-swing ↔ left-stance and vice versa.
        # Measure: when right foot is off ground AND left foot is on ground
        # (R-swing / L-stance), or left off AND right on (L-swing / R-stance).
        # Ideal alternation_score = 1.0 (all single-support is proper alternation).
        r_swing_mask = contact_r < _CONTACT_THRESH
        l_swing_mask = contact_l < _CONTACT_THRESH
        r_stance_mask = contact_r >= _CONTACT_THRESH
        l_stance_mask = contact_l >= _CONTACT_THRESH

        proper_alt = (r_swing_mask & l_stance_mask) | (l_swing_mask & r_stance_mask)
        any_single = r_swing_mask | l_swing_mask   # at least one foot off ground
        if any_single.sum() > 0:
            alt_score = float(proper_alt.sum() / any_single.sum())
        else:
            alt_score = 0.0

        # ── Phase lag (step offset between legs) ─────────────────────
        # Find onset times of each right-swing and left-swing phase.
        # The inter-leg phase lag should ideally be ~half the stride period.
        def _onset_indices(contact: np.ndarray):
            """Return timestep indices where foot lifts off (1→0 transition)."""
            mask = contact < _CONTACT_THRESH
            onsets = []
            for t in range(1, len(mask)):
                if mask[t] and not mask[t - 1]:
                    onsets.append(t)
            return onsets

        r_onsets = _onset_indices(contact_r)
        l_onsets = _onset_indices(contact_l)

        if r_onsets and l_onsets:
            # For each R onset find nearest L onset and compute lag
            lags = []
            for ro in r_onsets:
                nearest = min(l_onsets, key=lambda lo: abs(lo - ro))
                lags.append(abs(nearest - ro))
            phase_lag = float(np.mean(lags))
        else:
            phase_lag = 0.0

        # ── Hip angles ───────────────────────────────────────────────
        hip_r  = obs[:, 4]
        hip_l  = obs[:, 9]
        knee_r = obs[:, 6]
        knee_l = obs[:, 11]

        hip_r_range = float(np.max(hip_r) - np.min(hip_r))
        hip_l_range = float(np.max(hip_l) - np.min(hip_l))
        hip_range   = (hip_r_range + hip_l_range) / 2.0

        # ── Hip extension at push-off ─────────────────────────────────
        # Just before foot leaves ground (stance→swing transition) the hip
        # should be extended backward (negative angle = leg behind body).
        def _hip_at_liftoff(hip: np.ndarray, contact: np.ndarray):
            """Mean hip angle in the last 3 steps before each liftoff."""
            angles = []
            for t in range(3, len(contact)):
                if contact[t] < _CONTACT_THRESH and contact[t - 1] >= _CONTACT_THRESH:
                    angles.append(np.mean(hip[max(0, t - 3): t]))
            return float(np.mean(angles)) if angles else 0.0

        hip_ext_r = _hip_at_liftoff(hip_r, contact_r)
        hip_ext_l = _hip_at_liftoff(hip_l, contact_l)
        hip_ext_pushoff = (hip_ext_r + hip_ext_l) / 2.0

        # ── Hip / knee angles at mid-swing ───────────────────────────
        # During swing, the hip should flex forward (positive) and knee bend.
        def _angle_at_midswing(angle: np.ndarray, contact: np.ndarray):
            """Mean angle value at the midpoint of each swing phase."""
            vals = []
            in_swing = False
            swing_start = 0
            for t, c in enumerate(contact):
                if c < _CONTACT_THRESH:
                    if not in_swing:
                        in_swing = True
                        swing_start = t
                else:
                    if in_swing:
                        mid = (swing_start + t) // 2
                        vals.append(angle[mid])
                    in_swing = False
            return float(np.mean(vals)) if vals else 0.0

        hip_flex_r = _angle_at_midswing(hip_r, contact_r)
        hip_flex_l = _angle_at_midswing(hip_l, contact_l)
        hip_flex_swing = (hip_flex_r + hip_flex_l) / 2.0

        knee_flex_r = _angle_at_midswing(knee_r, contact_r)
        knee_flex_l = _angle_at_midswing(knee_l, contact_l)
        knee_flex_swing = (knee_flex_r + knee_flex_l) / 2.0

        # ── Hip–knee coordination ─────────────────────────────────────
        coord_r = _safe_corr(hip_r, knee_r)
        coord_l = _safe_corr(hip_l, knee_l)
        hip_knee_coord = float((abs(coord_r) + abs(coord_l)) / 2.0)

        # ── Inter-leg hip anticorrelation ─────────────────────────────
        # Right and left hips should move in OPPOSITE directions during walking
        # (when R hip swings forward, L hip should swing back).  Ideal = -1 corr.
        # We store as 1 - corr so higher = more human-like.
        inter_leg_corr   = _safe_corr(hip_r, hip_l)
        inter_leg_anticorr = float(1.0 - inter_leg_corr) / 2.0  # maps [-1,1]→[0,1]

        # ── Stance leg straightness ───────────────────────────────────
        stance_r_mask = contact_r >= _CONTACT_THRESH
        stance_l_mask = contact_l >= _CONTACT_THRESH

        knee_r_stance = knee_r[stance_r_mask] if stance_r_mask.any() else np.array([0.0])
        knee_l_stance = knee_l[stance_l_mask] if stance_l_mask.any() else np.array([0.0])
        stance_straight = float(
            (np.mean(np.abs(knee_r_stance)) + np.mean(np.abs(knee_l_stance))) / 2.0
        )

        # ── Contact per metre ─────────────────────────────────────────
        total_contacts = int(
            np.sum(contact_r >= _CONTACT_THRESH) + np.sum(contact_l >= _CONTACT_THRESH)
        )
        contact_per_m = total_contacts / dist_safe

        # ── Step length symmetry ──────────────────────────────────────
        # Estimate step length from swing duration × speed.  Symmetric = 0.
        def _mean_swing_len(swing_lens: np.ndarray, spd: float) -> float:
            if len(swing_lens) == 0:
                return 0.0
            return float(np.mean(swing_lens)) * spd * dt * _SPEED_SCALE

        step_r = _mean_swing_len(swing_r, float(np.mean(speeds_x)))
        step_l = _mean_swing_len(swing_l, float(np.mean(speeds_x)))
        step_mean = (step_r + step_l) / 2.0 + 1e-6
        step_len_sym = abs(step_r - step_l) / step_mean

        # ── Accumulate ───────────────────────────────────────────────
        forward_distances.append(forward_dist)
        mean_speeds.append(mean_spd)
        falls.append(int(fell))
        tilts.append(mean_tilt)
        tilt_vars.append(tilt_var)
        lateral_stabs.append(lat_stab)
        energies.append(energy)
        smoothnesses.append(smoothness)
        spec_resistances.append(spec_res)
        gait_sym_indices.append(gait_sym)
        r_swing_freqs.append(r_freq)
        l_swing_freqs.append(l_freq)
        mean_swing_durs.append(mean_swing)
        mean_stance_durs.append(mean_stance)
        double_sups.append(double_sup)
        single_sups.append(single_sup)
        flights.append(flight)
        alternation_scores.append(alt_score)
        phase_lag_steps_list.append(phase_lag)
        hip_knee_coords.append(hip_knee_coord)
        inter_leg_anticorrs.append(inter_leg_anticorr)
        hip_ranges.append(hip_range)
        hip_ext_pushoffs.append(hip_ext_pushoff)
        hip_flex_swings.append(hip_flex_swing)
        knee_flex_swings.append(knee_flex_swing)
        stance_straights.append(stance_straight)
        contact_per_metres.append(contact_per_m)
        step_len_syms.append(step_len_sym)

    if not tilts:   # all episodes were too short
        return _empty_metrics()

    def _m(lst): return float(np.mean(lst))

    metrics = {
        # ── Basic locomotion ────────────────────────────────────────
        "average_forward_distance":  _m(forward_distances),
        "average_speed":             _m(mean_speeds),
        "fall_rate":                 _m(falls),

        # ── Posture & stability ─────────────────────────────────────
        "torso_tilt":                _m(tilts),
        "torso_tilt_variance":       _m(tilt_vars),
        "lateral_stability":         _m(lateral_stabs),

        # ── Energy & smoothness ─────────────────────────────────────
        "energy_consumption":        _m(energies),
        "action_smoothness":         _m(smoothnesses),
        "specific_resistance":       _m(spec_resistances),

        # ── Gait cycle ──────────────────────────────────────────────
        "gait_symmetry_index":       _m(gait_sym_indices),
        "right_leg_swing_freq":      _m(r_swing_freqs),
        "left_leg_swing_freq":       _m(l_swing_freqs),
        "mean_swing_duration":       _m(mean_swing_durs),
        "mean_stance_duration":      _m(mean_stance_durs),
        "double_support_fraction":   _m(double_sups),
        "single_support_fraction":   _m(single_sups),
        "flight_fraction":           _m(flights),

        # ── Phase alternation (KEY human-gait metrics) ───────────────
        "alternation_score":         _m(alternation_scores),
        "phase_lag_steps":           _m(phase_lag_steps_list),

        # ── Hip & knee coordination ─────────────────────────────────
        "hip_knee_coordination":     _m(hip_knee_coords),
        "inter_leg_hip_anticorr":    _m(inter_leg_anticorrs),
        "hip_angle_range":           _m(hip_ranges),
        "hip_extension_at_pushoff":  _m(hip_ext_pushoffs),
        "hip_flexion_at_swing_peak": _m(hip_flex_swings),
        "knee_flexion_at_swing_peak":_m(knee_flex_swings),

        # ── Contact quality ─────────────────────────────────────────
        "stance_leg_straightness":   _m(stance_straights),
        "contact_per_metre":         _m(contact_per_metres),
        "step_length_symmetry":      _m(step_len_syms),
    }
    return metrics


def _empty_metrics() -> Dict[str, float]:
    """Return a zeroed metrics dict when no episodes are available."""
    return {
        "average_forward_distance":  0.0,
        "average_speed":             0.0,
        "fall_rate":                 1.0,
        "torso_tilt":                0.0,
        "torso_tilt_variance":       0.0,
        "lateral_stability":         0.0,
        "energy_consumption":        0.0,
        "action_smoothness":         0.0,
        "specific_resistance":       999.0,
        "gait_symmetry_index":       1.0,
        "right_leg_swing_freq":      0.0,
        "left_leg_swing_freq":       0.0,
        "mean_swing_duration":       0.0,
        "mean_stance_duration":      0.0,
        "double_support_fraction":   0.0,
        "single_support_fraction":   0.0,
        "flight_fraction":           0.0,
        "alternation_score":         0.0,
        "phase_lag_steps":           0.0,
        "hip_knee_coordination":     0.0,
        "inter_leg_hip_anticorr":    0.0,
        "hip_angle_range":           0.0,
        "hip_extension_at_pushoff":  0.0,
        "hip_flexion_at_swing_peak": 0.0,
        "knee_flexion_at_swing_peak":0.0,
        "stance_leg_straightness":   0.0,
        "contact_per_metre":         0.0,
        "step_length_symmetry":      1.0,
    }

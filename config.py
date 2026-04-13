# =============================================================================
# config.py — Central configuration for the LLM-guided RL experiment
# =============================================================================

import os

# --- Training ---
TRAINING_TIMESTEPS  = 500_000        # SAC timesteps per iteration
EVALUATION_EPISODES = 15             # episodes to evaluate per iteration
MAX_ITERATIONS      = 22        # total LLM reward-improvement cycles
N_ENVS              = 4              # parallel envs for rollout (SubprocVecEnv)
                                     # set to 1 to fall back to DummyVecEnv

# --- Simulation Rendering ---
# Set True  → pygame window opens after each training iteration
# Set False → fully headless (useful on servers without a display)
RENDER_DEMO = True

# Set True  → pygame window stays open DURING training so you can watch
# the agent actively learning. Warning: slows down training significantly.
RENDER_TRAINING = False

# --- Paths ---
REWARD_FILE      = "rewards/current_reward.py"  
REWARD_ARCHIVE   = "rewards/archive"
MODEL_DIR        = "models"
LOG_DIR          = "logs"
METRICS_LOG      = "logs/metrics_history.json"
HUMAN_HINT_FILE  = "logs/human_hint.txt"

# --- LLM ---
LLM_API_KEY  = os.environ.get("GEMINI_API_KEY", "AIzaSyDU9xnhcHo7KfUd5Qs_ocCAH0YoRhQ3D3I")  # loaded via env var
MODEL_NAME   = "gemini-2.5-flash"                     # Changed to flash because 3.1 Pro has a limit of 0 on the Free Tier

# --- Environment ---
ENV_ID = "BipedalWalker-v3"   # base registration name (used as fallback)

# =============================================================================
# Hard Environment — Procedural Terrain + External Disturbances
# =============================================================================

# Master switch: use HardBipedalEnv instead of vanilla BipedalWalker-v3
USE_HARD_ENV = True

# ── Terrain generation ────────────────────────────────────────────────────────
# Toggle chunk-based procedural infinite terrain
ENABLE_TERRAIN_GENERATION = True

# Initial difficulty: 0.0 = pure flat, 1.0 = hardest mix (stairs + slopes)
# The curriculum unlocks new terrain types as this value rises:
#   0.0 – 0.3  →  FLAT only
#   0.3 – 0.55 →  FLAT + UNEVEN
#   0.55– 0.75 →  FLAT + UNEVEN + SLOPE
#   0.75– 1.0  →  all types including STAIRS
TERRAIN_DIFFICULTY_LEVEL = 0.0

# Width of each terrain chunk in metres (Testing cycles through them)
CHUNK_LENGTH = 15.0

# Peak-to-peak amplitude of height variation for UNEVEN terrain (m)
MAX_HEIGHT_VARIATION = 0.35

# (min°, max°) slope for SLOPE terrain (negative = downhill)
SLOPE_RANGE = (-12.0, 12.0)

# Riser height per step for STAIRS terrain (m)
STEP_HEIGHT = 0.40

# Randomly pick terrain type each chunk (True) vs. cycle through them sequentially (False)
RANDOMISE_TERRAIN = False

# ── External disturbances ─────────────────────────────────────────────────────
# Toggle random lateral force pushes applied to the hull
ENABLE_DISTURBANCE = False

# Lateral force range (N) per push — direction is randomised each time
DISTURBANCE_FORCE_RANGE = (50.0, 200.0)

# Probability [0, 1] that a disturbance push occurs on any given step
DISTURBANCE_FREQUENCY = 0.03

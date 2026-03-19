# =============================================================================
# config.py — Central configuration for the LLM-guided RL experiment
# =============================================================================

# --- Training ---
TRAINING_TIMESTEPS  = 400_000      # PPO timesteps per iteration  (2M total across 10 iters)
EVALUATION_EPISODES = 10           # episodes to evaluate the policy per iteration
MAX_ITERATIONS      = 10           # total LLM reward-improvement cycles

# --- Simulation Rendering ---
# Set to True  → a pygame window opens after each training iteration so you
#                can watch the walker improve over time.
# Set to False → fully headless (useful on servers without a display).
RENDER_DEMO = True

# --- Paths ---
REWARD_FILE      = "rewards/current_reward.py"
REWARD_ARCHIVE   = "rewards/archive"
MODEL_DIR        = "models"
LOG_DIR          = "logs"
METRICS_LOG      = "logs/metrics_history.json"
HUMAN_HINT_FILE  = "logs/human_hint.txt"

import os

# --- LLM ---
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "")   # Loads safely via env var OPENAI_API_KEY
MODEL_NAME   = "gpt-4o"                      # or "gpt-3.5-turbo"

# --- Environment ---
ENV_ID = "BipedalWalker-v3"

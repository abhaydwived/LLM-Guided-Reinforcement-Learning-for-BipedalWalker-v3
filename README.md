<div align="center">
  <h1>🤖 LLM-Guided Reinforcement Learning for BipedalWalker-v3</h1>
  <img src="simulation.gif?v=2" alt="BipedalWalker Simulation Demo" width="600" style="border-radius: 8px; margin: 15px 0;"/>
  <p><i>An automated testbed utilizing GPT-4o to dynamically optimize reward functions and train robust PPO policies.</i></p>
  
  <p>
    <img src="https://img.shields.io/badge/Training_Iterations-10-blue?style=for-the-badge" alt="Iterations" />
    <img src="https://img.shields.io/badge/Timesteps_per_Iteration-400k-green?style=for-the-badge" alt="Timesteps Per Iteration" />
    <img src="https://img.shields.io/badge/Total_Timesteps-4M-orange?style=for-the-badge" alt="Total Timesteps" />
  </p>
</div>

Welcome to the **Automated LLM-Guided Reinforcement Learning Testbed**. This project leverages the modern `BipedalWalker-v3` environment from Gymnasium to orchestrate a continuous cycle of agent training and intelligent reward shaping. By combining Stable Baselines3's PPO algorithm with the reasoning capabilities of Large Language Models (LLMs), the framework autonomously engineers and iteratively refines its own reward functions based on extracted locomotion metrics.

## Features

- **Automated Reward Engineering**: An LLM agent iteratively refines the reward function based on evaluation metrics.
- **PPO Training Loop**: Uses Stable Baselines 3 for robust RL training (`PPO`).
- **Comprehensive Evaluation**: Extracts meaningful locomotion metrics (e.g., average forward distance, fall rate, torso tilt, energy consumption).
- **Visualization**: Includes a pygame-based viewer to watch the trained policies in action along with support for rendering episodes during the training loop.
- **Metrics Tracking**: Maintains a history of evaluation metrics to track improvement over iterations.

## Directory Structure

```text
llm_rl_bipedal/
├── config.py                  # Central configuration for hyper-parameters, API keys, etc.
├── main_loop.py               # Main execution script; coordinates the iterative training and LLM guiding process.
├── watch_policy.py            # Utility script to visualize a trained model in the environment.
├── requirements.txt           # Python dependencies.
├── README.md                  # Project documentation.
├── env/                       # Environment configuration and testing
│   ├── bipedal_env.py         
│   └── test_env.py            
├── evaluation/                # Evaluation & metric computations
│   └── metrics.py             # Computes locomotion metrics from episode rollout data.
├── llm/                       # LLM integration for reward generation
│   ├── prompt_builder.py      # Constructs the prompt for the LLM using evaluation metrics.
│   └── reward_generator.py    # Interfaces with OpenAI API to generate and save the new reward function.
├── logs/                      # Log files and metrics history (`metrics_history.json`).
├── models/                    # Saved PPO models (`.zip` format).
├── rewards/                   # The active and historically generated reward functions.
│   ├── current_reward.py      # The currently active reward function.
│   └── archive/               # Archive of previously generated reward functions.
└── rl/                        # RL training and policy rollout
    ├── reward_wrapper.py      # Gymnasium wrapper to inject the custom reward function.
    ├── test_policy.py         # Evaluates a trained policy and collects rollout data.
    └── train.py               # PPO training loop.
```

## Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed. This project uses SWIG for Box2D physics, so you may need to install the SWIG system dependency first before installing `box2d-py`:
- **Windows**: Download `swig.exe` and add it to your PATH, or install via `conda install swig`.
- **Ubuntu/Debian**: `sudo apt-get install swig`
- **macOS**: `brew install swig`

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup LLM API Key
In `config.py`, replace the `LLM_API_KEY` with your OpenAI API key, or set it as an environment variable `OPENAI_API_KEY`.

## Usage

### 1. Running the Main Training Loop
To start the automated reward engineering pipeline, run the `main_loop.py` script. The script performs the following steps `MAX_ITERATIONS` times:
1. Trains the PPO agent with the current reward function.
2. Evaluates the locomotion performance to extract metrics.
3. Builds an LLM prompt containing the metrics.
4. Calls the LLM to generate an improved reward function.
5. Saves metrics to `logs/metrics_history.json`.

```bash
python main_loop.py
```

### 2. Watching a Trained Policy
You can visualize any of the models generated during the training phases using `watch_policy.py`.

```bash
# Watch the latest saved model
python watch_policy.py

# Watch a specific iteration
python watch_policy.py --iter 3

# Watch a specific model directly
python watch_policy.py --model models/ppo_bipedal_iter_3.zip

# Save the simulation rollout as an animated GIF
python watch_policy.py --gif simulation.gif

# Change the number of episodes shown
python watch_policy.py --episodes 5
```

## Configuration

The project is highly configurable via `config.py`. Here are the most important settings:

- **Training**: Adjust `TRAINING_TIMESTEPS` (PPO timesteps per iteration), `EVALUATION_EPISODES`, and `MAX_ITERATIONS` to balance compute time and learning capability.
- **Rendering**: Set `RENDER_DEMO = True` to briefly show the walker's progress in a window at the end of each iteration, or `False` for headless servers.
- **LLM Settings**: Switch between `gpt-4o` and `gpt-3.5-turbo` via `MODEL_NAME`.

## How the Reward is Updated
1. The RL agent relies on the reward computed by `rewards/current_reward.py`.
2. After a training phase, `evaluation/metrics.py` parses rollout trajectories.
3. `llm/prompt_builder.py` feeds these metrics into an engineered prompt.
4. `llm/reward_generator.py` parses the LLM response code and safely overwrites `rewards/current_reward.py`, effectively archiving the old reward in `rewards/archive/`.

Enjoy optimizing your Bipedal Walker!

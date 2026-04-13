<div align="center">
  <h1>LLM-guided reinforcement learning for BipedalWalker-v3</h1>

  <p>
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python Version" />
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
    <img src="https://img.shields.io/badge/Build-passing-brightgreen?style=for-the-badge&logo=githubactions" alt="CI Status" />
  </p>
  
  <p><i>An automated testbed utilizing Gemini 1.5 Pro to dynamically optimize reward functions and train robust policies.</i></p>
  
  <br/>
  
  <h3>Version 2: Advanced gait and custom terrains</h3>
  <img src="demof.gif" alt="Bipedal walker agent demonstrating version 2 refined gait, 15cm step size, and alternating leg swing on procedurally generated custom stair terrain" width="600" style="border-radius: 8px; margin: 15px 0;"/>
  <p>
    <img src="https://img.shields.io/badge/Training_Iterations-20-blue?style=for-the-badge" alt="Version 2 Iterations" />
    <img src="https://img.shields.io/badge/Timesteps_per_Iteration-500k-green?style=for-the-badge" alt="Version 2 Timesteps Per Iteration" />
    <img src="https://img.shields.io/badge/Total_Timesteps-10M-orange?style=for-the-badge" alt="Version 2 Total Timesteps" />
  </p>

  <h3>Version 1: Core framework</h3>
  <img src="simulation.gif?v=2" alt="Bipedal walker agent demonstrating version 1 basic forward locomotion on flat terrain at early iteration" width="400" style="border-radius: 8px; margin: 15px 0;"/>
  <p>
    <img src="https://img.shields.io/badge/Training_Iterations-10-blue?style=for-the-badge" alt="Version 1 Iterations" />
    <img src="https://img.shields.io/badge/Timesteps_per_Iteration-400k-green?style=for-the-badge" alt="Version 1 Timesteps Per Iteration" />
    <img src="https://img.shields.io/badge/Total_Timesteps-4M-orange?style=for-the-badge" alt="Version 1 Total Timesteps" />
  </p>
</div>

This project provides an automated testbed leveraging the `BipedalWalker-v3` environment from Gymnasium to orchestrate a continuous cycle of agent training and intelligent reward shaping. By integrating Stable Baselines3's PPO algorithm with the reasoning capabilities of Large Language Models, the framework autonomously engineers and iteratively refines reward functions based on extracted locomotion metrics.

## Version 2 features (latest)

Version 2 introduces physical controls, custom terrain handling, and a continuous training pipeline.

- **Continuous training pipeline**: Automated training resume system without manual restarts to preserve learning progress.
- **Advanced locomotion metrics**: 21 gait-specific evaluation metrics for comprehensive tracking.
- **Context-aware LLM prompting**: Prompt builder provides the LLM with trend-aware context for iterative reward refinement.
- **Custom terrain generation and evaluation**: Agent actively trained and evaluated across 4 diverse continuous terrain types: flat surfaces, uneven ground, slopes, and stairs (both ascending and descending), with evaluation aligned precisely to the respective custom geometry.
- **Refined bipedal gait**: Integration of a human hint file to encourage natural movement, including alternating leg swing, 15cm step size, and increased walking speed.
- **Improved visualization**: Vertical camera tracking system properly follows the agent across varying elevations and eliminates visual rendering artifacts.

## Version 1 features (core framework)

The original testbed demonstrating LLM-driven reward shaping on flat terrain.

- **Automated reward engineering**: An LLM agent iteratively refines the reward function based on evaluation metrics.
- **PPO/SAC training loop**: Uses Stable Baselines3 for robust RL training.
- **Basic evaluation**: Fundamental locomotion metrics including average forward distance, fall rate, torso tilt, and energy consumption.
- **Visualization**: Pygame-based viewer to watch trained policies with support for rendering episodes during the training loop.
- **Metrics tracking**: Preserves history of evaluation metrics to track improvement over iterations.
- **Peak performance**: The best performing model and optimized reward function were successfully discovered at Iteration 9 (out of 10).

## Results and benchmarks

| Metric | LLM-guided PPO (Iter 17) | Vanilla PPO baseline |
| :--- | :--- | :--- |
| **Max forward distance** | 151.10 m (Completed)| -0.76 m (Failed) |
| **Energy consumption** | 3645.95 | 963.51 |
| **Gait symmetry index** | 0.023 (Highly Symmetrical) | 2.000 (Asymmetrical) |
| **Torso tilt (rad)** | 0.086 | N/A (Fell) |
| **Convergence iteration** | Iteration 17 | Iteration 9 |

**Reward curve description**: By iteration 17, the model successfully navigated the full 151m diverse terrain continuously across stairs and slopes, achieving a highly symmetrical gait (0.023) and maintaining stable posture (torso tilt: 0.086 rad). In contrast, the Vanilla PPO baseline failed to converge entirely.

## Directory structure

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
│   └── reward_generator.py    # Interfaces with the LLM API to generate and save the new reward function.
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

### Prerequisites

Ensure Python 3.8+ is installed. This project uses SWIG for Box2D physics.

- **Windows**: Download `swig.exe` and add it to the system PATH.
- **Ubuntu/Debian**: `sudo apt-get install swig`
- **macOS**: `brew install swig`

### Environment setup

Create and activate a virtual environment before installing dependencies.

#### Using venv
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

#### Using conda (optional)
```bash
conda create -n bipedal_llm python=3.10
conda activate bipedal_llm

pip install -r requirements.txt
```

### Troubleshooting

- **SWIG not found**: Verify that the SWIG executable is in the system PATH. On Windows, executing `swig -version` in a new terminal should return the version string.
- **Box2D build error**: Occurs if Microsoft Visual C++ Build Tools (Windows) or `build-essential` (Linux) are missing. Install the required C++ compiler for the operating system.
- **Pygame display error**: When running in headless environments (like SSH or WSL), pygame may crash attempting to access a display. Set the environment variable `SDL_VIDEODRIVER=dummy` to bypass this constraint during background training.

## Configuration reference

Modify the training and evaluation parameters in `config.py`.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `TRAINING_TIMESTEPS` | `int` | `400000` | Number of PPO timesteps per RL iteration. |
| `EVALUATION_EPISODES` | `int` | `10` | Number of episodes to run for performance metrics extraction. |
| `MAX_ITERATIONS` | `int` | `10` | Total number of LLM reward generation loops to perform. |
| `RENDER_DEMO` | `bool` | `False` | Whether to render the agent's progress graphically after each iteration. |
| `MODEL_NAME` | `str` | `"gemini-1.5-pro"` | Identifier for the LLM used for reward shaping. |
| `LLM_API_KEY` | `str` | `""` | API key for the corresponding LLM provider. |
| `ENVIRONMENT_NAME` | `str` | `"BipedalWalker-v3"` | Target Gymnasium environment instance. |

## Usage

### LLM API setup

In `config.py`, replace the `LLM_API_KEY` string with a valid API key, or export it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Executing the pipeline

Run the `main_loop.py` script to initiate the automated reward engineering pipeline. The script sequentially executes PPO training, metric evaluation, prompt building, LLM querying, and reward archiving for the configured number of iterations.

```bash
python main_loop.py
```

### Visualizing models

Evaluate or visualize saved models using `watch_policy.py`.

```bash
# Watch the latest saved model
python watch_policy.py

# Watch a specific iteration
python watch_policy.py --iter 3

# Watch a specific model directly
python watch_policy.py --model models/ppo_bipedal_iter_3.zip

# Save the simulation rollout as an animated GIF
python watch_policy.py --gif simulation.gif

# Define the number of episodes to visualize
python watch_policy.py --episodes 5
```

## Hardware requirements

| Component | Minimum specification | Recommended specification |
| :--- | :--- | :--- |
| CPU | 4 cores, 2.5 GHz | 8+ cores, 3.0 GHz+ |
| GPU | Not strictly required | NVIDIA GPU with CUDA support |
| RAM | 8 GB | 16 GB |

**Expected wall-clock time**: Running the full pipeline (20 iterations, 10M total timesteps) requires approximately 8 hours on the minimum specifications and 3.5 hours on the recommended specifications.

## Known issues and limitations

- **Headless rendering**: Rendering operations will fail on systems without a configured display server. Disable `RENDER_DEMO` in `config.py` when executing over SSH or within WSL environments.
- **LLM API rate limits**: High iteration frequencies or extensive prompt contexts can exceed standard API rate limits. Ensure adequate quota limits for the provider account.
- **Estimated costs**: Processing a full pipeline execution generates approximately 1.5M input tokens and 250,000 output tokens, leading to an estimated cost of ~$0.80 per full run depending on current Gemini 1.5 Pro API pricing.
- **GPU vs CPU scaling**: While policy inference executes adequately on CPU, RL training via Stable Baselines3 scales significantly better with GPU acceleration. Expect a 4x to 6x reduction in required training time when utilizing a dedicated GPU.

## Security notice

This framework inherently executes arbitrary Python code generated by an LLM via the `reward_generator.py` script. 

To mitigate risks, the system implements a constrained execution scope for dynamically generated reward code. The LLM output is parsed to extract only the necessary `reward_function` signature before it is persisted to disk. A rollback mechanism archives the exact state of the previous functional reward logic in the `rewards/archive/` directory. If a newly generated reward function contains syntax errors or fails to execute during the PPO rollout, the training loop automatically reverts to the most recent stable checkpoint and requests a corrected implementation from the LLM. 

Do not run this framework in production environments holding accessible sensitive data without utilizing proper sandbox isolation (e.g., Docker containers).

## Reward update mechanism

1. The RL agent queries the active reward computed by `rewards/current_reward.py`.
2. Following training, `evaluation/metrics.py` parses the rollout trajectories to assemble performance statistics.
3. `llm/prompt_builder.py` normalizes these metrics and integrates them into the evaluation prompt.
4. `llm/reward_generator.py` parses the valid Python code from the LLM response, replaces the logic in `rewards/current_reward.py`, and logs the superseded code to `rewards/archive/`.

## Contributing

Contributions to improve the pipeline, add metric parameters, or port to other environments are welcome.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/improvement-name`).
3. Commit the changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/improvement-name`).
5. Open a Pull Request.

Ensure that any new functionality aligns with the existing code style and includes updated documentation if applicable.

## Acknowledgements

- **[Gymnasium](https://gymnasium.farama.org/)**: For maintaining the rigorous `BipedalWalker-v3` continuous control environment.
- **[Stable Baselines3](https://stable-baselines3.readthedocs.io/)**: For providing robust, production-ready reinforcement learning algorithms.
- **Google DeepMind**: For providing the Gemini 2.5 flash base models powering the automated reward reasoning.

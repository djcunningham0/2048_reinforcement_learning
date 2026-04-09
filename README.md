# 2048 reinforcement learning

This repo implements several reinforcement learning techniques to play the game [2048](https://play2048.co).

Check out my blog post for more details:
https://dannycunningham.com/posts/2026-02-12-just-show-me-the-recipe/

## Methodologies

Several temporal difference (TD) learning methodologies are implemented:

- Deep Q-Network (DQN) with a convolutional neural network (CNN)
- TD-afterstate learning with CNN
- TD-afterstate learning with N-tuple network

Expectimax search is also implemented to improve agent performance at runtime.

## Repository structure

- `rl_2048/`: code for implementing the 2048 game environment and training RL agents
    - `afterstate/`: implementation of the TD-afterstate CNN agent
    - `dqn/`: implementation of the DQN agent
    - `ntuple/`: implementation of the TD-afterstate N-tuple network
    - `expectimax.py`: expectimax search
    - `game.py`: 2048 game engine
    - `inference.py`: utilities for using trained agents for inference (playing a game)
    - `network.py`: definition of the CNN used for the DQN and TD-afterstate agent
    - `profiler.py`: timing utilities
- `rust_2048/`: implementation of the game engine and N-tuple network in Rust for faster training and inference (Claude Code adapted the Python code; see module-level README for details)
- `rust_2048_py/`: PyO3 bindings that expose `rust_2048` to Python (built with maturin by Claude Code)
- `scripts/`:
    - `train_*.py`: scripts for training each agent, including TensorBoard logging (logging is a bit messy and could be improved)
    - `evaluate_search.py`: evaluate the performance of a given agent
    - `play.py`: play 2048 in the terminal (validate that the game engine works)
    - `watch_agent.py`: watch a trained agent play 2048 in the terminal
    - `play_web.py`: watch a trained agent play "Weddy-48", a custom 2048 implementation (URL is omitted from GitHub for privacy, so this script will not run)
    - `play_web_rust.py`: same as `play_web.py`, but only supports N-tuple networks from the Rust implementation
- `tests/`: unit tests for Python modules

For instructions on running any of the scripts in `scripts/`, see the docstring at the top of each script or run `python -m scripts.<SCRIPT_NAME> --help` (omitting `.py` from the script name).

## Installing

The repo is a Python package.
Install the package locally using pip:

```
pip install -e .
```

For development, install the dev extras and run the tests with:

```
pip install -e '.[dev]'
python -m pytest
```

To also build and install the Rust Python bindings (required for `play_web_rust.py` and any Rust-backed N-tuple workflows), install [maturin](https://www.maturin.rs/) and run:

```
pip install maturin
maturin develop --release --manifest-path rust_2048_py/Cargo.toml
```

## Quickstart

Train a small N-tuple agent and then watch it play in the terminal:

```
python -m scripts.train_ntuple --max-episodes 100000
python -m scripts.watch_agent <path/to/checkpoint.npz> --model-type ntuple
```

For faster training, use the Rust implementation — see [rust_2048/README.md](rust_2048/README.md).

## Results

Results for all models at various depths of expectimax search.
Data points represent the results of 50 games played with each model.
Each model was trained for ~24 hours (100K games for the DRL methods, 10 million games for the N-tuple network).
I stopped at 1-ply expectimax for the deep reinforcement learning models because the runtime is extremely slow at greater depths.

#### Without Expectimax

| Model                 | Average Score | Max Score | Max Tile | 2048% | 4096% | 8192% | 16384% |
| --------------------- | ------------- | --------- | -------- | ----- | ----- | ----- | ------ |
| Deep Q-learning (DQN) | 12,000        | 31,000    | 2048     | 8%    | 0%    | 0%    | 0%     |
| Afterstate CNN        | 19,000        | 65,000    | 4096     | 42%   | 6%    | 0%    | 0%     |
| N-tuple network       | 194,000       | 361,000   | 16384    | 100%  | 100%  | 88%   | 38%    |

#### Expectimax depth = 1

| Model           | Average Score | Max Score | Max Tile | 2048% | 4096% | 8192% | 16384% |
| --------------- | ------------- | --------- | -------- | ----- | ----- | ----- | ------ |
| DQN             | 18,000        | 34,000    | 2048     | 36%   | 0%    | 0%    | 0%     |
| Afterstate CNN  | 53,000        | 147,000   | 8192     | 92%   | 58%   | 2%    | 0%     |
| N-tuple network | 273,000       | 372,000   | 16384    | 100%  | 100%  | 100%  | 70%    |

#### Expectimax depth = 2

| Model           | Average Score | Max Score | Max Tile | 2048% | 4096% | 8192% | 16384% |
| --------------- | ------------- | --------- | -------- | ----- | ----- | ----- | ------ |
| N-tuple network | 327,000       | 385,000   | 16384    | 100%  | 100%  | 100%  | 92%    |

#### Expectimax depth = 3

| Model           | Average Score | Max Score | Max Tile | 2048% | 4096% | 8192% | 16384% |
| --------------- | ------------- | --------- | -------- | ----- | ----- | ----- | ------ |
| N-tuple network | 325,000       | 387,000   | 16384    | 100%  | 100%  | 100%  | 100%   |

## License

MIT — see [LICENSE](LICENSE).

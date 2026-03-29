"""Afterstate TD learning hyperparameter configuration."""

from dataclasses import dataclass

from rl_2048.device import default_device


@dataclass
class AfterstateConfig:
    """
    Hyperparameters for TD-afterstate learning.

    - `buffer_capacity`: max number of transitions to store in replay buffer
    - `batch_size`: number of transitions to sample for each training step
    - `gamma`: discount factor for future rewards
    - `lr`: learning rate
    - `grad_clip_norm`: max norm for gradient clipping
    - `target_sync_interval`: how many steps between syncing target network
    - `train_start`: minimum replay buffer size before starting training (fill buffer
      with random actions until then)
    - `max_episodes`: maximum number of training episodes
    - `eval_interval`: how many episodes between evaluations
    - `eval_episodes`: how many episodes to run for each evaluation
    - `train_freq`: train every N environment steps (higher = faster but less sample
      efficient)
    - `restart`: whether to use the "restart" strategy (start episodes from the midpoint
      of the previous episode's play record)
    - `restart_min_length`: minimum episode length to trigger a restart (if shorter,
      start fresh)
    - `device`: "cpu", "cuda", or "mps" for training (auto-detected by default)
    """

    buffer_capacity: int = 100_000
    batch_size: int = 128
    train_freq: int = 4
    gamma: float = 0.99
    lr: float = 1e-4
    grad_clip_norm: float = 10.0
    target_sync_interval: int = 5_000
    train_start: int = 10_000
    max_episodes: int = 100_000
    eval_interval: int = 500
    eval_episodes: int = 25
    restart: bool = False
    restart_min_length: int = 10
    device: str = default_device()


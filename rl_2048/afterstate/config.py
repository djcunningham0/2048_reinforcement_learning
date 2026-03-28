"""Afterstate TD learning hyperparameter configuration."""

from dataclasses import dataclass

import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class AfterstateConfig:
    """
    Hyperparameters for TD-afterstate learning.

    Same structure as DQNConfig. Separate class allows the two approaches
    to diverge over time.
    """

    buffer_capacity: int = 100_000
    batch_size: int = 128
    train_freq: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.001
    epsilon_decay_steps: int = 100_000
    gamma: float = 0.99
    lr: float = 1e-4
    grad_clip_norm: float = 10.0
    target_sync_interval: int = 5_000
    train_start: int = 10_000
    max_episodes: int = 100_000
    eval_interval: int = 500
    eval_episodes: int = 50
    device: str = _default_device()

    def epsilon_at(self, step: int) -> float:
        """Linear epsilon decay."""
        frac = min(step / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

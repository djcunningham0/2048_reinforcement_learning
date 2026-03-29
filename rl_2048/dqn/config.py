"""DQN hyperparameter configuration."""

from dataclasses import dataclass

from rl_2048.device import default_device


@dataclass
class DQNConfig:
    """
    All DQN hyperparameters with sensible defaults.

    - `buffer_capacity`: max number of transitions to store in replay buffer
    - `batch_size`: number of transitions to sample for each training step
    - `epsilon_start`: initial epsilon for epsilon-greedy action selection
    - `epsilon_end`: final epsilon after decay
    - `epsilon_decay_steps`: number of steps over which to linearly decay epsilon
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
    - `device`: "cpu", "cuda", or "mps" for training (auto-detected by default)
    """

    buffer_capacity: int = 100_000
    batch_size: int = 128
    train_freq: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000
    gamma: float = 0.99
    lr: float = 1e-4
    grad_clip_norm: float = 10.0
    target_sync_interval: int = 5_000
    train_start: int = 10_000
    max_episodes: int = 100_000
    eval_interval: int = 500
    eval_episodes: int = 25
    device: str = default_device()

    def epsilon_at(self, step: int) -> float:
        """Linear epsilon decay (so random actions are less likely over time)."""
        frac = min(step / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

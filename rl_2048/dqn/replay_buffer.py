"""Replay buffer for DQN training."""

import random
from collections import deque
from dataclasses import dataclass

import torch

from rl_2048.game import Action


@dataclass(slots=True)
class Transition:
    state: torch.Tensor  # (16, 4, 4)
    action: Action
    reward: float
    next_state: torch.Tensor  # (16, 4, 4)
    done: bool
    valid_mask: torch.Tensor  # (4,) boolean mask of valid actions in next_state


@dataclass(slots=True)
class BatchedTransitions:
    states: torch.Tensor  # (batch, 16, 4, 4)
    actions: torch.Tensor  # (batch,) long
    rewards: torch.Tensor  # (batch,) float32
    next_states: torch.Tensor  # (batch, 16, 4, 4)
    dones: torch.Tensor  # (batch,) float32
    valid_masks: torch.Tensor  # (batch, 4) bool


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer."""

    def __init__(self, capacity: int):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> BatchedTransitions:
        transitions = random.sample(self._buffer, batch_size)
        return BatchedTransitions(
            states=torch.stack([t.state for t in transitions]),
            actions=torch.tensor([t.action for t in transitions], dtype=torch.long),
            rewards=torch.tensor([t.reward for t in transitions], dtype=torch.float32),
            next_states=torch.stack([t.next_state for t in transitions]),
            dones=torch.tensor([t.done for t in transitions], dtype=torch.float32),
            valid_masks=torch.stack([t.valid_mask for t in transitions]),
        )

    def __len__(self) -> int:
        return len(self._buffer)

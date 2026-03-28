"""
Replay buffer for TD-afterstate learning.

Afterstate encodings and next-step afterstates are pre-computed at push time.
"""

import random
from collections import deque
from dataclasses import dataclass

import torch

from rl_2048.game import Action, Board, apply_action, encode_state


@dataclass(slots=True)
class AfterstateTransition:
    afterstate: torch.Tensor  # (16, 4, 4)
    next_afterstates: torch.Tensor  # (4, 16, 4, 4)
    next_rewards: torch.Tensor  # (4,)
    next_valid_mask: torch.Tensor  # (4,) bool
    done: bool


@dataclass(slots=True)
class BatchedAfterstateTransitions:
    afterstates: torch.Tensor  # (batch, 16, 4, 4)
    next_afterstates: torch.Tensor  # (batch, 4, 16, 4, 4)
    next_rewards: torch.Tensor  # (batch, 4) merge reward per action from next_state
    next_valid_masks: torch.Tensor  # (batch, 4) bool
    dones: torch.Tensor  # (batch,) float32


def make_transition(
    afterstate: Board,
    next_state: Board,
    done: bool,
) -> AfterstateTransition:
    """Build a pre-computed transition from raw boards."""
    next_afterstates, next_rewards, next_mask = _compute_next_afterstates(next_state)
    return AfterstateTransition(
        afterstate=encode_state(afterstate),
        next_afterstates=next_afterstates,
        next_rewards=next_rewards,
        next_valid_mask=next_mask,
        done=done,
    )


class AfterstateReplayBuffer:
    """Fixed-capacity FIFO replay buffer for afterstate transitions."""

    def __init__(self, capacity: int):
        self._buffer: deque[AfterstateTransition] = deque(maxlen=capacity)

    def push(self, transition: AfterstateTransition):
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> BatchedAfterstateTransitions:
        transitions = random.sample(self._buffer, batch_size)
        return BatchedAfterstateTransitions(
            afterstates=torch.stack([t.afterstate for t in transitions]),
            next_afterstates=torch.stack([t.next_afterstates for t in transitions]),
            next_rewards=torch.stack([t.next_rewards for t in transitions]),
            next_valid_masks=torch.stack([t.next_valid_mask for t in transitions]),
            dones=torch.tensor([t.done for t in transitions], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


def _compute_next_afterstates(
    board: Board,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute afterstates, rewards, and valid mask for all 4 actions from a board.

    Returns (afterstates, rewards, valid_mask):
        afterstates: (4, 16, 4, 4)
        rewards: (4,)
        valid_mask: (4,) bool
    """
    afterstates = torch.zeros(4, 16, 4, 4)
    rewards = torch.zeros(4)
    valid_mask = torch.zeros(4, dtype=torch.bool)

    for action in Action:
        new_board, reward = apply_action(board, action)
        if new_board != board:
            afterstates[action] = encode_state(new_board)
            rewards[action] = reward
            valid_mask[action] = True

    return afterstates, rewards, valid_mask

"""
Replay buffer for TD-afterstate learning.

Afterstate encodings and next-step afterstates are computed on the fly during batch
sampling.
"""

import random
from collections import deque
from dataclasses import dataclass

import torch

from rl_2048.game import Action, Board, apply_action, encode_state


@dataclass(slots=True)
class AfterstateTransition:
    afterstate: Board  # board after slide+merge, before tile spawn
    next_state: Board  # board after tile spawn
    done: bool


@dataclass(slots=True)
class BatchedAfterstateTransitions:
    afterstates: torch.Tensor  # (batch, 16, 4, 4)
    next_afterstates: torch.Tensor  # (batch, 4, 16, 4, 4)
    next_rewards: torch.Tensor  # (batch, 4) merge reward per action from next_state
    next_valid_masks: torch.Tensor  # (batch, 4) bool
    dones: torch.Tensor  # (batch,) float32


class AfterstateReplayBuffer:
    """Fixed-capacity FIFO replay buffer for afterstate transitions."""

    def __init__(self, capacity: int):
        self._buffer: deque[AfterstateTransition] = deque(maxlen=capacity)

    def push(self, transition: AfterstateTransition):
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> BatchedAfterstateTransitions:
        transitions = random.sample(self._buffer, batch_size)

        afterstates = torch.stack([encode_state(t.afterstate) for t in transitions])

        next_data = [_compute_next_afterstates(t.next_state) for t in transitions]

        return BatchedAfterstateTransitions(
            afterstates=afterstates,
            next_afterstates=torch.stack([d[0] for d in next_data]),
            next_rewards=torch.stack([d[1] for d in next_data]),
            next_valid_masks=torch.stack([d[2] for d in next_data]),
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

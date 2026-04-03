"""
Replay buffer for TD-afterstate learning.

Afterstate encodings and next-step afterstates are pre-computed at push time.
"""

import random
from collections import deque
from dataclasses import dataclass

import torch

from rl_2048.game import Action, Board, apply_action, encode_state_into


@dataclass(slots=True)
class AfterstateInfo:
    """Pre-computed afterstates for all 4 actions from a board state.

    Shared between action selection (forward pass) and transition building
    (stored as next_afterstates), so each board's afterstates are computed once.
    """

    boards: list[Board]  # raw afterstate board per action (original board if invalid)
    encoded: torch.Tensor  # (4, 16, 4, 4)
    rewards: torch.Tensor  # (4,)
    valid_mask: torch.Tensor  # (4,) bool


@dataclass(slots=True)
class AfterstateTransition:
    """
    One afterstate transition.

    afterstate: encoded afterstate of the action taken
    next_afterstates: encoded afterstates for all 4 actions from the next board
    next_rewards: rewards for all 4 actions from the next board
    next_valid_mask: valid mask for all 4 actions from the next board
    done: whether the next board is terminal (no valid actions)
    """

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


class AfterstateComputer:
    """Pre-allocates scratch tensors so ``compute_all_afterstates`` creates no temporaries."""

    def __init__(self):
        self._encoded = torch.zeros(4, 16, 4, 4)
        self._rewards = torch.zeros(4)
        self._valid_mask = torch.zeros(4, dtype=torch.bool)

    def __call__(self, board: Board) -> AfterstateInfo:
        encoded = self._encoded
        rewards = self._rewards
        valid_mask = self._valid_mask

        encoded.zero_()
        rewards.zero_()
        valid_mask.zero_()

        boards: list[Board] = []
        for action in Action:
            new_board, reward = apply_action(board, action)
            boards.append(new_board)
            if new_board != board:
                encode_state_into(new_board, encoded[action])
                rewards[action] = reward
                valid_mask[action] = True

        return AfterstateInfo(
            boards=boards,
            encoded=encoded.clone(),
            rewards=rewards.clone(),
            valid_mask=valid_mask.clone(),
        )


_default_computer = AfterstateComputer()


def compute_all_afterstates(board: Board) -> AfterstateInfo:
    """Compute afterstates, rewards, and valid mask for all 4 actions from a board."""
    return _default_computer(board)


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

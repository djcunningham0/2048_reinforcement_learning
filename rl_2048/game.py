"""Game engine for 2048"""

import random
from enum import IntEnum

import torch

PROBABILITY_SPAWN_2 = 0.9

Board = tuple[int, ...]  # flat tuple of 16 ints, indexed as r * 4 + c

_EMPTY_BOARD: Board = (0,) * 16

# Precomputed line indices for each action direction.
# Each action has 4 lines of 4 indices. The indices are ordered so that
# sliding "left" along the line is equivalent to the original action.
_LINES: dict[int, tuple[tuple[int, ...], ...]] = {
    0: ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)),  # UP
    1: ((3, 2, 1, 0), (7, 6, 5, 4), (11, 10, 9, 8), (15, 14, 13, 12)),  # RIGHT
    2: ((12, 8, 4, 0), (13, 9, 5, 1), (14, 10, 6, 2), (15, 11, 7, 3)),  # DOWN
    3: ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)),  # LEFT
}

# Tile value -> one-hot channel index for encode_state
_TILE_TO_CHANNEL = {0: 0, **{2**k: k for k in range(1, 16)}}


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Game2048:
    def __init__(self):
        self.board: Board = _EMPTY_BOARD
        self.score: int = 0

    def reset(self):
        """Clear the board and spawn 2 tiles."""
        self.board = _EMPTY_BOARD
        self.score = 0
        self._spawn_tile()
        self._spawn_tile()

    def step(self, action: Action) -> float:
        """Apply a move and return the reward."""
        new_board, reward = apply_action(self.board, action)
        if new_board == self.board:
            return 0.0
        self.board = new_board
        self.score += int(reward)
        self._spawn_tile()
        return reward

    def _spawn_tile(self):
        """Place a 2 (90%) or 4 (10%) on a random empty cell."""
        value = 2 if random.random() < PROBABILITY_SPAWN_2 else 4
        self.place_tile(value)

    def place_tile(self, value: int):
        """Place a tile of a specific value on a random empty cell."""
        empty = [i for i in range(16) if self.board[i] == 0]
        if not empty:
            raise ValueError("Cannot place tile: board is full")
        idx = random.choice(empty)
        self.board = self.board[:idx] + (value,) + self.board[idx + 1 :]

    def get_valid_actions(self) -> list[Action]:
        """Return actions that would change the board."""
        return [a for a in Action if _can_move(self.board, a)]


def apply_action(board: Board, action: Action) -> tuple[Board, float]:
    """Apply action deterministically. Returns (new_board, merge_score)."""
    cells = list(board)
    total_score = 0.0
    for line_indices in _LINES[action]:
        row = [cells[i] for i in line_indices]
        new_row, score = _slide_row_left(row)
        total_score += score
        for i, idx in enumerate(line_indices):
            cells[idx] = new_row[i]
    return tuple(cells), total_score


def _can_move(board: Board, action: Action) -> bool:
    """Check if an action would change the board."""
    for line_indices in _LINES[action]:
        vals = (
            board[line_indices[0]],
            board[line_indices[1]],
            board[line_indices[2]],
            board[line_indices[3]],
        )
        seen_empty = False
        for i in range(4):
            if vals[i] == 0:
                seen_empty = True
            else:
                if seen_empty:
                    return True
                if i + 1 < 4 and vals[i] == vals[i + 1]:
                    return True
    return False


def _slide_row_left(row: list[int]) -> tuple[list[int], int]:
    """Slide and merge a single row to the left. Returns (new_row, score)."""
    tiles = [x for x in row if x != 0]
    result: list[int] = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            result.append(merged)
            score += merged
            i += 2
        else:
            result.append(tiles[i])
            i += 1
    result.extend([0] * (4 - len(result)))
    return result, score


def encode_state(board: Board) -> torch.Tensor:
    """
    One-hot encode the board. Output shape: (16, 4, 4).

    Channel 0 = empty, channel k = tile 2^k.
    Output shape: (16, 4, 4).

    Example
    -------
    Board:
        [[0, 2, 4, 0],
         [8, 0, 0, 2],
         [0, 0, 0, 0],
         [4, 2, 0, 16]]

    Channel indices (intermediate):
        [[0, 1, 2, 0],
         [3, 0, 0, 1],
         [0, 0, 0, 0],
         [2, 1, 0, 4]]

    Output:
    Result tensor[k, r, c] = 1.0 where k is the channel index for that cell,
    and 0.0 everywhere else. Each (r, c) position has exactly one channel set.

    tensor[0] (empty):
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [1, 1, 1, 1],
         [0, 0, 1, 0]]

    tensor[1] (tile 2):
        [[0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 1, 0, 0]]

    tensor[2] (tile 4):
        [[0, 0, 1, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0]]

    ...

    tensor[15] (tile 32768):
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    """
    tensor = torch.zeros(16, 4, 4)
    encode_state_into(board, tensor)
    return tensor


def encode_state_into(board: Board, out: torch.Tensor):
    """One-hot encode ``board`` into the pre-allocated ``out`` tensor (16, 4, 4) in-place."""
    out.zero_()
    for i, val in enumerate(board):
        r, c = divmod(i, 4)
        out[_TILE_TO_CHANNEL[val], r, c] = 1.0


def make_board(rows: list[list[int]]) -> Board:
    """
    Convert a 4x4 nested list to a flat board tuple. Useful for tests.

    Example usage:
    >>> board = make_board([
    ...     [0, 2, 4, 0],
    ...     [8, 0, 0, 2],
    ...     [0, 0, 0, 0],
    ...     [4, 2, 0, 16],
    ... ])
    >>> print(board)
    (0, 2, 4, 0, 8, 0, 0, 2, 0, 0, 0, 0, 4, 2, 0, 16)
    """
    return tuple(val for row in rows for val in row)

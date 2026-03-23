"""Game engine for 2048"""

import random
from enum import IntEnum

import torch

PROBABILITY_SPAWN_2 = 0.9


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Game2048:
    def __init__(self) -> None:
        self.board: list[list[int]] = [[0] * 4 for _ in range(4)]
        self.score: int = 0
        self._done: bool = False

    def reset(self) -> list[list[int]]:
        """Clear the board and spawn 2 tiles."""
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self._done = False
        self._spawn_tile()
        self._spawn_tile()
        return self._copy_board()

    def step(self, action: Action) -> tuple[list[list[int]], float, bool]:
        """
        Apply a move, spawn a tile, and check game over.

        Returns (board, reward, done).
        """
        new_board, reward = self._apply_action(action)
        if new_board == self.board:
            # Invalid move — no change, no spawn, game-over status unchanged
            return self._copy_board(), 0.0, self._done
        self.board = new_board
        self.score += int(reward)
        self._spawn_tile()
        self._done = self._is_game_over()
        return self._copy_board(), reward, self._done

    def get_valid_actions(self) -> list[Action]:
        """Return actions that actually change the board."""
        return [a for a in Action if self._can_move(a)]

    def _can_move(self, action: Action) -> bool:
        """Check if an action would change the board."""
        lines = _orient(self._copy_board(), action)
        return any(_row_can_move_left(line) for line in lines)

    def _copy_board(self) -> list[list[int]]:
        return [row[:] for row in self.board]

    def _spawn_tile(self) -> None:
        """Place a 2 (90%) or 4 (10%) on a random empty cell."""
        empty = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if not empty:
            return
        r, c = random.choice(empty)
        self.board[r][c] = 2 if random.random() < PROBABILITY_SPAWN_2 else 4

    def _apply_action(self, action: Action) -> tuple[list[list[int]], float]:
        """Apply action without mutating board. Returns (new_board, score)."""
        # Rotate so we can always slide left, then rotate back
        board = self._copy_board()
        board = _orient(board, action)
        total_score = 0.0
        for i in range(4):
            board[i], score = _slide_row_left(board[i])
            total_score += score
        board = _unorient(board, action)
        return board, total_score

    def _is_game_over(self) -> bool:
        """Game is over when no empty cells and no adjacent tiles match."""
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    return False
                if c + 1 < 4 and self.board[r][c] == self.board[r][c + 1]:
                    return False
                if r + 1 < 4 and self.board[r][c] == self.board[r + 1][c]:
                    return False
        return True


def _orient(board: list[list[int]], action: Action) -> list[list[int]]:
    """Rotate/flip board so that the desired action becomes a left slide."""
    if action == Action.LEFT:
        return board
    if action == Action.RIGHT:
        return [row[::-1] for row in board]
    if action == Action.UP:
        return [list(col) for col in zip(*board)]
    # down — transpose then reverse each row
    return [list(col)[::-1] for col in zip(*board)]


def _unorient(board: list[list[int]], action: Action) -> list[list[int]]:
    """Reverse the orient transformation."""
    if action == Action.LEFT:
        return board
    if action == Action.RIGHT:
        return [row[::-1] for row in board]
    if action == Action.UP:
        return [list(col) for col in zip(*board)]
    # down — reverse each row then transpose
    reversed_rows = [row[::-1] for row in board]
    return [list(col) for col in zip(*reversed_rows)]


def _row_can_move_left(row: list[int]) -> bool:
    """Check if sliding this row left would change it."""
    seen_empty = False
    for i in range(4):
        if row[i] == 0:
            seen_empty = True
        else:
            # A tile can move into an earlier empty space
            if seen_empty:
                return True
            # A tile can merge with the next non-zero tile
            if i + 1 < 4 and row[i] == row[i + 1]:
                return True
    return False


def _slide_row_left(row: list[int]) -> tuple[list[int], int]:
    """
    Slide and merge a single row to the left.

    Returns (new row, score gained).
    """
    # Filter out zeros
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
    # Pad with zeros
    result.extend([0] * (4 - len(result)))
    return result, score


def encode_state(board: list[list[int]]) -> torch.Tensor:
    """
    One-hot encode the board.

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
    b = torch.tensor(board, dtype=torch.int32)
    # Map zeros to channel 0, ..., tile 2^k to channel k
    # clamp(0, 15) ensures safe indexing for tiles > 32768
    channels = torch.where(
        b == 0,
        torch.zeros_like(b),
        b.clamp(min=1).float().log2().to(torch.int32),
    )
    channels = channels.clamp(0, 15)
    tensor = torch.zeros(16, 4, 4)
    tensor.scatter_(0, channels.unsqueeze(0), 1.0)
    return tensor

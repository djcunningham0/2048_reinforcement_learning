"""
Bitboard engine for 2048.

Packs the entire 4x4 board into a single 64-bit integer. Each cell is a
4-bit nibble storing the exponent (empty=0, tile 2=1, tile 4=2, ...,
tile 32768=15). Moves are computed via precomputed lookup tables for O(1)
row operations.

Bit layout
----------
Row 0 occupies bits 48-63, row 3 occupies bits 0-15.
Within each 16-bit row, col 0 is the highest nibble (bits 12-15).
Cell (r, c) is at bit offset ``(3 - r) * 16 + (3 - c) * 4``.
"""

import random

import torch

from rl_2048.game import Action, Board, PROBABILITY_SPAWN_2

BitBoard = int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reverse_row(row: int) -> int:
    """Reverse 4 nibbles in a 16-bit row."""
    return (
        ((row & 0xF) << 12)
        | ((row >> 4 & 0xF) << 8)
        | ((row >> 8 & 0xF) << 4)
        | (row >> 12)
    )


def _cell_shift(r: int, c: int) -> int:
    """Bit offset for cell (r, c)."""
    return (3 - r) * 16 + (3 - c) * 4


# ---------------------------------------------------------------------------
# Lookup tables — built once at import time
# ---------------------------------------------------------------------------


def _build_tables() -> tuple[list[int], list[int], list[int]]:
    move_left: list[int] = [0] * 65536
    move_right: list[int] = [0] * 65536
    row_score: list[int] = [0] * 65536

    for row in range(65536):
        # Extract nibbles (left to right: a b c d)
        nibbles = [
            (row >> 12) & 0xF,
            (row >> 8) & 0xF,
            (row >> 4) & 0xF,
            row & 0xF,
        ]

        # Slide left: compact then merge
        tiles = [n for n in nibbles if n != 0]
        result: list[int] = []
        score = 0
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged = tiles[i] + 1  # exponent increments by 1
                result.append(merged)
                score += 1 << merged  # 2^(e+1)
                i += 2
            else:
                result.append(tiles[i])
                i += 1
        result.extend([0] * (4 - len(result)))

        packed = (result[0] << 12) | (result[1] << 8) | (result[2] << 4) | result[3]
        move_left[row] = packed
        row_score[row] = score

        # Right = reverse, slide left, reverse
        rev = _reverse_row(row)
        left_of_rev = move_left[rev] if rev < row else -1  # may not be computed yet
        # We'll fill move_right in a second pass
        move_right[row] = 0  # placeholder

    # Second pass for move_right (all move_left entries are now populated)
    for row in range(65536):
        rev = _reverse_row(row)
        move_right[row] = _reverse_row(move_left[rev])

    return move_left, move_right, row_score


_MOVE_LEFT, _MOVE_RIGHT, _ROW_SCORE = _build_tables()


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def _transpose(bb: BitBoard) -> BitBoard:
    """Transpose the 4x4 nibble matrix (swap rows and columns)."""
    result = 0
    for r in range(4):
        for c in range(4):
            shift_src = _cell_shift(r, c)
            shift_dst = _cell_shift(c, r)
            nibble = (bb >> shift_src) & 0xF
            result |= nibble << shift_dst
    return result


# ---------------------------------------------------------------------------
# Move functions
# ---------------------------------------------------------------------------


def _move_left(bb: BitBoard) -> tuple[BitBoard, int]:
    score = 0
    result = 0
    for r in range(4):
        shift = (3 - r) * 16
        row = (bb >> shift) & 0xFFFF
        result |= _MOVE_LEFT[row] << shift
        score += _ROW_SCORE[row]
    return result, score


def _move_right(bb: BitBoard) -> tuple[BitBoard, int]:
    score = 0
    result = 0
    for r in range(4):
        shift = (3 - r) * 16
        row = (bb >> shift) & 0xFFFF
        result |= _MOVE_RIGHT[row] << shift
        score += _ROW_SCORE[_reverse_row(row)]
    return result, score


def _move_up(bb: BitBoard) -> tuple[BitBoard, int]:
    t = _transpose(bb)
    moved, score = _move_left(t)
    return _transpose(moved), score


def _move_down(bb: BitBoard) -> tuple[BitBoard, int]:
    t = _transpose(bb)
    moved, score = _move_right(t)
    return _transpose(moved), score


_MOVE_FNS = {
    Action.LEFT: _move_left,
    Action.RIGHT: _move_right,
    Action.UP: _move_up,
    Action.DOWN: _move_down,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_action(board: BitBoard, action: Action) -> tuple[BitBoard, float]:
    """Apply action deterministically. Returns (new_board, merge_score)."""
    new_board, score = _MOVE_FNS[action](board)
    return new_board, float(score)


def get_valid_actions(board: BitBoard) -> list[Action]:
    """Return actions that would change the board."""
    return [a for a in Action if _MOVE_FNS[a](board)[0] != board]


def encode_state(board: BitBoard) -> torch.Tensor:
    """One-hot encode the board. Output shape: (16, 4, 4).

    Channel 0 = empty, channel k = tile 2^k.
    """
    tensor = torch.zeros(16, 4, 4)
    for r in range(4):
        for c in range(4):
            exp = (board >> _cell_shift(r, c)) & 0xF
            tensor[exp, r, c] = 1.0
    return tensor


def make_bitboard(rows: list[list[int]]) -> BitBoard:
    """Convert a 4x4 nested list of tile values to a bitboard."""
    bb = 0
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            if val:
                exp = val.bit_length() - 1
                bb |= exp << _cell_shift(r, c)
    return bb


def board_to_bitboard(board: Board) -> BitBoard:
    """Convert a flat tuple board (tile values) to a bitboard."""
    bb = 0
    for i, val in enumerate(board):
        if val:
            r, c = divmod(i, 4)
            exp = val.bit_length() - 1
            bb |= exp << _cell_shift(r, c)
    return bb


def bitboard_to_board(bb: BitBoard) -> Board:
    """Convert a bitboard to a flat tuple board (tile values)."""
    cells: list[int] = []
    for r in range(4):
        for c in range(4):
            exp = (bb >> _cell_shift(r, c)) & 0xF
            cells.append(0 if exp == 0 else (1 << exp))
    return tuple(cells)


class BitBoardGame2048:
    """2048 game using bitboard representation."""

    def __init__(self):
        self.board: BitBoard = 0
        self.score: int = 0

    def reset(self):
        """Clear the board and spawn 2 tiles."""
        self.board = 0
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
        empty: list[tuple[int, int]] = []
        for r in range(4):
            for c in range(4):
                if (self.board >> _cell_shift(r, c)) & 0xF == 0:
                    empty.append((r, c))
        if not empty:
            return
        r, c = random.choice(empty)
        exp = 1 if random.random() < PROBABILITY_SPAWN_2 else 2
        self.board |= exp << _cell_shift(r, c)

    def get_valid_actions(self) -> list[Action]:
        """Return actions that would change the board."""
        return get_valid_actions(self.board)

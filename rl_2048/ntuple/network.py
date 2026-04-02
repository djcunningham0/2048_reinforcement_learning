"""
N-tuple network for 2048 board evaluation.

Uses lookup tables (LUTs) indexed by patterns of board tile values, summed across all
patterns and all 8 symmetries of the square.
"""

from pathlib import Path

import numba
import numpy as np
import torch

from rl_2048.game import Board

# Precomputed tile value -> LUT index (0-15). Index 0 = empty, index k = tile 2^k.
# Supports tile values up to 2^15 = 32768.
_TILE_LOOKUP = np.zeros(32769, dtype=np.int64)
for _k in range(1, 16):
    _TILE_LOOKUP[2**_k] = _k


def _build_symmetries() -> tuple[tuple[int, ...], ...]:
    """
    Precompute the 8 symmetry permutations of a 4x4 grid. Each permutation maps position
    index -> new position index.

    Returns a tuple of 8 tuples (4 rotations x 2 flip states), each containing 16 ints.

    Example
    -------
    >>> syms = _build_symmetries()
    >>> syms[0]  # identity (no rotation, no flip)
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    >>> syms[2]  # 90 degree clockwise rotation, no flip
    (3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12)
    """
    symmetries = []
    for rotate in range(4):
        for flip in (False, True):
            perm = [0] * 16
            for pos in range(16):
                r, c = divmod(pos, 4)
                # Apply rotation (CW)
                for _ in range(rotate):
                    r, c = c, 3 - r
                # Apply horizontal flip
                if flip:
                    c = 3 - c
                perm[pos] = r * 4 + c
            symmetries.append(tuple(perm))
    return tuple(symmetries)


_SYMMETRIES = _build_symmetries()


@numba.njit(cache=True)
def _evaluate(
    board_indices: np.ndarray,
    sym_positions: np.ndarray,
    powers: np.ndarray,
    luts: np.ndarray,
    lut_offsets: np.ndarray,
    num_patterns: int,
) -> float:
    """Evaluate a board by summing LUT lookups across all patterns and symmetries."""
    total = 0.0
    for i in range(num_patterns):
        offset = lut_offsets[i]
        for s in range(8):
            lut_index = 0
            for k in range(sym_positions.shape[2]):
                lut_index += board_indices[sym_positions[i, s, k]] * powers[i, k]
            total += luts[offset + lut_index]
    return total


@numba.njit(cache=True)
def _update(
    board_indices: np.ndarray,
    sym_positions: np.ndarray,
    powers: np.ndarray,
    luts: np.ndarray,
    lut_offsets: np.ndarray,
    num_patterns: int,
    delta: float,
):
    """Add delta to every LUT entry accessed when evaluating this board."""
    for i in range(num_patterns):
        offset = lut_offsets[i]
        for s in range(8):
            lut_index = 0
            for k in range(sym_positions.shape[2]):
                lut_index += board_indices[sym_positions[i, s, k]] * powers[i, k]
            luts[offset + lut_index] += delta


class NTupleNetwork:
    """N-tuple network with lookup tables and 8-fold symmetry."""

    def __init__(self, patterns: list[tuple[int, ...]], v_init: float = 0.0):
        self.patterns = patterns
        self.num_patterns = len(patterns)
        tuple_size = len(patterns[0])

        if any(len(p) != tuple_size for p in patterns):
            raise ValueError("All patterns must have the same length")

        # Precompute transformed patterns for all symmetries as a 3D numpy array
        # Shape: (num_patterns, 8, tuple_size)
        self._sym_patterns = np.empty(
            (self.num_patterns, 8, tuple_size), dtype=np.int64
        )
        for i, pattern in enumerate(patterns):
            for s, sym in enumerate(_SYMMETRIES):
                for k, p in enumerate(pattern):
                    self._sym_patterns[i, s, k] = sym[p]

        # Powers of 16 for index computation: [16^(n-1), 16^(n-2), ..., 1]
        # Shape: (num_patterns, tuple_size)
        self._powers = np.empty((self.num_patterns, tuple_size), dtype=np.int64)
        for i, pattern in enumerate(patterns):
            n = len(pattern)
            for k in range(n):
                self._powers[i, k] = 16 ** (n - 1 - k)

        # Concatenated LUT — all patterns share one flat array, accessed via offsets
        lut_sizes = [16 ** len(p) for p in patterns]
        self._lut_offsets = np.array(
            [sum(lut_sizes[:i]) for i in range(len(lut_sizes))], dtype=np.int64
        )
        total_size = sum(lut_sizes)
        if v_init:
            per_weight = v_init / (self.num_patterns * 8)
            self._luts = np.full(total_size, per_weight, dtype=np.float64)
        else:
            self._luts = np.zeros(total_size, dtype=np.float64)

    @property
    def luts(self) -> list[np.ndarray]:
        """View into the concatenated LUT array, split per pattern (for save/load)."""
        views = []
        sizes = [16 ** len(p) for p in self.patterns]
        for i, size in enumerate(sizes):
            offset = self._lut_offsets[i]
            views.append(self._luts[offset : offset + size])
        return views

    @luts.setter
    def luts(self, value: list[np.ndarray]):
        """Set LUT data from a list of per-pattern arrays (used by load)."""
        for i, arr in enumerate(value):
            offset = self._lut_offsets[i]
            size = 16 ** len(self.patterns[i])
            self._luts[offset : offset + size] = arr.astype(np.float64)

    def _board_indices(self, board: Board) -> np.ndarray:
        """Convert board tile values to LUT indices (0-15).

        Example
        -------
        >>> board = (128, 64, 2, 4, 0, 8, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0)
        >>> network._board_indices(board)
        array([7, 6, 1, 2, 0, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0])
        """
        return _TILE_LOOKUP[np.array(board, dtype=np.int64)]

    def evaluate(self, board: Board) -> float:
        """Evaluate a board by summing LUT lookups across all patterns and symmetries."""
        indices = self._board_indices(board)
        return _evaluate(
            indices,
            self._sym_patterns,
            self._powers,
            self._luts,
            self._lut_offsets,
            self.num_patterns,
        )

    def update(self, board: Board, delta: float):
        """Add delta to every LUT entry accessed when evaluating this board."""
        indices = self._board_indices(board)
        _update(
            indices,
            self._sym_patterns,
            self._powers,
            self._luts,
            self._lut_offsets,
            self.num_patterns,
            delta,
        )

    def evaluate_batch(self, boards: list[Board]) -> torch.Tensor:
        """Evaluate a batch of boards. Conforms to the ValueFunction protocol."""
        values = torch.empty(len(boards))
        for j, board in enumerate(boards):
            values[j] = self.evaluate(board)
        return values

    def save(self, path: str | Path):
        """Save network to a compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict: dict[str, np.ndarray] = {
            f"lut_{i}": lut.copy() for i, lut in enumerate(self.luts)
        }
        # Store pattern lengths and flat pattern data as int arrays
        save_dict["pattern_lengths"] = np.array(
            [len(p) for p in self.patterns], dtype=np.int32
        )
        save_dict["pattern_data"] = np.array(
            [pos for p in self.patterns for pos in p], dtype=np.int32
        )
        np.savez_compressed(path, **save_dict)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> "NTupleNetwork":
        """Load network from a .npz file."""
        data = np.load(path, allow_pickle=False)
        lengths = data["pattern_lengths"].tolist()
        flat = data["pattern_data"].tolist()
        patterns: list[tuple[int, ...]] = []
        offset = 0
        for n in lengths:
            patterns.append(tuple(flat[offset : offset + n]))
            offset += n
        network = cls(patterns)
        lut_list = [data[f"lut_{i}"] for i in range(len(patterns))]
        network.luts = lut_list
        return network

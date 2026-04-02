"""
N-tuple network for 2048 board evaluation.

Uses lookup tables (LUTs) indexed by patterns of board tile values, summed across all
patterns and all 8 symmetries of the square.
"""

from pathlib import Path

import numpy as np
import torch

from rl_2048.game import Board, _TILE_TO_CHANNEL


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


class NTupleNetwork:
    """N-tuple network with lookup tables and 8-fold symmetry."""

    def __init__(self, patterns: list[tuple[int, ...]]):
        self.patterns = patterns
        self.num_patterns = len(patterns)

        # Precompute transformed patterns for all symmetries
        # sym_patterns[pattern_idx][sym_idx] = tuple of transformed positions
        self._sym_patterns: list[list[tuple[int, ...]]] = []
        for pattern in patterns:
            sym_group = []
            for sym in _SYMMETRIES:
                sym_group.append(tuple(sym[p] for p in pattern))
            self._sym_patterns.append(sym_group)

        # Powers of 16 for index computation: [16^(n-1), 16^(n-2), ..., 1]
        self._powers: list[tuple[int, ...]] = []
        for pattern in patterns:
            n = len(pattern)
            self._powers.append(tuple(16 ** (n - 1 - i) for i in range(n)))

        # Allocate LUTs — one per pattern, shared across symmetries
        self.luts: list[np.ndarray] = []
        for pattern in patterns:
            n = len(pattern)
            self.luts.append(np.zeros(16**n, dtype=np.float32))

    def _board_indices(self, board: Board) -> list[int]:
        """Convert board tile values to LUT indices (0-15).

        Example
        -------
        >>> board = (128, 64, 2, 4, 0, 8, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0)
        >>> network._board_indices(board)
        [7, 6, 1, 2, 0, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0]
        """
        return [_TILE_TO_CHANNEL[v] for v in board]

    def evaluate(self, board: Board) -> float:
        """
        Evaluate a board by summing LUT lookups across all patterns and symmetries.
        """
        indices = self._board_indices(board)
        total = 0.0
        for i in range(self.num_patterns):
            powers = self._powers[i]
            lut = self.luts[i]
            for sym_positions in self._sym_patterns[i]:
                lut_index = 0
                for p, pw in zip(sym_positions, powers):
                    lut_index += indices[p] * pw
                total += float(lut[lut_index])
        return total

    def update(self, board: Board, delta: float):
        """Add delta to every LUT entry accessed when evaluating this board."""
        indices = self._board_indices(board)
        for i in range(self.num_patterns):
            powers = self._powers[i]
            lut = self.luts[i]
            for sym_positions in self._sym_patterns[i]:
                lut_index = 0
                for p, pw in zip(sym_positions, powers):
                    lut_index += indices[p] * pw
                lut[lut_index] += delta

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
            f"lut_{i}": lut for i, lut in enumerate(self.luts)
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
        for i in range(len(patterns)):
            network.luts[i] = data[f"lut_{i}"].astype(np.float32)
        return network

"""Tests for NTupleNetwork."""

import numpy as np
import pytest
import torch

from rl_2048.game import Board
from rl_2048.ntuple.network import NTupleNetwork, _SYMMETRIES

# Small 2-tuple patterns for fast tests
SMALL_PATTERNS = [(0, 1), (4, 5)]

# A non-trivial board for testing
SAMPLE_BOARD: Board = (
    2, 4, 8, 16,
    32, 64, 128, 256,
    512, 1024, 2048, 4096,
    8192, 16384, 32768, 0,
)  # fmt: skip


class TestSymmetries:
    def test_all_permutations_valid(self):
        """Each symmetry is a permutation of 0-15."""
        for sym in _SYMMETRIES:
            assert sorted(sym) == list(range(16))

    def test_eight_distinct_symmetries(self):
        assert len(_SYMMETRIES) == 8
        assert len(set(_SYMMETRIES)) == 8


class TestNTupleNetwork:
    def test_zero_initialization(self):
        network = NTupleNetwork(SMALL_PATTERNS)
        assert network.evaluate(SAMPLE_BOARD) == 0.0

    def test_update_changes_value(self):
        network = NTupleNetwork(SMALL_PATTERNS)
        delta = 1.0
        network.update(SAMPLE_BOARD, delta)
        value = network.evaluate(SAMPLE_BOARD)
        # Each update touches num_patterns * 8 entries, each incremented by delta.
        # Evaluate sums the same entries, so value = num_patterns * 8 * delta.
        expected = len(SMALL_PATTERNS) * 8 * delta
        assert value == pytest.approx(expected)

    def test_symmetry_invariance(self):
        """A board and its 90-degree rotation should evaluate identically."""
        network = NTupleNetwork(SMALL_PATTERNS)
        # Set some non-zero weights
        network.update(SAMPLE_BOARD, 5.0)

        # Rotate 90 CW: (r, c) -> (c, 3-r)
        rotated = [0] * 16
        for pos in range(16):
            r, c = divmod(pos, 4)
            new_r, new_c = c, 3 - r
            rotated[new_r * 4 + new_c] = SAMPLE_BOARD[pos]
        rotated_board: Board = tuple(rotated)

        assert network.evaluate(SAMPLE_BOARD) == pytest.approx(
            network.evaluate(rotated_board)
        )

    def test_different_boards_different_values(self):
        """After updating one board, a different board should have a different value."""
        network = NTupleNetwork(SMALL_PATTERNS)
        board_a: Board = (2, 4) + (0,) * 14
        board_b: Board = (8, 16) + (0,) * 14
        network.update(board_a, 1.0)
        # board_b was not updated, and its tile pattern differs from board_a
        assert network.evaluate(board_a) != network.evaluate(board_b)

    def test_evaluate_batch_matches_individual(self):
        network = NTupleNetwork(SMALL_PATTERNS)
        network.update(SAMPLE_BOARD, 3.0)
        boards = [SAMPLE_BOARD, (0,) * 16, (2, 4) + (0,) * 14]
        batch_values = network.evaluate_batch(boards)

        assert isinstance(batch_values, torch.Tensor)
        assert batch_values.shape == (3,)
        for i, board in enumerate(boards):
            assert batch_values[i].item() == pytest.approx(network.evaluate(board))

    def test_save_load_roundtrip(self, tmp_path):
        network = NTupleNetwork(SMALL_PATTERNS)
        network.update(SAMPLE_BOARD, 2.5)
        original_value = network.evaluate(SAMPLE_BOARD)

        path = tmp_path / "test_model.npz"
        network.save(path)
        loaded = NTupleNetwork.load(path)

        assert loaded.patterns == network.patterns
        assert loaded.evaluate(SAMPLE_BOARD) == pytest.approx(original_value)
        for i in range(len(network.luts)):
            np.testing.assert_array_equal(loaded.luts[i], network.luts[i])

    def test_standard_6tuple_patterns(self):
        """Verify standard patterns can be instantiated without error."""
        from rl_2048.ntuple.config import DEFAULT_PATTERNS

        network = NTupleNetwork(DEFAULT_PATTERNS)
        assert network.num_patterns == 4
        # Each LUT should have 16^6 entries
        for lut in network.luts:
            assert len(lut) == 16**6

"""Tests for expectimax search."""

import pytest
import torch

from rl_2048.game import Action, Board, apply_action, make_board
from rl_2048.expectimax import (
    DepthSchedule,
    _build_chance_node,
    _evaluate_leaves,
    expectimax_action,
)


def _zero_value_fn(boards: list[Board]) -> torch.Tensor:
    return torch.zeros(len(boards))


class TestExpectimaxAction:
    def test_depth_0_picks_highest_reward(self):
        """With V=0 everywhere, depth=0 should pick the action with highest merge reward."""
        # Board where LEFT, RIGHT, DOWN are valid
        # - LEFT: reward=4
        # - RIGHT: reward=4
        # - DOWN: reward=0
        board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        action = expectimax_action(board, _zero_value_fn, depth=0)
        _, left_reward = apply_action(board, Action.LEFT)
        _, right_reward = apply_action(board, Action.RIGHT)
        _, chosen_reward = apply_action(board, action)
        assert action in [Action.LEFT, Action.RIGHT]
        assert chosen_reward >= left_reward or chosen_reward >= right_reward
        assert chosen_reward == 4.0

    def test_invalid_actions_never_selected(self):
        """Should never return an action that doesn't change the board."""
        # Board where only DOWN and RIGHT are valid
        board = make_board([
            [2, 4, 2, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        for depth in [0, 1]:
            action = expectimax_action(board, _zero_value_fn, depth=depth)
            afterstate, _ = apply_action(board, action)
            assert (
                afterstate != board
            ), f"depth={depth} returned invalid action {action}"


class TestChanceNode:
    def test_probabilities_sum_to_1(self):
        """Probabilities across chance node children should sum to 1."""
        afterstate = make_board([
            [2, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        leaves: list[Board] = []
        chance = _build_chance_node(afterstate, reward=0.0, depth=1, leaves=leaves)
        total_prob = sum(prob for prob, _ in chance.children)
        assert abs(total_prob - 1.0) < 1e-9


class TestLeafEvaluation:
    def test_leaf_value_includes_reward(self):
        """Leaf evaluation should be max_a [reward + V(afterstate)]."""
        board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        # With V=0, leaf value = max reward across actions = 4 (merging the 2s)
        values = _evaluate_leaves([board], _zero_value_fn)
        assert values[0].item() == 4.0


class TestDepthSchedule:
    def test_fixed_always_returns_same_depth(self):
        empty_board = make_board([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
        ])
        full_board = make_board([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [2, 4, 8, 16],
            [32, 64, 128, 0],
        ])
        schedule = DepthSchedule.fixed(2)
        assert schedule.get_depth(empty_board) == 2
        assert schedule.get_depth(full_board) == 2

    @pytest.mark.parametrize(
        "empty_cells, expected_depth",
        [
            (8, 1),  # above high threshold
            (6, 1),  # exactly at high threshold
            (5, 2),  # between thresholds
            (3, 2),  # exactly at middle threshold
            (2, 3),  # between middle and fallback
            (0, 3),  # no empty cells (fallback)
        ],
    )
    def test_schedule_returns_correct_depth(self, empty_cells, expected_depth):
        schedule = DepthSchedule(thresholds=[(6, 1), (3, 2), (0, 3)])
        # Build a board with the desired number of empty cells
        flat = [0] * empty_cells + [2] * (16 - empty_cells)
        board = make_board([flat[i : i + 4] for i in range(0, 16, 4)])
        assert schedule.get_depth(board) == expected_depth

    def test_unsorted_thresholds_raises(self):
        with pytest.raises(ValueError, match="sorted descending"):
            DepthSchedule(thresholds=[(4, 2), (8, 1), (0, 3)])

    def test_missing_fallback_raises(self):
        with pytest.raises(ValueError, match="fallback"):
            DepthSchedule(thresholds=[(8, 1), (4, 2)])

    def test_expectimax_action_with_schedule(self):
        """expectimax_action should accept a DepthSchedule and return a valid action."""
        board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        schedule = DepthSchedule.fixed(1)
        action = expectimax_action(board, _zero_value_fn, schedule)
        afterstate, _ = apply_action(board, action)
        assert afterstate != board

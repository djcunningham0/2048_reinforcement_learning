"""Tests for expectimax search."""

import time

import torch

from rl_2048.game import Action, Board, apply_action, make_board
from rl_2048.expectimax import (
    _build_chance_node,
    _build_max_node,
    _evaluate_leaves,
    _evaluate_node,
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


class TestTimeBudget:
    def test_returns_valid_action(self):
        """Time-budgeted search should return a valid action."""
        board = make_board([
            [2, 4, 8, 16],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 4, 0],
        ])
        action = expectimax_action(board, _zero_value_fn, time_budget=0.5)
        afterstate, _ = apply_action(board, action)
        assert afterstate != board

    def test_tiny_budget_still_returns(self):
        """Even a very small time budget should return at least a greedy result."""
        board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        action = expectimax_action(board, _zero_value_fn, time_budget=0.0001)
        afterstate, _ = apply_action(board, action)
        assert afterstate != board

    def test_max_depth_caps_search(self):
        """max_depth=0 should behave like depth=0 regardless of time budget."""
        board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        action_capped = expectimax_action(
            board, _zero_value_fn, time_budget=10.0, max_depth=0
        )
        action_greedy = expectimax_action(board, _zero_value_fn, depth=0)
        assert action_capped == action_greedy

    def test_time_budget_respected(self):
        """Move selection should complete within a reasonable multiple of time_budget."""
        board = make_board([
            [2, 4, 8, 16],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 4, 0],
        ])
        budget = 0.1
        t0 = time.perf_counter()
        expectimax_action(board, _zero_value_fn, time_budget=budget)
        elapsed = time.perf_counter() - t0
        assert elapsed < budget * 3, f"Took {elapsed:.3f}s with budget {budget}s"


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
        """MaxLeafNode evaluation should be max_a [reward + V(afterstate)]."""
        board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        # Build a depth-0 max node which expands actions into afterstate leaves
        leaves: list[Board] = []
        node = _build_max_node(board, depth=0, leaves=leaves)
        leaf_values = _evaluate_leaves(leaves, _zero_value_fn)
        # With V=0, value = max reward across actions = 4 (merging the 2s)
        assert _evaluate_node(node, leaf_values) == 4.0

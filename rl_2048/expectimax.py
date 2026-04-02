"""
Expectimax search for 2048.

Main entrypoint is `expectimax_action`, which builds a depth-limited game tree of
alternating max (player move) and chance (random tile spawn) nodes, batch-evaluates all
leaf boards in a single forward pass, and backpropagates to select the best action.

Two value-function adapters are provided to wrap a `ConvNetwork`:
- `make_afterstate_value_fn` for afterstate-value models (output_dim=1).
- `make_dqn_value_fn` for DQN models (output_dim=4).
"""

import time
from dataclasses import dataclass
from typing import Protocol

import torch

from rl_2048.game import (
    Action,
    Board,
    PROBABILITY_SPAWN_2,
    apply_action,
    batch_encode_states,
)
from rl_2048.network import ConvNetwork


class ValueFunction(Protocol):
    """Batched board evaluator: list[Board] -> 1D tensor of values."""

    def __call__(self, boards: list[Board]) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Tree nodes
# ---------------------------------------------------------------------------


@dataclass
class _LeafNode:
    """Leaf afterstate to be evaluated directly by the value function."""

    index: int  # index into the collected leaves list


@dataclass
class _MaxLeafNode:
    """Terminal max node: picks best action using leaf afterstate values."""

    children: list[tuple[float, int]]  # (reward, leaf_index)


@dataclass
class _MaxNode:
    """Picks the best action. Children keyed by Action."""

    children: dict[Action, "_ChanceNode"]


@dataclass
class _ChanceNode:
    """Averages over random tile placements on an afterstate."""

    reward: float
    children: list[tuple[float, "_MaxNode | _MaxLeafNode | _LeafNode"]]


# ---------------------------------------------------------------------------
# Phase 1: tree expansion
# ---------------------------------------------------------------------------

_Node = _MaxNode | _MaxLeafNode | _LeafNode


class DeadlineExceeded(Exception):
    """Raised when tree expansion exceeds the time budget."""


def _build_max_node(
    board: Board,
    depth: int,
    leaves: list[Board],
    deadline: float | None = None,
) -> _Node:
    if depth <= 0:
        # Expand actions and store afterstates as leaves for direct evaluation
        children: list[tuple[float, int]] = []
        for action in Action:
            afterstate, reward = apply_action(board, action)
            if afterstate != board:
                idx = len(leaves)
                leaves.append(afterstate)
                children.append((reward, idx))
        if not children:
            # Terminal state — no valid moves
            idx = len(leaves)
            leaves.append(board)
            return _LeafNode(index=idx)
        return _MaxLeafNode(children=children)

    if deadline is not None and time.perf_counter() >= deadline:
        raise DeadlineExceeded

    action_children: dict[Action, _ChanceNode] = {}
    for action in Action:
        afterstate, reward = apply_action(board, action)
        if afterstate == board:
            continue
        action_children[action] = _build_chance_node(
            afterstate=afterstate,
            reward=reward,
            depth=depth,
            leaves=leaves,
            deadline=deadline,
        )

    if not action_children:
        # Terminal state — no valid actions
        idx = len(leaves)
        leaves.append(board)
        return _LeafNode(index=idx)

    return _MaxNode(children=action_children)


def _build_chance_node(
    afterstate: Board,
    reward: float,
    depth: int,
    leaves: list[Board],
    deadline: float | None = None,
) -> _ChanceNode:
    empty = [i for i in range(16) if afterstate[i] == 0]
    cell_prob = 1.0 / len(empty)
    children: list[tuple[float, _Node]] = []
    tile_probs = ((2, PROBABILITY_SPAWN_2), (4, 1.0 - PROBABILITY_SPAWN_2))
    for i in empty:
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineExceeded
        for tile, tile_prob in tile_probs:
            board = afterstate[:i] + (tile,) + afterstate[i + 1 :]
            prob = cell_prob * tile_prob
            child = _build_max_node(board, depth - 1, leaves, deadline)
            children.append((prob, child))
    return _ChanceNode(reward=reward, children=children)


# ---------------------------------------------------------------------------
# Phase 2: batch evaluation and backpropagation
# ---------------------------------------------------------------------------


def _evaluate_leaves(leaves: list[Board], value_fn: ValueFunction) -> torch.Tensor:
    """Evaluate leaf afterstates directly with the value function."""
    if not leaves:
        return torch.tensor([])
    return value_fn(leaves)


def _evaluate_node(
    node: _Node | _ChanceNode,
    leaf_values: torch.Tensor,
) -> float:
    if isinstance(node, _LeafNode):
        return leaf_values[node.index].item()
    if isinstance(node, _MaxLeafNode):
        return max(
            reward + leaf_values[idx].item() for reward, idx in node.children
        )
    if isinstance(node, _ChanceNode):
        expected = sum(
            prob * _evaluate_node(child, leaf_values) for prob, child in node.children
        )
        return node.reward + expected
    # _MaxNode
    return max(_evaluate_node(child, leaf_values) for child in node.children.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class _SearchResult:
    """Result from a single depth of expectimax search."""

    action: Action
    leaf_count: int
    elapsed: float


def _expectimax_at_depth(
    board: Board,
    value_fn: ValueFunction,
    depth: int,
    deadline: float | None = None,
) -> _SearchResult:
    """Run expectimax at a fixed depth. Raises DeadlineExceeded if time runs out."""
    t0 = time.perf_counter()
    leaves: list[Board] = []
    root_actions: list[tuple[Action, _ChanceNode]] = []

    for action in Action:
        afterstate, reward = apply_action(board, action)
        if afterstate == board:
            continue
        chance = _build_chance_node(
            afterstate=afterstate,
            reward=reward,
            depth=depth,
            leaves=leaves,
            deadline=deadline,
        )
        root_actions.append((action, chance))

    if not root_actions:
        return _SearchResult(Action.UP, 0, time.perf_counter() - t0)

    if deadline is not None and time.perf_counter() >= deadline:
        raise DeadlineExceeded

    leaf_values = _evaluate_leaves(leaves, value_fn)

    best_action = Action.UP
    best_value = float("-inf")
    for action, chance in root_actions:
        value = _evaluate_node(chance, leaf_values)
        if value > best_value:
            best_value = value
            best_action = action

    return _SearchResult(best_action, len(leaves), time.perf_counter() - t0)


def expectimax_action(
    board: Board,
    value_fn: ValueFunction,
    depth: int = 1,
    time_budget: float | None = None,
    max_depth: int = 10,
) -> Action:
    """
    Select the best action using expectimax search.

    Parameters
    ----------
    board : Board
        Current board state (pre-move, post-spawn).
    value_fn : ValueFunction
        Batched evaluation: list[Board] -> Tensor of V(afterstate) values.
    depth : int
        Number of max-chance plies (fixed-depth mode, used when time_budget is None).
    time_budget : float | None
        Seconds allowed per move. When set, uses iterative deepening from 0 up to
        max_depth, returning the best action from the deepest completed search.
    max_depth : int
        Upper bound on search depth in timed mode (ignored when time_budget is None).
    """
    if time_budget is None:
        return _expectimax_at_depth(board, value_fn, depth).action

    # Iterative deepening with cost estimation.
    # The dominant cost is the neural network forward pass on leaf boards,
    # which scales linearly with leaf count. Leaf count grows exponentially
    # with depth (~empty_cells * 2 per ply). We estimate next-depth cost
    # from the observed cost-per-leaf and leaf-count growth ratio.
    deadline = time.perf_counter() + time_budget
    result = _expectimax_at_depth(board, value_fn, 0)
    best_action = result.action

    prev = result
    for d in range(1, max_depth + 1):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        # Estimate next depth cost from leaf-count growth.
        # Branching factor per ply ≈ valid_actions × empty_cells × 2 (tile values).
        # Apply 2x safety margin since larger batches have super-linear overhead.
        if prev.leaf_count > 0 and prev.elapsed > 0:
            empty = sum(1 for v in board if v == 0)
            branching = max(10, empty * 2 * 3)
            est_cost = prev.elapsed * branching * 2
            if remaining < est_cost:
                break
        try:
            result = _expectimax_at_depth(board, value_fn, d, deadline)
        except DeadlineExceeded:
            break
        best_action = result.action
        prev = result

    return best_action


# ---------------------------------------------------------------------------
# Value function adapters
# ---------------------------------------------------------------------------


def make_afterstate_value_fn(
    model: ConvNetwork,
    device: str | torch.device,
) -> ValueFunction:
    """Wrap an afterstate model (output_dim=1) as a ValueFunction."""
    device = torch.device(device)

    def value_fn(boards: list[Board]) -> torch.Tensor:
        encoded = batch_encode_states(boards)
        with torch.no_grad():
            return model(encoded.to(device)).squeeze(-1).cpu()

    return value_fn


def make_dqn_value_fn(model: ConvNetwork, device: str | torch.device) -> ValueFunction:
    """
    Wrap a DQN model (output_dim=4) as a ValueFunction.

    Uses max_a Q(s, a) as the state value for afterstates.
    """
    device = torch.device(device)

    def value_fn(boards: list[Board]) -> torch.Tensor:
        encoded = batch_encode_states(boards)
        with torch.no_grad():
            q_all = model(encoded.to(device))  # (N, 4)
            return q_all.max(dim=1).values.cpu()

    return value_fn

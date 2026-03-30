"""
Expectimax search for 2048.

Main entrypoint is `expectimax_action`, which builds a depth-limited game tree of
alternating max (player move) and chance (random tile spawn) nodes, batch-evaluates all
leaf boards in a single forward pass, and backpropagates to select the best action.

Two value-function adapters are provided to wrap a `ConvNetwork`:
- `make_afterstate_value_fn` for afterstate-value models (output_dim=1).
- `make_dqn_value_fn` for DQN models (output_dim=4).

If new methods/model types are added, you should implement a new value-function adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from rl_2048.game import (
    Action,
    Board,
    PROBABILITY_SPAWN_2,
    apply_action,
    encode_state,
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
    """Leaf board to be evaluated by the value function."""

    index: int  # index into the collected leaves list


@dataclass
class _ChanceNode:
    """Averages over random tile placements on an afterstate."""

    reward: float
    children: list[tuple[float, _MaxNode | _LeafNode]]  # (probability, node)


@dataclass
class _MaxNode:
    """Picks the best action. Children keyed by Action."""

    children: dict[Action, _ChanceNode]


# ---------------------------------------------------------------------------
# Phase 1: tree expansion
# ---------------------------------------------------------------------------


def _build_max_node(
    board: Board,
    depth: int,
    leaves: list[Board],
) -> _MaxNode | _LeafNode:
    if depth <= 0:
        idx = len(leaves)
        leaves.append(board)
        return _LeafNode(index=idx)

    children: dict[Action, _ChanceNode] = {}
    for action in Action:
        afterstate, reward = apply_action(board, action)
        if afterstate == board:
            continue
        children[action] = _build_chance_node(afterstate, reward, depth, leaves)

    if not children:
        # Terminal state — no valid actions
        idx = len(leaves)
        leaves.append(board)
        return _LeafNode(index=idx)

    return _MaxNode(children=children)


def _build_chance_node(
    afterstate: Board,
    reward: float,
    depth: int,
    leaves: list[Board],
) -> _ChanceNode:
    empty = [i for i in range(16) if afterstate[i] == 0]
    cell_prob = 1.0 / len(empty)
    children: list[tuple[float, _MaxNode | _LeafNode]] = []
    tile_probs = ((2, PROBABILITY_SPAWN_2), (4, 1.0 - PROBABILITY_SPAWN_2))
    for i in empty:
        for tile, tile_prob in tile_probs:
            board = afterstate[:i] + (tile,) + afterstate[i + 1 :]
            prob = cell_prob * tile_prob
            child = _build_max_node(board, depth - 1, leaves)
            children.append((prob, child))
    return _ChanceNode(reward=reward, children=children)


# ---------------------------------------------------------------------------
# Phase 2: batch evaluation and backpropagation
# ---------------------------------------------------------------------------


def _evaluate_leaves(leaf_boards: list[Board], value_fn: ValueFunction) -> torch.Tensor:
    """
    Evaluate each leaf board as max_a [reward_a + V(afterstate_a)].

    Returns a 1D tensor of values, one per leaf.
    """
    if not leaf_boards:
        return torch.tensor([])

    # Collect all afterstates across all leaves
    all_afterstates: list[Board] = []
    leaf_info: list[tuple[int, list[float]]] = []  # (count, rewards) per leaf

    for board in leaf_boards:
        afterstates_rewards = []
        for action in Action:
            afterstate, reward = apply_action(board, action)
            if afterstate != board:
                afterstates_rewards.append((afterstate, reward))

        if not afterstates_rewards:
            leaf_info.append((0, []))
            continue

        for afterstate, _ in afterstates_rewards:
            all_afterstates.append(afterstate)
        leaf_info.append((
            len(afterstates_rewards),
            [r for _, r in afterstates_rewards],
        ))

    # Single batched forward pass for all afterstates
    if all_afterstates:
        all_values = value_fn(all_afterstates)
    else:
        all_values = torch.tensor([])

    # Unpack: max_a [reward_a + V(afterstate_a)] per leaf
    results = torch.zeros(len(leaf_boards))
    idx = 0
    for i, (count, rewards) in enumerate(leaf_info):
        if count == 0:
            results[i] = 0.0  # terminal
        else:
            values = all_values[idx : idx + count]
            reward_t = torch.tensor(rewards, dtype=values.dtype, device=values.device)
            results[i] = (reward_t + values).max().item()
            idx += count
    return results


def _evaluate_node(
    node: _MaxNode | _LeafNode | _ChanceNode,
    leaf_values: torch.Tensor,
) -> float:
    if isinstance(node, _LeafNode):
        return leaf_values[node.index].item()
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


def expectimax_action(board: Board, value_fn: ValueFunction, depth: int = 1) -> Action:
    """
    Select the best action using expectimax search.

    Builds a game tree of alternating max (player) and chance (random tile spawn) nodes
    up to the given depth, collects all leaf boards, evaluates them in a single batched
    forward pass via `value_fn`, then propagates expected values back up the tree to
    pick the highest-value root action.

    Parameters
    ----------
    board : Board
        Current board state (pre-move, post-spawn).
    value_fn : ValueFunction
        Batched evaluation: list[Board] -> Tensor of V(afterstate) values.
        (For example, use `make_afterstate_value_fn` or `make_dqn_value_fn` to wrap
        a `ConvNetwork`.
    depth : int
        Number of max-chance plies to expand. depth=0 means leaves are
        evaluated directly (greedy one-step lookahead via `_evaluate_leaves`).
    """
    # build tree for each root action, evaluate independently
    best_action = Action.UP
    best_value = float("-inf")

    # collect all root actions' chance nodes, then batch-evaluate all leaves
    leaves: list[Board] = []
    root_actions: list[tuple[Action, _ChanceNode]] = []

    for action in Action:
        afterstate, reward = apply_action(board, action)
        if afterstate == board:
            continue
        chance = _build_chance_node(afterstate, reward, depth, leaves)
        root_actions.append((action, chance))

    if not root_actions:
        return Action.UP  # terminal, doesn't matter

    # batch evaluate all leaves
    leaf_values = _evaluate_leaves(leaves, value_fn)

    # backpropagate to find best action
    for action, chance in root_actions:
        value = _evaluate_node(chance, leaf_values)
        if value > best_value:
            best_value = value
            best_action = action

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
        encoded = torch.stack([encode_state(b) for b in boards])
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
        encoded = torch.stack([encode_state(b) for b in boards])
        with torch.no_grad():
            q_all = model(encoded.to(device))  # (N, 4)
            return q_all.max(dim=1).values.cpu()

    return value_fn

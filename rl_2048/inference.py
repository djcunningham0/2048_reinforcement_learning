"""Load trained models and select actions at inference time."""

import torch

from rl_2048.game import Action, Board, apply_action, downgrade_board, encode_state
from rl_2048.network import ConvNetwork
from rl_2048.ntuple.network import NTupleNetwork

MODEL_TYPES = ("dqn", "afterstate", "ntuple")


def load_model(
    checkpoint_path: str,
    device: str,
    model_type: str,
) -> ConvNetwork | NTupleNetwork:
    """Load a trained model from a checkpoint file."""
    if model_type == "ntuple":
        return NTupleNetwork.load(checkpoint_path)
    output_dim = 4 if model_type == "dqn" else 1
    model = ConvNetwork(output_dim=output_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["online_net"])
    model.eval()
    return model.to(torch.device(device))


def select_action_dqn(
    model: ConvNetwork,
    board: Board,
    valid_actions: list[Action],
    device: str,
    downgrade_threshold: int | None = None,
) -> Action:
    """Select an action using a DQN model (greedy)."""
    if downgrade_threshold is not None:
        board = downgrade_board(board, downgrade_threshold)
    state = encode_state(board)
    with torch.no_grad():
        q_values = model(state.unsqueeze(0).to(device)).squeeze(0)
        mask = torch.full_like(q_values, float("-inf"))
        for a in valid_actions:
            mask[a] = 0.0
        q_values = q_values + mask
        return Action(q_values.argmax().item())


def select_action_afterstate(
    model: ConvNetwork,
    board: Board,
    valid_actions: list[Action],
    device: str,
    downgrade_threshold: int | None = None,
) -> Action:
    """Select an action using an afterstate value model (greedy)."""
    afterstates = [apply_action(board, a) for a in valid_actions]
    eval_boards = [s for s, _ in afterstates]
    if downgrade_threshold is not None:
        eval_boards = [downgrade_board(s, downgrade_threshold) for s in eval_boards]
    encoded = torch.stack([encode_state(s) for s in eval_boards])
    with torch.no_grad():
        values = model(encoded.to(device)).squeeze(-1)
    rewards = torch.tensor([r for _, r in afterstates], device=device)
    action_values = rewards + values
    return valid_actions[action_values.argmax().item()]


def select_action_ntuple(
    model: NTupleNetwork,
    board: Board,
    valid_actions: list[Action],
    device: str,  # unused but kept for consistent signature
    downgrade_threshold: int | None = None,
) -> Action:
    """Select an action using an N-tuple network (greedy)."""
    best_action = valid_actions[0]
    best_value = float("-inf")
    for a in valid_actions:
        afterstate, reward = apply_action(board, a)
        eval_board = afterstate
        if downgrade_threshold is not None:
            eval_board = downgrade_board(eval_board, downgrade_threshold)
        value = reward + model.evaluate(tuple(eval_board))
        if value > best_value:
            best_value = value
            best_action = a
    return best_action

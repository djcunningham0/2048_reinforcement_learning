"""Load trained models and select actions at inference time."""

import torch

from rl_2048.game import Action, Board, apply_action, encode_state
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
) -> Action:
    """Select an action using a DQN model (greedy)."""
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
) -> Action:
    """Select an action using an afterstate value model (greedy)."""
    afterstates = [apply_action(board, a) for a in valid_actions]
    encoded = torch.stack([encode_state(s) for s, _ in afterstates])
    with torch.no_grad():
        values = model(encoded.to(device)).squeeze(-1)
    rewards = torch.tensor([r for _, r in afterstates], device=device)
    action_values = rewards + values
    return valid_actions[action_values.argmax().item()]


def select_action_ntuple(
    model: NTupleNetwork,
    board: Board,
    valid_actions: list[Action],
    _device: str,
) -> Action:
    """Select an action using an N-tuple network (greedy)."""
    best_action = valid_actions[0]
    best_value = float("-inf")
    for a in valid_actions:
        afterstate, reward = apply_action(board, a)
        value = reward + model.evaluate(tuple(afterstate))
        if value > best_value:
            best_value = value
            best_action = a
    return best_action

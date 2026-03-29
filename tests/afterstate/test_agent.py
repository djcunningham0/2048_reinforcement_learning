"""Tests for afterstate agent."""

import torch

from rl_2048.afterstate.agent import AfterstateAgent
from rl_2048.afterstate.config import AfterstateConfig
from rl_2048.afterstate.replay_buffer import (
    AfterstateReplayBuffer,
    AfterstateTransition,
    compute_all_afterstates,
)
from rl_2048.game import Action, apply_action, make_board


def _make_board_with_valid_actions():
    return make_board([
        [2, 4, 0, 0],
        [8, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])


class TestAfterstateAgent:
    def test_returns_consistent_action(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)
        board = _make_board_with_valid_actions()
        info = compute_all_afterstates(board)
        actions = {agent.select_action(info) for _ in range(20)}
        assert len(actions) == 1

    def test_invalid_actions_never_selected(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)
        board = _make_board_with_valid_actions()
        info = compute_all_afterstates(board)
        valid_set = set(info.valid_mask.nonzero(as_tuple=False).view(-1).tolist())
        for _ in range(50):
            action = agent.select_action(info)
            assert action.value in valid_set

    def test_select_action_returns_valid_action(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)
        board = _make_board_with_valid_actions()
        info = compute_all_afterstates(board)
        action = agent.select_action(info)
        expected_afterstate, _ = apply_action(board, action)
        assert expected_afterstate != board  # action is valid

    def test_target_sync_produces_identical_outputs(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)

        board = _make_board_with_valid_actions()
        info = compute_all_afterstates(board)
        buf = AfterstateReplayBuffer(100)
        for _ in range(16):
            afterstate, _ = apply_action(board, Action.LEFT)
            buf.push(
                AfterstateTransition(
                    afterstate=info.encoded[Action.LEFT],
                    next_afterstates=info.encoded,
                    next_rewards=info.rewards,
                    next_valid_mask=info.valid_mask,
                    done=False,
                )
            )
        agent.train_step(buf.sample(16))

        x = torch.randn(1, 16, 4, 4, device=agent.device)
        online_out = agent.online_net(x).clone()

        # They should differ after training
        before_sync = agent.target_net(x).clone()
        assert not torch.allclose(before_sync, online_out, atol=1e-6)

        agent.sync_target_network()
        after_sync = agent.target_net(x)
        assert torch.allclose(after_sync, online_out, atol=1e-6)

"""Tests for afterstate agent."""

import torch

from rl_2048.afterstate.agent import AfterstateAgent
from rl_2048.afterstate.config import AfterstateConfig
from rl_2048.afterstate.replay_buffer import (
    AfterstateReplayBuffer,
    AfterstateTransition,
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
    def test_epsilon_zero_returns_consistent_action(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)
        board = _make_board_with_valid_actions()
        valid = [Action.LEFT, Action.DOWN]
        actions = {agent.select_action(board, valid, epsilon=0.0)[0] for _ in range(20)}
        assert len(actions) == 1

    def test_epsilon_one_explores(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)
        board = _make_board_with_valid_actions()
        valid = list(Action)
        # Filter to actual valid actions
        valid = [a for a in Action if apply_action(board, a)[0] != board]
        actions = {
            agent.select_action(board, valid, epsilon=1.0)[0] for _ in range(100)
        }
        assert len(actions) >= 2

    def test_invalid_actions_never_selected(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)
        board = _make_board_with_valid_actions()
        valid = [Action.LEFT, Action.DOWN]
        for _ in range(50):
            action, _ = agent.select_action(board, valid, epsilon=0.5)
            assert action in valid

    def test_select_action_returns_correct_afterstate(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)
        board = _make_board_with_valid_actions()
        valid = [Action.LEFT, Action.DOWN]
        action, afterstate = agent.select_action(board, valid, epsilon=0.0)
        expected_afterstate, _ = apply_action(board, action)
        assert afterstate == expected_afterstate

    def test_target_sync_produces_identical_outputs(self):
        config = AfterstateConfig(device="cpu")
        agent = AfterstateAgent(config)

        board = _make_board_with_valid_actions()
        buf = AfterstateReplayBuffer(100)
        for _ in range(16):
            afterstate, _ = apply_action(board, Action.LEFT)
            buf.push(
                AfterstateTransition(
                    afterstate=afterstate, next_state=board, done=False
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

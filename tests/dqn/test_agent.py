"""Tests for DQN agent."""

import torch

import pytest

from rl_2048.game import Action
from rl_2048.dqn.agent import DQNAgent
from rl_2048.dqn.config import DQNConfig
from rl_2048.dqn.replay_buffer import BatchedTransitions, Transition


def _batch_transitions(transitions: list[Transition]) -> BatchedTransitions:
    return BatchedTransitions(
        states=torch.stack([t.state for t in transitions]),
        actions=torch.tensor([t.action for t in transitions], dtype=torch.long),
        rewards=torch.tensor([t.reward for t in transitions], dtype=torch.float32),
        next_states=torch.stack([t.next_state for t in transitions]),
        dones=torch.tensor([t.done for t in transitions], dtype=torch.float32),
        valid_masks=torch.stack([t.valid_mask for t in transitions]),
    )


def _make_transition(action: Action = Action.UP, reward: float = 1.0) -> Transition:
    return Transition(
        state=torch.randn(16, 4, 4),
        action=action,
        reward=reward,
        next_state=torch.randn(16, 4, 4),
        done=False,
        valid_mask=torch.ones(4, dtype=torch.bool),
    )


class TestDQNAgent:
    def test_epsilon_zero_returns_argmax(self):
        config = DQNConfig()
        agent = DQNAgent(config)
        state = torch.randn(16, 4, 4)
        # With epsilon=0, should always return the same action
        actions = {
            agent.select_action(state, list(Action), epsilon=0.0) for _ in range(20)
        }
        assert len(actions) == 1

    def test_epsilon_one_explores(self):
        config = DQNConfig()
        agent = DQNAgent(config)
        state = torch.randn(16, 4, 4)
        actions = {
            agent.select_action(state, list(Action), epsilon=1.0) for _ in range(100)
        }
        # Should pick at least 2 different actions with high probability
        assert len(actions) >= 2

    def test_invalid_actions_never_selected(self):
        config = DQNConfig()
        agent = DQNAgent(config)
        state = torch.randn(16, 4, 4)
        valid = [Action.UP, Action.DOWN]
        for _ in range(50):
            action = agent.select_action(state, valid, epsilon=0.5)
            assert action in valid

    def test_loss_decreases_on_repeated_batch(self):
        config = DQNConfig(lr=1e-3)
        agent = DQNAgent(config)
        transitions = [_make_transition(action=Action(i % 4)) for i in range(32)]
        batch = _batch_transitions(transitions)

        losses = []
        for _ in range(20):
            loss = agent.train_step(batch)
            losses.append(loss)

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"  # fmt: skip

    def test_target_sync_produces_identical_outputs(self):
        config = DQNConfig()
        agent = DQNAgent(config)

        # Train online net so it diverges from target
        batch = _batch_transitions([_make_transition() for _ in range(16)])
        agent.train_step(batch)

        x = torch.randn(1, 16, 4, 4, device=agent.device)
        before_sync = agent.target_net(x).clone()
        online_out = agent.online_net(x).clone()

        # They should differ after training
        assert not torch.allclose(before_sync, online_out, atol=1e-6)

        # After syncing, they should be the same
        agent.sync_target_network()
        after_sync = agent.target_net(x)
        assert torch.allclose(after_sync, online_out, atol=1e-6)


class TestDQNConfig:
    def test_epsilon_schedule(self):
        config = DQNConfig(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=100)
        assert config.epsilon_at(0) == 1.0
        assert config.epsilon_at(100) == pytest.approx(0.01)
        assert config.epsilon_at(50) == pytest.approx(0.505)
        assert config.epsilon_at(200) == pytest.approx(0.01)

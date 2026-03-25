"""Tests for replay buffer."""

import torch

from rl_2048.dqn.replay_buffer import BatchedTransitions, ReplayBuffer, Transition
from rl_2048.game import Action


def _make_transition(action: Action = Action.UP) -> Transition:
    return Transition(
        state=torch.zeros(18, 4, 4),
        action=action,
        reward=1.0,
        next_state=torch.zeros(18, 4, 4),
        done=False,
        valid_mask=torch.ones(4, dtype=torch.bool),
    )


class TestReplayBuffer:
    def test_len(self):
        buf = ReplayBuffer(100)
        assert len(buf) == 0
        buf.push(_make_transition())
        assert len(buf) == 1

    def test_capacity_eviction(self):
        buf = ReplayBuffer(3)
        for i in range(5):
            buf.push(_make_transition(action=i))
        assert len(buf) == 3
        # First two should have been evicted
        actions = {t.action for t in buf._buffer}
        assert actions == {2, 3, 4}

    def test_sample_returns_batched_transitions(self):
        buf = ReplayBuffer(100)
        for i in range(10):
            buf.push(_make_transition(action=i))
        batch = buf.sample(5)
        assert isinstance(batch, BatchedTransitions)
        assert batch.states.shape == (5, 18, 4, 4)
        assert batch.actions.shape == (5,)
        assert batch.rewards.shape == (5,)
        assert batch.next_states.shape == (5, 18, 4, 4)
        assert batch.dones.shape == (5,)
        assert batch.valid_masks.shape == (5, 4)

    def test_sample_from_underfilled(self):
        buf = ReplayBuffer(100)
        for i in range(3):
            buf.push(_make_transition(action=i))
        batch = buf.sample(3)
        assert batch.states.shape[0] == 3

    def test_sample_raises_on_too_large(self):
        buf = ReplayBuffer(100)
        buf.push(_make_transition())
        try:
            buf.sample(5)
            assert False, "Should have raised"
        except ValueError:
            pass

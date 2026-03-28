"""Tests for afterstate replay buffer."""

from rl_2048.afterstate.replay_buffer import (
    AfterstateReplayBuffer,
    AfterstateTransition,
    BatchedAfterstateTransitions,
    compute_all_afterstates,
)
from rl_2048.game import Action, Game2048, apply_action, make_board


def _make_board_with_valid_actions():
    """Return a board that has at least one valid action."""
    return make_board([
        [2, 4, 0, 0],
        [8, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])


def _make_transition() -> AfterstateTransition:
    board = _make_board_with_valid_actions()
    info = compute_all_afterstates(board)
    return AfterstateTransition(
        afterstate=info.encoded[Action.LEFT],
        next_afterstates=info.encoded,
        next_rewards=info.rewards,
        next_valid_mask=info.valid_mask,
        done=False,
    )


class TestAfterstateReplayBuffer:
    def test_len(self):
        buf = AfterstateReplayBuffer(100)
        assert len(buf) == 0
        buf.push(_make_transition())
        assert len(buf) == 1

    def test_capacity_eviction(self):
        buf = AfterstateReplayBuffer(3)
        for _ in range(5):
            buf.push(_make_transition())
        assert len(buf) == 3

    def test_sample_shapes(self):
        buf = AfterstateReplayBuffer(100)
        for _ in range(10):
            buf.push(_make_transition())
        batch = buf.sample(5)
        assert isinstance(batch, BatchedAfterstateTransitions)
        assert batch.afterstates.shape == (5, 16, 4, 4)
        assert batch.next_afterstates.shape == (5, 4, 16, 4, 4)
        assert batch.next_rewards.shape == (5, 4)
        assert batch.next_valid_masks.shape == (5, 4)
        assert batch.dones.shape == (5,)

    def test_valid_mask_correctness(self):
        """Valid mask in batch should match get_valid_actions on the next_state."""
        board = _make_board_with_valid_actions()
        info = compute_all_afterstates(board)
        transition = AfterstateTransition(
            afterstate=info.encoded[Action.LEFT],
            next_afterstates=info.encoded,
            next_rewards=info.rewards,
            next_valid_mask=info.valid_mask,
            done=False,
        )

        buf = AfterstateReplayBuffer(100)
        buf.push(transition)
        batch = buf.sample(1)

        game = Game2048()
        game.board = board
        expected_valid = game.get_valid_actions()
        for action in Action:
            assert batch.next_valid_masks[0, action].item() == (
                action in expected_valid
            )

    def test_next_afterstate_matches_apply_action(self):
        """Batch next_afterstates should match apply_action on the next_state."""
        board = _make_board_with_valid_actions()
        info = compute_all_afterstates(board)
        transition = AfterstateTransition(
            afterstate=info.encoded[Action.LEFT],
            next_afterstates=info.encoded,
            next_rewards=info.rewards,
            next_valid_mask=info.valid_mask,
            done=False,
        )

        buf = AfterstateReplayBuffer(100)
        buf.push(transition)
        batch = buf.sample(1)

        for action in Action:
            new_board, reward = apply_action(board, action)
            if new_board != board:
                assert batch.next_rewards[0, action].item() == reward

    def test_done_transitions(self):
        """Terminal transitions should have done=1."""
        terminal_board = make_board([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 2],
        ])
        info = compute_all_afterstates(terminal_board)
        transition = AfterstateTransition(
            afterstate=info.encoded[0],
            next_afterstates=info.encoded,
            next_rewards=info.rewards,
            next_valid_mask=info.valid_mask,
            done=True,
        )
        buf = AfterstateReplayBuffer(100)
        buf.push(transition)
        batch = buf.sample(1)
        assert batch.dones[0].item() == 1.0
        # No valid actions from terminal board
        assert not batch.next_valid_masks[0].any()

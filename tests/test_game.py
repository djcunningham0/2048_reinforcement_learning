import random

import torch

from rl_2048.game import (
    Action,
    Game2048,
    _slide_row_left,
    apply_action,
    encode_state,
    make_board,
)


class TestSlideRowLeft:
    def test_no_merge(self):
        assert _slide_row_left([0, 2, 0, 4]) == ([2, 4, 0, 0], 0)

    def test_simple_merge(self):
        assert _slide_row_left([2, 2, 0, 0]) == ([4, 0, 0, 0], 4)

    def test_double_merge(self):
        assert _slide_row_left([2, 2, 4, 4]) == ([4, 8, 0, 0], 12)

    def test_no_cascade(self):
        # [2, 2, 4, 0] -> [4, 4, 0, 0], NOT [8, 0, 0, 0]
        assert _slide_row_left([2, 2, 4, 0]) == ([4, 4, 0, 0], 4)

    def test_triple(self):
        # [2, 2, 2, 0] -> [4, 2, 0, 0] (leftmost pair merges)
        assert _slide_row_left([2, 2, 2, 0]) == ([4, 2, 0, 0], 4)

    def test_all_same(self):
        assert _slide_row_left([2, 2, 2, 2]) == ([4, 4, 0, 0], 8)

    def test_already_packed(self):
        assert _slide_row_left([2, 4, 8, 16]) == ([2, 4, 8, 16], 0)

    def test_empty_row(self):
        assert _slide_row_left([0, 0, 0, 0]) == ([0, 0, 0, 0], 0)


class TestGame2048:
    def test_reset_spawns_two_tiles(self):
        game = Game2048()
        game.reset()
        non_zero = sum(1 for v in game.board if v != 0)
        assert non_zero == 2

    def test_reset_clears_score(self):
        game = Game2048()
        game.reset()
        game.score = 100
        game.reset()
        assert game.score == 0

    def test_step_spawns_one_tile_on_valid_move(self):
        game = Game2048()
        game.board = make_board([
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        # Slide left — tile moves from (0,3) to (0,0), one new tile spawned
        game.step(Action.LEFT)
        non_zero = sum(1 for v in game.board if v != 0)
        assert non_zero == 2  # moved tile + 1 spawn

    def test_step_score_accumulates(self):
        game = Game2048()
        game.board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = game.step(Action.LEFT)
        assert reward == 4
        assert game.score == 4

    def test_slide_directions(self):
        """Test all four directions produce correct results."""
        game = Game2048()
        random.seed(42)

        # Test left
        game.board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = game.step(Action.LEFT)
        assert game.board[0] == 4  # (0, 0)
        assert reward == 4

        # Test right
        game.board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = game.step(Action.RIGHT)
        assert game.board[3] == 4  # (0, 3)
        assert reward == 4

        # Test up
        game.board = make_board([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = game.step(Action.UP)
        assert game.board[0] == 4  # (0, 0)
        assert reward == 4

        # Test down
        game.board = make_board([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = game.step(Action.DOWN)
        assert game.board[12] == 4  # (3, 0)
        assert reward == 4

    def test_game_over_full_board_no_merges(self):
        game = Game2048()
        game.board = make_board([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 2],
        ])
        game.step(Action.LEFT)  # no-op move
        assert len(game.get_valid_actions()) == 0

    def test_not_game_over_full_board_with_merge(self):
        game = Game2048()
        game.board = make_board([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 4],  # last two can merge
        ])
        game.step(Action.UP)  # no-op move
        assert len(game.get_valid_actions()) > 0

    def test_not_game_over_empty_cells(self):
        game = Game2048()
        game.board = make_board([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 0],
        ])
        game.step(Action.UP)  # no-op move
        assert len(game.get_valid_actions()) > 0

    def test_get_valid_actions(self):
        game = Game2048()
        # Tile at (0,1) — can slide left, right, or down, but not up (already top row)
        game.board = make_board([
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        valid = game.get_valid_actions()
        assert Action.LEFT in valid
        assert Action.RIGHT in valid
        assert Action.DOWN in valid
        assert Action.UP not in valid  # already at top

    def test_get_valid_actions_packed_corner(self):
        """A single tile in top-left corner can only go right or down."""
        game = Game2048()
        game.board = make_board([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        valid = game.get_valid_actions()
        assert Action.LEFT not in valid
        assert Action.UP not in valid
        assert Action.RIGHT in valid
        assert Action.DOWN in valid

    def test_invalid_move_no_spawn(self):
        """A move that doesn't change the board should not spawn a tile."""
        game = Game2048()
        game.board = make_board([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        board_before = game.board
        # Slide left — tile already at left edge, no change
        reward = game.step(Action.LEFT)
        assert reward == 0
        # Board should be unchanged (no spawn since move was invalid)
        assert game.board == board_before


class TestEncodeState:
    def test_shape(self):
        board = (0,) * 16
        t = encode_state(board)
        assert t.shape == (16, 4, 4)

    def test_one_hot(self):
        """Each cell should have exactly one channel active."""
        board = make_board([
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [0, 0, 0, 0],
        ])
        t = encode_state(board)
        sums = t.sum(dim=0)
        assert torch.allclose(sums, torch.ones(4, 4))

    def test_empty_cell_channel_0(self):
        board = (0,) * 16
        t = encode_state(board)
        assert t[0].sum() == 16  # all cells on channel 0
        assert t[1:].sum() == 0  # nothing on other channels

    def test_specific_tile(self):
        board = list((0,) * 16)
        board[1 * 4 + 2] = 8  # 8 = 2^3 -> channel 3
        t = encode_state(tuple(board))
        assert t[3, 1, 2] == 1.0
        assert t[0, 1, 2] == 0.0  # not empty


class TestApplyAction:
    def test_slide_left(self):
        board = make_board([
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        new_board, score = apply_action(board, Action.LEFT)
        assert new_board[0] == 2  # tile at (0,0)
        assert score == 0

    def test_merge(self):
        board = make_board([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        new_board, score = apply_action(board, Action.LEFT)
        assert new_board[0] == 4
        assert score == 4

    def test_matches_game_method(self):
        """Standalone apply_action should match Game2048.step (minus spawn)."""
        board = make_board([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 2, 4, 4],
            [0, 0, 0, 2],
        ])
        game = Game2048()
        game.board = board
        # Compare the deterministic part (before spawn)
        new_board, score = apply_action(board, Action.LEFT)
        expected_board = make_board([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [4, 8, 0, 0],
            [2, 0, 0, 0],
        ])
        assert new_board == expected_board
        assert score == 12


class TestGetValidActions:
    def test_corner_tile(self):
        game = Game2048()
        game.board = make_board([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        valid = game.get_valid_actions()
        assert Action.LEFT not in valid
        assert Action.UP not in valid
        assert Action.RIGHT in valid
        assert Action.DOWN in valid

    def test_no_valid_actions(self):
        game = Game2048()
        game.board = make_board([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 2],
        ])
        assert game.get_valid_actions() == []

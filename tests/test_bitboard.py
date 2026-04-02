import random

import torch

from rl_2048 import game
from rl_2048.bitboard import (
    BitBoardGame2048,
    _MOVE_LEFT,
    _MOVE_RIGHT,
    _ROW_SCORE,
    _reverse_row,
    _transpose,
    apply_action,
    bitboard_to_board,
    board_to_bitboard,
    encode_state,
    get_valid_actions,
    make_bitboard,
)
from rl_2048.game import Action


class TestReversRow:
    def test_identity(self):
        # Palindrome row: [1, 2, 2, 1]
        row = 0x1221
        assert _reverse_row(row) == 0x1221

    def test_simple(self):
        # [1, 0, 0, 0] -> [0, 0, 0, 1]
        assert _reverse_row(0x1000) == 0x0001

    def test_double_reverse(self):
        for row in [0x0000, 0x1234, 0xABCD, 0xFFFF]:
            assert _reverse_row(_reverse_row(row)) == row


class TestLookupTables:
    def test_empty_row(self):
        assert _MOVE_LEFT[0x0000] == 0x0000
        assert _ROW_SCORE[0x0000] == 0

    def test_simple_slide(self):
        # [0, 1, 0, 0] -> [1, 0, 0, 0], no merge
        assert _MOVE_LEFT[0x0100] == 0x1000
        assert _ROW_SCORE[0x0100] == 0

    def test_simple_merge(self):
        # [1, 1, 0, 0] -> [2, 0, 0, 0], score = 2^2 = 4
        assert _MOVE_LEFT[0x1100] == 0x2000
        assert _ROW_SCORE[0x1100] == 4

    def test_double_merge(self):
        # [1, 1, 2, 2] -> [2, 3, 0, 0], score = 4 + 8 = 12
        assert _MOVE_LEFT[0x1122] == 0x2300
        assert _ROW_SCORE[0x1122] == 12

    def test_no_cascade(self):
        # [1, 1, 2, 0] -> [2, 2, 0, 0], score = 4
        assert _MOVE_LEFT[0x1120] == 0x2200
        assert _ROW_SCORE[0x1120] == 4

    def test_triple(self):
        # [1, 1, 1, 0] -> [2, 1, 0, 0], score = 4
        assert _MOVE_LEFT[0x1110] == 0x2100
        assert _ROW_SCORE[0x1110] == 4

    def test_all_same(self):
        # [1, 1, 1, 1] -> [2, 2, 0, 0], score = 8
        assert _MOVE_LEFT[0x1111] == 0x2200
        assert _ROW_SCORE[0x1111] == 8

    def test_already_packed(self):
        # [1, 2, 3, 4] -> [1, 2, 3, 4], no merge
        assert _MOVE_LEFT[0x1234] == 0x1234
        assert _ROW_SCORE[0x1234] == 0

    def test_move_right_simple_merge(self):
        # [0, 0, 1, 1] -> [0, 0, 0, 2], score = 4
        assert _MOVE_RIGHT[0x0011] == 0x0002

    def test_move_right_slide(self):
        # [1, 0, 0, 0] -> [0, 0, 0, 1]
        assert _MOVE_RIGHT[0x1000] == 0x0001


class TestTranspose:
    def test_diagonal_unchanged(self):
        # Board with tiles only on the diagonal
        bb = make_bitboard([
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 16],
        ])
        assert _transpose(bb) == bb

    def test_known_swap(self):
        # Tile at (0, 1) should move to (1, 0)
        bb = make_bitboard([
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        expected = make_bitboard([
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        assert _transpose(bb) == expected

    def test_double_transpose_identity(self):
        bb = make_bitboard([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 2],
        ])
        assert _transpose(_transpose(bb)) == bb

    def test_empty_board(self):
        assert _transpose(0) == 0


class TestConversion:
    def test_round_trip(self):
        board = game.make_board([
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [0, 0, 0, 0],
        ])
        assert bitboard_to_board(board_to_bitboard(board)) == board

    def test_round_trip_empty(self):
        board = (0,) * 16
        assert bitboard_to_board(board_to_bitboard(board)) == board

    def test_round_trip_full(self):
        board = game.make_board([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 2],
        ])
        assert bitboard_to_board(board_to_bitboard(board)) == board

    def test_make_bitboard_matches(self):
        rows = [
            [0, 2, 4, 0],
            [8, 0, 0, 2],
            [0, 0, 0, 0],
            [4, 2, 0, 16],
        ]
        bb = make_bitboard(rows)
        board = game.make_board(rows)
        assert bb == board_to_bitboard(board)
        assert bitboard_to_board(bb) == board


class TestApplyAction:
    def test_slide_left(self):
        bb = make_bitboard([
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        new_bb, score = apply_action(bb, Action.LEFT)
        expected = make_bitboard([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        assert new_bb == expected
        assert score == 0

    def test_merge(self):
        bb = make_bitboard([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        new_bb, score = apply_action(bb, Action.LEFT)
        expected = make_bitboard([
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        assert new_bb == expected
        assert score == 4

    def test_matches_game_py(self):
        """Bitboard apply_action matches game.apply_action."""
        rows = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 2, 4, 4],
            [0, 0, 0, 2],
        ]
        board = game.make_board(rows)
        bb = make_bitboard(rows)

        new_board, game_score = game.apply_action(board, Action.LEFT)
        new_bb, bb_score = apply_action(bb, Action.LEFT)

        assert bitboard_to_board(new_bb) == new_board
        assert bb_score == game_score

    def test_all_directions(self):
        rows = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        board = game.make_board(rows)
        bb = make_bitboard(rows)

        for action in Action:
            new_board, game_score = game.apply_action(board, action)
            new_bb, bb_score = apply_action(bb, action)
            assert bitboard_to_board(new_bb) == new_board, f"Mismatch for {action}"
            assert bb_score == game_score, f"Score mismatch for {action}"

    def test_up_move(self):
        bb = make_bitboard([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        new_bb, score = apply_action(bb, Action.UP)
        expected = make_bitboard([
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        assert new_bb == expected
        assert score == 4

    def test_down_move(self):
        bb = make_bitboard([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        new_bb, score = apply_action(bb, Action.DOWN)
        expected = make_bitboard([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [4, 0, 0, 0],
        ])
        assert new_bb == expected
        assert score == 4

    def test_right_move(self):
        bb = make_bitboard([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        new_bb, score = apply_action(bb, Action.RIGHT)
        expected = make_bitboard([
            [0, 0, 0, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        assert new_bb == expected
        assert score == 4


class TestGetValidActions:
    def test_corner_tile(self):
        bb = make_bitboard([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        valid = get_valid_actions(bb)
        assert Action.LEFT not in valid
        assert Action.UP not in valid
        assert Action.RIGHT in valid
        assert Action.DOWN in valid

    def test_no_valid_actions(self):
        bb = make_bitboard([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 2],
        ])
        assert get_valid_actions(bb) == []

    def test_full_board_with_merge(self):
        bb = make_bitboard([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 4],
        ])
        assert len(get_valid_actions(bb)) > 0


class TestBitBoardGame2048:
    def test_reset_spawns_two_tiles(self):
        g = BitBoardGame2048()
        g.reset()
        board = bitboard_to_board(g.board)
        non_zero = sum(1 for v in board if v != 0)
        assert non_zero == 2

    def test_reset_clears_score(self):
        g = BitBoardGame2048()
        g.reset()
        g.score = 100
        g.reset()
        assert g.score == 0

    def test_step_spawns_one_tile_on_valid_move(self):
        g = BitBoardGame2048()
        g.board = make_bitboard([
            [0, 0, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        g.step(Action.LEFT)
        board = bitboard_to_board(g.board)
        non_zero = sum(1 for v in board if v != 0)
        assert non_zero == 2

    def test_step_score_accumulates(self):
        g = BitBoardGame2048()
        g.board = make_bitboard([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = g.step(Action.LEFT)
        assert reward == 4
        assert g.score == 4

    def test_invalid_move_no_spawn(self):
        g = BitBoardGame2048()
        g.board = make_bitboard([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        board_before = g.board
        reward = g.step(Action.LEFT)
        assert reward == 0
        assert g.board == board_before

    def test_slide_directions(self):
        g = BitBoardGame2048()
        random.seed(42)

        # Left
        g.board = make_bitboard([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = g.step(Action.LEFT)
        assert bitboard_to_board(g.board)[0] == 4
        assert reward == 4

        # Right
        g.board = make_bitboard([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = g.step(Action.RIGHT)
        assert bitboard_to_board(g.board)[3] == 4
        assert reward == 4

        # Up
        g.board = make_bitboard([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = g.step(Action.UP)
        assert bitboard_to_board(g.board)[0] == 4
        assert reward == 4

        # Down
        g.board = make_bitboard([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        reward = g.step(Action.DOWN)
        assert bitboard_to_board(g.board)[12] == 4
        assert reward == 4

    def test_game_over(self):
        g = BitBoardGame2048()
        g.board = make_bitboard([
            [2, 4, 8, 16],
            [16, 8, 4, 2],
            [2, 4, 8, 16],
            [16, 8, 4, 2],
        ])
        assert g.get_valid_actions() == []


class TestEncodeState:
    def test_shape(self):
        t = encode_state(0)
        assert t.shape == (16, 4, 4)

    def test_one_hot(self):
        bb = make_bitboard([
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [0, 0, 0, 0],
        ])
        t = encode_state(bb)
        sums = t.sum(dim=0)
        assert torch.allclose(sums, torch.ones(4, 4))

    def test_empty_cell_channel_0(self):
        t = encode_state(0)
        assert t[0].sum() == 16
        assert t[1:].sum() == 0

    def test_specific_tile(self):
        bb = make_bitboard([
            [0, 0, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        t = encode_state(bb)
        assert t[3, 1, 2] == 1.0  # 8 = 2^3 -> channel 3
        assert t[0, 1, 2] == 0.0


class TestCrossValidation:
    """Verify bitboard matches game.py for many random boards."""

    def _random_boards(self, n: int = 200) -> list[game.Board]:
        """Generate boards by playing random games."""
        boards: list[game.Board] = []
        g = game.Game2048()
        for _ in range(n):
            g.reset()
            for _ in range(random.randint(0, 50)):
                valid = g.get_valid_actions()
                if not valid:
                    break
                g.step(random.choice(valid))
            boards.append(g.board)
        return boards

    def test_apply_action_matches(self):
        random.seed(123)
        for board in self._random_boards():
            bb = board_to_bitboard(board)
            for action in Action:
                new_board, game_score = game.apply_action(board, action)
                new_bb, bb_score = apply_action(bb, action)
                assert bitboard_to_board(new_bb) == new_board
                assert bb_score == game_score

    def test_get_valid_actions_matches(self):
        random.seed(456)
        for board in self._random_boards():
            bb = board_to_bitboard(board)
            game_valid = [a for a in Action if game.apply_action(board, a)[0] != board]
            bb_valid = get_valid_actions(bb)
            assert bb_valid == game_valid

    def test_encode_state_matches(self):
        random.seed(789)
        for board in self._random_boards(50):
            bb = board_to_bitboard(board)
            game_tensor = game.encode_state(board)
            bb_tensor = encode_state(bb)
            assert torch.equal(game_tensor, bb_tensor)

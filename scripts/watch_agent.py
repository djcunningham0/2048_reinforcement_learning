"""Watch a trained agent play 2048 in the terminal."""

import argparse
import curses
import time

from rl_2048.game import Action, Game2048
from rl_2048.expectimax import (
    DepthSchedule,
    expectimax_action,
    make_afterstate_value_fn,
    make_dqn_value_fn,
    parse_depth,
)
from rl_2048.inference import (
    MODEL_TYPES,
    load_model,
    select_action_afterstate,
    select_action_dqn,
    select_action_ntuple,
)
from scripts.play import CELL_W, _tile_attr

DELAY_STEPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def draw(
    stdscr: curses.window,
    game: Game2048,
    action: Action | None,
    game_over: bool,
    model_type: str,
    depth: int | DepthSchedule = 0,
):
    stdscr.erase()
    label = model_type.upper()
    if isinstance(depth, DepthSchedule):
        depth_str = "  depth=adaptive"
    elif depth > 0:
        depth_str = f"  depth={depth}"
    else:
        depth_str = ""
    stdscr.addstr(0, 0, f"2048 {label}{depth_str} — q: quit  r: new game  ←/→: speed")

    stdscr.addstr(1, 0, f"Score: {game.score:<10}  Max tile: {max(game.board)}")

    if action is not None:
        stdscr.addstr(2, 0, f"Action: {action.name.lower()}")

    stdscr.addstr(4, 0, ("+" + "-" * CELL_W) * 4 + "+")
    for r in range(4):
        stdscr.addstr(5 + r * 2, 0, "|")
        for c in range(4):
            v = game.board[r * 4 + c]
            cell = str(v) if v else "."
            stdscr.addstr(f"{cell:^{CELL_W}}", _tile_attr(v))
            stdscr.addstr("|")
        stdscr.addstr(6 + r * 2, 0, ("+" + "-" * CELL_W) * 4 + "+")

    if game_over:
        stdscr.addstr(
            14, 0, f"GAME OVER — Score: {game.score}  Max tile: {max(game.board)}"
        )
        stdscr.addstr(15, 0, "Press r for new game, q to quit")

    stdscr.refresh()


def watch(
    stdscr: curses.window,
    model,
    device: str,
    delay_idx: int,
    model_type: str,
    depth: int | DepthSchedule = 0,
):
    use_search = isinstance(depth, DepthSchedule) or depth > 0
    if use_search:
        if model_type == "ntuple":
            value_fn = model.evaluate_batch
        elif model_type == "afterstate":
            value_fn = make_afterstate_value_fn(model, device)
        else:
            value_fn = make_dqn_value_fn(model, device)

        _depth = depth  # capture for closure

        def select_action(_model, board, _valid_actions, _device):
            return expectimax_action(board, value_fn, _depth)

    else:
        action_fns = {
            "dqn": select_action_dqn,
            "afterstate": select_action_afterstate,
            "ntuple": select_action_ntuple,
        }
        select_action = action_fns[model_type]

    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)
    curses.init_pair(5, curses.COLOR_RED, -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)
    curses.init_pair(7, curses.COLOR_BLUE, -1)
    stdscr.nodelay(True)

    game = Game2048()
    game.reset()
    action = None
    game_over = not game.get_valid_actions()

    while True:
        draw(stdscr, game, action, game_over, model_type, depth)

        if not game_over:
            delay = DELAY_STEPS[delay_idx]
            if delay > 0:
                time.sleep(delay)
            valid_actions = game.get_valid_actions()
            if valid_actions:
                action = select_action(model, game.board, valid_actions, device)
                game.step(action)
                game_over = not game.get_valid_actions()
            else:
                game_over = True

        key = stdscr.getch()
        if key == ord("q"):
            break
        elif key == ord("r"):
            game.reset()
            action = None
            game_over = False
        elif key == curses.KEY_RIGHT:
            delay_idx = min(len(DELAY_STEPS) - 1, delay_idx + 1)
        elif key == curses.KEY_LEFT:
            delay_idx = max(0, delay_idx - 1)

        if game_over:
            draw(stdscr, game, action, game_over, model_type, depth)
            stdscr.nodelay(False)
            while True:
                key = stdscr.getch()
                if key == ord("q"):
                    return
                elif key == ord("r"):
                    game.reset()
                    action = None
                    game_over = False
                    stdscr.nodelay(True)
                    break


def main():
    parser = argparse.ArgumentParser(description="Watch a trained agent play 2048")
    parser.add_argument(
        "checkpoint", type=str, help="Path to checkpoint file (.pt or .npz)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_TYPES,
        default="afterstate",
        help="Model type (default: afterstate)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds between moves (default: 0.0)",
    )
    parser.add_argument(
        "--depth",
        type=parse_depth,
        default=0,
        help=(
            "Search depth: an integer, 'adaptive', or a custom "
            "schedule like '10:1,6:2,0:3' (default: 0 = greedy)"
        ),
    )
    args = parser.parse_args()

    depth = args.depth

    delay_idx = min(
        range(len(DELAY_STEPS)), key=lambda i: abs(DELAY_STEPS[i] - args.delay)
    )
    model = load_model(args.checkpoint, args.device, args.model_type)
    curses.wrapper(
        lambda stdscr: watch(
            stdscr, model, args.device, delay_idx, args.model_type, depth
        )
    )


if __name__ == "__main__":
    main()

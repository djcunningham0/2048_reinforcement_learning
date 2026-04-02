"""Watch a trained agent play 2048 in the terminal."""

import argparse
import curses
import time

import torch

from rl_2048.network import ConvNetwork
from rl_2048.game import Action, Game2048, apply_action, encode_state
from rl_2048.expectimax import (
    expectimax_action,
    make_afterstate_value_fn,
    make_dqn_value_fn,
)
from scripts.play import CELL_W, _tile_attr

MODEL_TYPES = ("dqn", "afterstate")
DELAY_STEPS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def draw(
    stdscr: curses.window,
    game: Game2048,
    action: Action | None,
    game_over: bool,
    model_type: str,
    depth: int = 0,
):
    stdscr.erase()
    label = model_type.upper()
    depth_str = f"  depth={depth}" if depth > 0 else ""
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


def load_model(checkpoint_path: str, device: str, model_type: str) -> ConvNetwork:
    output_dim = 4 if model_type == "dqn" else 1
    model = ConvNetwork(output_dim=output_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["online_net"])
    model.eval()
    return model.to(torch.device(device))


def select_action_dqn(
    model: ConvNetwork,
    board: list[int],
    valid_actions: list[Action],
    device: str,
) -> Action:
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
    board: list[int],
    valid_actions: list[Action],
    device: str,
) -> Action:
    afterstates = [apply_action(board, a) for a in valid_actions]
    encoded = torch.stack([encode_state(s) for s, _ in afterstates])
    with torch.no_grad():
        values = model(encoded.to(device)).squeeze(-1)
    rewards = torch.tensor([r for _, r in afterstates], device=device)
    action_values = rewards + values
    return valid_actions[action_values.argmax().item()]


def watch(
    stdscr: curses.window,
    model: ConvNetwork,
    device: str,
    delay_idx: int,
    model_type: str,
    depth: int = 0,
):
    if depth > 0:
        if model_type == "afterstate":
            value_fn = make_afterstate_value_fn(model, device)
        else:
            value_fn = make_dqn_value_fn(model, device)

        def select_action(_model, board, _valid_actions, _device):
            return expectimax_action(board, value_fn, depth)

    else:
        select_action = (
            select_action_dqn if model_type == "dqn" else select_action_afterstate
        )

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
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file (.pt)")
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
        type=int,
        default=0,
        help="Expectimax search depth in plies (default: 0 = greedy)",
    )
    args = parser.parse_args()

    delay_idx = min(
        range(len(DELAY_STEPS)), key=lambda i: abs(DELAY_STEPS[i] - args.delay)
    )
    model = load_model(args.checkpoint, args.device, args.model_type)
    curses.wrapper(
        lambda stdscr: watch(
            stdscr, model, args.device, delay_idx, args.model_type, args.depth
        )
    )


if __name__ == "__main__":
    main()

"""
Use a trained agent to play the custom 2048 on a live website via browser automation.

Usage:
    python -m scripts.play_web checkpoints/checkpoint.pt --model-type afterstate
    python -m scripts.play_web checkpoints/ntuple_model.npz --model-type ntuple --depth adaptive
    ...
"""

import argparse
import os
import time

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page

from rl_2048.game import Action, Board, apply_action
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


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable not set.")
    return value


load_dotenv()
GAME_URL = require_env("GAME_URL")
GAME_STORAGE_KEY = require_env("GAME_STORAGE_KEY")
GAME_CONTAINER_SELECTOR = require_env("GAME_CONTAINER_SELECTOR")
NEW_GAME_BUTTON_SELECTOR = require_env("NEW_GAME_BUTTON_SELECTOR")

ACTION_KEYS = {
    Action.UP: "ArrowUp",
    Action.RIGHT: "ArrowRight",
    Action.DOWN: "ArrowDown",
    Action.LEFT: "ArrowLeft",
}

READ_BOARD_JS = f"""() => {{
    const raw = localStorage.getItem("{GAME_STORAGE_KEY}");
    if (!raw) return null;
    const state = JSON.parse(raw);
    const board = [];
    for (let r = 0; r < 4; r++) {{
        for (let c = 0; c < 4; c++) {{
            const cell = state.grid[r][c];
            board.push(cell ? cell.value : 0);
        }}
    }}
    return {{ board: board, score: state.score, over: state.over }};
}}"""


def read_board(page: Page) -> tuple[Board, int, bool] | None:
    """
    Read the board state from the browser's localStorage.

    Returns (board, score, game_over) or None if no game state found.
    """
    result = page.evaluate(READ_BOARD_JS)
    if result is None:
        return None
    return tuple(result["board"]), result["score"], result["over"]


def get_valid_actions(board: Board) -> list[Action]:
    """Return actions that would change the board."""
    return [a for a in Action if apply_action(board, a)[0] != board]


def send_action(page: Page, action: Action):
    """Send an arrow key press to the game."""
    page.keyboard.press(ACTION_KEYS[action])


def wait_for_board_change(
    page: Page,
    old_board: Board,
    timeout: float = 2.0,
    poll_interval: float = 0.02,
) -> tuple[Board, int, bool] | None:
    """Poll until the board state changes or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = read_board(page)
        if state is None:
            return None
        board, score, game_over = state
        if board != old_board or game_over:
            return state
        time.sleep(poll_interval)

    # Timeout — board never changed; move may not have been applied
    print(f"\n[WARNING] Board did not change within {timeout} seconds")
    return read_board(page)


def build_select_action(
    model,
    device: str,
    model_type: str,
    depth: int | DepthSchedule,
):
    """Build an action-selection function."""
    use_search = isinstance(depth, DepthSchedule) or depth > 0
    if use_search:
        if model_type == "ntuple":
            value_fn = model.evaluate_batch
        elif model_type == "afterstate":
            value_fn = make_afterstate_value_fn(model, device)
        else:
            value_fn = make_dqn_value_fn(model, device)

        def select_action(_model, board, _valid_actions, _device):
            return expectimax_action(board, value_fn, depth)

        return select_action

    action_fns = {
        "dqn": select_action_dqn,
        "afterstate": select_action_afterstate,
        "ntuple": select_action_ntuple,
    }
    return action_fns[model_type]


def game_loop(
    page: Page,
    model,
    device: str,
    model_type: str,
    depth: int | DepthSchedule,
    move_delay: float,
):
    select_action = build_select_action(model, device, model_type, depth)

    while True:
        input("Press Enter to start playing (set up the game in the browser first)...")

        move_count = 0
        while True:
            state = read_board(page)
            if state is None:
                print("No game state found in localStorage. Start a game first.")
                break

            board, score, game_over = state
            if game_over:
                print(
                    f"\nGAME OVER — Score: {score}  "
                    f"Max tile: {max(board)}  "
                    f"Moves: {move_count}"
                )
                break

            valid_actions = get_valid_actions(board)
            if not valid_actions:
                print(
                    f"\nNo valid moves — Score: {score}  "
                    f"Max tile: {max(board)}  "
                    f"Moves: {move_count}"
                )
                break

            action = select_action(model, board, valid_actions, device)
            send_action(page, action)

            new_state = wait_for_board_change(page, board)
            if new_state is None or new_state[0] == board:
                continue

            board, score, game_over = new_state
            move_count += 1

            print(
                f"\rMove {move_count:<5} {action.name:<6} "
                f"Score: {score:<8} Max: {max(board)}",
                end="",
                flush=True,
            )

            if move_delay > 0:
                time.sleep(move_delay)

        response = input("\nPress Enter to play again, or q to quit: ")
        if response.strip().lower() == "q":
            break

        # Click "New Game" button
        page.click(NEW_GAME_BUTTON_SELECTOR)
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Use a trained agent to play 2048 on a live website"
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=MODEL_TYPES,
        default="afterstate",
        help="Model type (default: afterstate)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--depth",
        type=parse_depth,
        default=0,
        help=(
            "Search depth: an integer, 'adaptive', or a custom "
            "schedule like '10:1,6:2,0:3' (default: 0 = greedy)"
        ),
    )
    parser.add_argument(
        "--move-delay",
        type=float,
        default=0.0,
        help="Seconds between moves (default: 0.0)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=GAME_URL,
        help=f"Game URL (default: {GAME_URL})",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device, args.model_type)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(args.url, wait_until="networkidle")
        print(f"Opened {args.url}")

        # Disable CSS animations so the game renders instantly
        page.add_style_tag(
            content="* { transition: none !important; animation: none !important; }"
        )

        # Click on the game area so keyboard events are captured
        page.click(GAME_CONTAINER_SELECTOR)

        game_loop(
            page=page,
            model=model,
            device=args.device,
            model_type=args.model_type,
            depth=args.depth,
            move_delay=args.move_delay,
        )

        input("Press Enter to close the browser...")
        browser.close()


if __name__ == "__main__":
    main()

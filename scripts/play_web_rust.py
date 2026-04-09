"""
Use a Rust-powered N-tuple agent to play 2048 on a live website via browser automation.

Uses Playwright (Python) for fast browser interaction and rust_2048_py (Rust via PyO3)
for N-tuple network inference and expectimax search.

NOTE: this script plays a custom 2048 implementation (URL omitted for privacy), not the
official 2048 game. The game mechanics are the same, but some changes would likely be
needed to support the official version, or any other URL.

Usage:
    python -m scripts.play_web_rust rust_2048/checkpoints/checkpoint.bin
    python -m scripts.play_web_rust rust_2048/checkpoints/checkpoint.bin --depth adaptive
    python -m scripts.play_web_rust rust_2048/checkpoints/checkpoint.bin --depth 2
"""

import argparse
import os
import time

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page

from rl_2048.game import Action
from rust_2048_py import RustNTupleNetwork, board_from_python


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


def read_board(page: Page) -> tuple[tuple[int, ...], int, bool] | None:
    """Read the board state from the browser's localStorage."""
    result = page.evaluate(READ_BOARD_JS)
    if result is None:
        return None
    return tuple(result["board"]), result["score"], result["over"]


def send_action(page: Page, action: Action):
    page.keyboard.press(ACTION_KEYS[action])


def wait_for_board_change(
    page: Page,
    old_board: tuple[int, ...],
    timeout: float = 2.0,
    poll_interval: float = 0.02,
) -> tuple[tuple[int, ...], int, bool] | None:
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

    print(f"\n[WARNING] Board did not change within {timeout} seconds")
    return read_board(page)


def select_action(
    network: RustNTupleNetwork,
    board: tuple[int, ...],
    depth: str,
) -> Action | None:
    """Select an action using the Rust N-tuple network."""
    board_u64 = board_from_python(list(board))
    if depth == "0":
        result = network.select_action(board_u64)
    else:
        result = network.expectimax_action(board_u64, depth)
    if result is None:
        return None
    return Action(result)


def game_loop(
    page: Page,
    network: RustNTupleNetwork,
    depth: str,
    move_delay: float,
):
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

            action = select_action(network, board, depth)
            if action is None:
                print(
                    f"\nNo valid moves — Score: {score}  "
                    f"Max tile: {max(board)}  "
                    f"Moves: {move_count}"
                )
                break

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

        page.click(NEW_GAME_BUTTON_SELECTOR)
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Play 2048 on a live website with a Rust-powered N-tuple agent"
    )
    parser.add_argument(
        "checkpoint", type=str, help="Path to Rust checkpoint file (.bin)"
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="0",
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

    network = RustNTupleNetwork.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(args.url, wait_until="networkidle")
        print(f"Opened {args.url}")

        page.add_style_tag(
            content="* { transition: none !important; animation: none !important; }"
        )
        page.click(GAME_CONTAINER_SELECTOR)

        game_loop(
            page=page,
            network=network,
            depth=args.depth,
            move_delay=args.move_delay,
        )

        input("Press Enter to close the browser...")
        browser.close()


if __name__ == "__main__":
    main()

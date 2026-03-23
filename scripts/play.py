"""Interactive terminal game, using arrow keys to move."""

import curses

from rl_2048.game import Action, Game2048

KEY_TO_ACTION = {
    curses.KEY_UP: Action.UP,
    curses.KEY_RIGHT: Action.RIGHT,
    curses.KEY_DOWN: Action.DOWN,
    curses.KEY_LEFT: Action.LEFT,
}

CELL_W = 7  # width of each board cell in characters

# Map tile value -> color pair index (initialized in play())
# fmt: off
TILE_COLOR = {
    2: 1,     # white
    4: 2,     # cyan
    8: 3,     # green
    16: 4,    # yellow
    32: 5,    # red
    64: 6,    # magenta
    128: 7,   # blue
    256: 2,   # cyan + bold
    512: 3,   # green + bold
    1024: 4,  # yellow + bold
    2048: 5,   # red + bold
    4096: 6,   # magenta + bold
    8192: 7,   # blue + bold
    16384: 1,  # white + bold
    32768: 2,  # cyan + bold
}
# fmt: on


def _tile_attr(v: int) -> int:
    pair = TILE_COLOR.get(v, 0)
    attr = curses.color_pair(pair)
    if v >= 256:
        attr |= curses.A_BOLD
    return attr


def draw(stdscr: curses.window, game: Game2048, msg: str) -> None:
    stdscr.erase()
    stdscr.addstr(0, 0, "2048 — arrows: move  r: reset  q: quit")
    stdscr.addstr(1, 0, f"Score: {game.score:<10}  {msg}")

    stdscr.addstr(3, 0, ("+" + "-" * CELL_W) * 4 + "+")
    for r in range(4):
        stdscr.addstr(4 + r * 2, 0, "|")
        for c in range(4):
            v = game.board[r][c]
            cell = str(v) if v else "."
            stdscr.addstr(f"{cell:^{CELL_W}}", _tile_attr(v))
            stdscr.addstr("|")
        stdscr.addstr(5 + r * 2, 0, ("+" + "-" * CELL_W) * 4 + "+")

    if game._done:
        stdscr.addstr(15, 0, "*** GAME OVER *** — press r to restart")

    stdscr.refresh()


def play(stdscr: curses.window) -> None:
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

    game = Game2048()
    game.reset()
    msg = ""

    while True:
        draw(stdscr, game, msg)
        key = stdscr.getch()

        if key == ord("q"):
            break
        elif key == ord("r"):
            game.reset()
            msg = ""
        elif key in KEY_TO_ACTION:
            if game._done:
                msg = "Game over — press r to restart"
                continue
            action = KEY_TO_ACTION[key]
            board_before = game._copy_board()
            game.step(action)
            if game.board == board_before:
                msg = f"{action.name.lower()}: invalid move"
            else:
                msg = f"moved {action.name.lower()}"


if __name__ == "__main__":
    curses.wrapper(play)

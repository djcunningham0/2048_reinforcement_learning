"""Evaluate model performance across different expectimax search depths."""

import argparse
import statistics
import time
from collections import Counter

from rl_2048.game import Action, Game2048, apply_action, encode_state
from rl_2048.expectimax import (
    expectimax_action,
    make_afterstate_value_fn,
    make_dqn_value_fn,
)
from scripts.watch_agent import (
    load_model,
    select_action_afterstate,
    select_action_dqn,
)

MODEL_TYPES = ("dqn", "afterstate")


def run_game(
    select_fn: callable,
) -> tuple[int, int, float]:
    """Play one game. Returns (score, max_tile, avg_ms_per_move)."""
    game = Game2048()
    game.reset()
    total_time = 0.0
    moves = 0

    while True:
        valid = game.get_valid_actions()
        if not valid:
            break
        t0 = time.perf_counter()
        action = select_fn(game.board, valid)
        total_time += time.perf_counter() - t0
        game.step(action)
        moves += 1

    avg_ms = (total_time / max(moves, 1)) * 1000
    return game.score, max(game.board), avg_ms


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate expectimax search at various depths"
    )
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
        "--depths",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Search depths to evaluate (default: 0 1 2)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Games per depth (default: 50)",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device, args.model_type)

    if args.model_type == "afterstate":
        value_fn = make_afterstate_value_fn(model, args.device)
        greedy_fn = lambda board, valid: select_action_afterstate(
            model, board, valid, args.device
        )
    else:
        value_fn = make_dqn_value_fn(model, args.device)
        greedy_fn = lambda board, valid: select_action_dqn(
            model, board, valid, args.device
        )

    results_summary = []

    for depth in args.depths:
        if depth == 0:
            select_fn = greedy_fn
        else:
            d = depth  # capture for closure
            select_fn = lambda board, valid, _d=d: expectimax_action(
                board, value_fn, _d
            )

        scores = []
        max_tiles = []
        times_ms = []

        print(f"\n--- Depth {depth} ({args.games} games) ---")
        for i in range(args.games):
            score, max_tile, avg_ms = run_game(select_fn)
            scores.append(score)
            max_tiles.append(max_tile)
            times_ms.append(avg_ms)
            print(
                f"  game {i + 1:>{len(str(args.games))}}/{args.games}: "
                f"score={score:>7,}  max={max_tile:>5}  "
                f"avg={avg_ms:>8.1f}ms/move"
            )

        tile_counts = Counter(max_tiles)
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)

        print(f"\nDepth {depth} summary:")
        print(
            f"  score: mean={mean_score:,.0f}  median={median_score:,.0f}  "
            f"max={max(scores):,}"
        )
        print(f"  avg time/move: {statistics.mean(times_ms):.1f}ms")
        print("  max tile distribution:")
        for tile in sorted(tile_counts, reverse=True):
            pct = tile_counts[tile] / len(max_tiles) * 100
            print(f"    {tile:>5}: {pct:5.1f}% ({tile_counts[tile]}/{len(max_tiles)})")

        results_summary.append(
            (depth, mean_score, median_score, statistics.mean(times_ms))
        )

    if len(results_summary) > 1:
        print("\n=== Comparison ===")
        print(
            f"{'Depth':>6} {'Mean Score':>12} {'Median Score':>14} {'Avg ms/move':>12}"
        )
        for depth, mean_s, med_s, avg_t in results_summary:
            print(f"{depth:>6} {mean_s:>12,.0f} {med_s:>14,.0f} {avg_t:>12.1f}")


if __name__ == "__main__":
    main()

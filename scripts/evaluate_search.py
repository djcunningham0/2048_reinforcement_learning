"""Evaluate model performance across different expectimax search depths.

Usage examples::

    python -m scripts.evaluate_search checkpoints/model.pt
    python -m scripts.evaluate_search checkpoints/model.pt --depths 0 1 2 3
    python -m scripts.evaluate_search checkpoints/model.pt --depths adaptive
    python -m scripts.evaluate_search checkpoints/model.pt --depths 1 adaptive 10:1,6:2,0:3
    python -m scripts.evaluate_search checkpoints/model.pt --model-type dqn --games 100
    python -m scripts.evaluate_search checkpoints/ntuple.npz --model-type ntuple --depths 0 1
"""

import argparse
import statistics
import time
from collections import Counter

from rl_2048.game import Action, Game2048, apply_action, encode_state
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
        "--depths",
        type=parse_depth,
        nargs="+",
        default=[0, 1, 2],
        help=(
            "Search depths to evaluate (default: 0 1 2). "
            "Each value can be an integer, 'adaptive', or a custom "
            "schedule like '10:1,6:2,0:3'."
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Games per depth (default: 50)",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device, args.model_type)

    if args.model_type == "ntuple":
        value_fn = model.evaluate_batch
        greedy_fn = lambda board, valid: select_action_ntuple(
            model, board, valid, args.device
        )
    elif args.model_type == "afterstate":
        value_fn = make_afterstate_value_fn(model, args.device)
        greedy_fn = lambda board, valid: select_action_afterstate(
            model, board, valid, args.device
        )
    else:
        value_fn = make_dqn_value_fn(model, args.device)
        greedy_fn = lambda board, valid: select_action_dqn(
            model, board, valid, args.device
        )

    def _label(d: int | DepthSchedule) -> str:
        return "adaptive" if isinstance(d, DepthSchedule) else str(d)

    configs = [(_label(d), d) for d in args.depths]

    results_summary = []

    for label, depth in configs:
        if depth == 0:
            select_fn = greedy_fn
        else:
            _depth = depth  # capture for closure
            select_fn = lambda board, valid, _d=_depth: expectimax_action(
                board, value_fn, _d
            )

        scores = []
        max_tiles = []
        times_ms = []

        print(f"\n--- Depth {label} ({args.games} games) ---")
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

        print(f"\nDepth {label} summary:")
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
            (label, mean_score, median_score, statistics.mean(times_ms))
        )

    if len(results_summary) > 1:
        print("\n=== Comparison ===")
        print(
            f"{'Depth':>8} {'Mean Score':>12} {'Median Score':>14} {'Avg ms/move':>12}"
        )
        for label, mean_s, med_s, avg_t in results_summary:
            print(f"{label:>8} {mean_s:>12,.0f} {med_s:>14,.0f} {avg_t:>12.1f}")


if __name__ == "__main__":
    main()

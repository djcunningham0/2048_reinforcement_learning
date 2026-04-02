"""Training entry point for n-tuple network agent."""

import argparse
import logging

from rl_2048.ntuple.config import NTupleConfig
from rl_2048.ntuple.train import train


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train n-tuple network agent for 2048")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="TensorBoard run name (default: ntuple_{timestamp}). Use slashes for grouping.",
    )
    args = parser.parse_args()

    config = NTupleConfig()
    for field in ["max_episodes", "lr", "eval_interval", "eval_episodes"]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(config, field, val)

    train(
        config,
        run_dir=args.run_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

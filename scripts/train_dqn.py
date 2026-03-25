"""Training entry point for DQN agent."""

import argparse
import logging

from rl_2048.dqn.config import DQNConfig
from rl_2048.dqn.train import train


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train DQN agent for 2048")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--buffer-capacity", type=int, default=None)
    parser.add_argument("--train-start", type=int, default=None)
    parser.add_argument("--train-freq", type=int, default=None)
    parser.add_argument("--target-sync-interval", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    config = DQNConfig()
    for field in [
        "max_episodes",
        "lr",
        "batch_size",
        "buffer_capacity",
        "train_start",
        "train_freq",
        "target_sync_interval",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
        "gamma",
        "grad_clip_norm",
        "eval_interval",
        "eval_episodes",
        "device",
    ]:
        val = getattr(args, field.replace("-", "_"), None)
        if val is not None:
            setattr(config, field, val)

    train(config, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()

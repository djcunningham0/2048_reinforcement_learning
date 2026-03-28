"""TD-afterstate training loop."""

import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from rl_2048.afterstate.agent import AfterstateAgent
from rl_2048.afterstate.config import AfterstateConfig
from rl_2048.afterstate.replay_buffer import (
    AfterstateReplayBuffer,
    make_transition,
)
from rl_2048.game import Game2048
from rl_2048.profiler import Profiler

logger = logging.getLogger(__name__)


def train(
    config: AfterstateConfig,
    run_dir: str = "runs",
    run_name: str | None = None,
):
    """Main TD-afterstate training loop."""
    logger.info("Training started")

    group_name = run_name or "afterstate"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(run_dir) / group_name / timestamp

    writer = SummaryWriter(log_dir=str(run_path))
    logger.info("TensorBoard log dir: %s", writer.log_dir)
    hparams_subdir = _log_hyperparams(writer, config)
    checkpoint_path = run_path / hparams_subdir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    game = Game2048()
    agent = AfterstateAgent(config)
    _log_network(writer, agent)
    logger.info("Device: %s", agent.device)
    buffer = AfterstateReplayBuffer(config.buffer_capacity)

    global_step = 0
    recent_losses: list[float] = []
    last_eval: dict = {}
    profiler = Profiler()

    try:
        for episode in range(1, config.max_episodes + 1):
            global_step = _run_episode(
                game=game,
                agent=agent,
                buffer=buffer,
                config=config,
                global_step=global_step,
                recent_losses=recent_losses,
                writer=writer,
                profiler=profiler,
            )

            max_tile = max(game.board)
            avg_loss = (
                sum(recent_losses[-100:]) / len(recent_losses[-100:])
                if recent_losses
                else 0.0
            )

            writer.add_scalar("train/score", game.score, episode)
            writer.add_scalar("train/max_tile", max_tile, episode)
            writer.add_scalar("train/epsilon", config.epsilon_at(global_step), episode)
            writer.add_scalar("train/avg_loss", avg_loss, episode)
            writer.add_scalar("train/buffer_size", len(buffer), episode)
            writer.add_scalar("train/global_step", global_step, episode)

            if episode % 100 == 0:
                logger.info(
                    "Ep %d | Score: %d | Max tile: %d | ε: %.3f | Avg loss: %.4f | Steps: %d | Buffer size: %d",
                    episode,
                    game.score,
                    max_tile,
                    config.epsilon_at(global_step),
                    avg_loss,
                    global_step,
                    len(buffer),
                )
                profiler.log_and_reset(episode, writer)

            if episode % config.eval_interval == 0:
                profiler.begin()
                last_eval = evaluate(game, agent, config.eval_episodes)
                profiler.record("eval")
                logger.info(
                    "EVAL Ep %d | Mean score: %.0f | Max: %d | Tiles: %s",
                    episode,
                    last_eval["mean_score"],
                    last_eval["max_score"],
                    last_eval["tile_distribution"],
                )
                writer.add_scalar("eval/mean_score", last_eval["mean_score"], episode)
                writer.add_scalar("eval/max_score", last_eval["max_score"], episode)
                for tile_val, count in last_eval["tile_distribution"].items():
                    writer.add_scalar(f"eval/tile_{tile_val}", count, episode)

                profiler.begin()
                ep_str = str(episode).zfill(len(str(config.max_episodes)))
                agent.save(checkpoint_path / f"checkpoint_ep{ep_str}.pt", global_step)
                profiler.record("checkpoint_save")

        agent.save(checkpoint_path / "final.pt", global_step)
        logger.info(
            "Training complete. Model saved to %s",
            checkpoint_path / "final.pt",
        )
    finally:
        writer.add_scalar("final/mean_score", last_eval.get("mean_score", 0.0))
        writer.add_scalar("final/max_score", last_eval.get("max_score", 0))
        writer.close()


def _run_episode(
    game: Game2048,
    agent: AfterstateAgent,
    buffer: AfterstateReplayBuffer,
    config: AfterstateConfig,
    global_step: int,
    recent_losses: list[float],
    writer: SummaryWriter,
    profiler: Profiler,
) -> int:
    """Play one training episode. Returns updated global_step."""
    game.reset()
    valid_actions = game.get_valid_actions()

    while valid_actions:
        profiler.begin()
        epsilon = config.epsilon_at(global_step)
        action, afterstate = agent.select_action(game.board, valid_actions, epsilon)
        profiler.record("select_action")

        game.step(action)
        next_valid = game.get_valid_actions()
        done = len(next_valid) == 0
        profiler.record("env_step")

        buffer.push(make_transition(afterstate, game.board, done))
        profiler.record("buffer_push")

        valid_actions = next_valid
        global_step += 1

        if len(buffer) >= config.train_start:
            if global_step % config.train_freq == 0:
                profiler.begin()
                batch = buffer.sample(config.batch_size)
                profiler.record("sample")

                loss = agent.train_step(batch)
                profiler.record("train_step")

                recent_losses.append(loss)
                writer.add_scalar("train/step_loss", loss, global_step)

            if global_step % config.target_sync_interval == 0:
                profiler.begin()
                agent.sync_target_network()
                profiler.record("target_sync")
                logger.debug("Step %d: target network synced", global_step)

    return global_step


def evaluate(game: Game2048, agent: AfterstateAgent, num_episodes: int) -> dict:
    """Run evaluation episodes with epsilon=0 (greedy)."""
    scores: list[int] = []
    max_tiles: list[int] = []
    for _ in range(num_episodes):
        game.reset()
        valid_actions = game.get_valid_actions()
        while valid_actions:
            action, _ = agent.select_action(game.board, valid_actions, epsilon=0.0)
            game.step(action)
            valid_actions = game.get_valid_actions()
        scores.append(game.score)
        max_tiles.append(max(game.board))

    tile_counts = Counter(max_tiles)
    return {
        "mean_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "tile_distribution": dict(sorted(tile_counts.items())),
    }


def _log_hyperparams(writer: SummaryWriter, config: AfterstateConfig) -> str:
    """Log hyperparameters for TensorBoard.

    Returns the hparams subdirectory name used by TensorBoard.
    """
    hparam_dict = {
        k: v
        for k, v in asdict(config).items()
        if isinstance(v, (int, float, str, bool))
    }
    metric_dict = {
        "final/mean_score": 0.0,
        "final/max_score": 0,
    }
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)
    return run_name


def _log_network(writer: SummaryWriter, agent: AfterstateAgent):
    """Log network architecture and parameter count to TensorBoard."""
    model = agent.online_net
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Network: %d params (%d trainable)", total_params, trainable_params)
    writer.add_text("network/architecture", f"```\n{model}\n```")
    writer.add_text(
        "network/params",
        f"Total: {total_params:,} | Trainable: {trainable_params:,}",
    )

"""Training loop for N-tuple networks."""

import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from rl_2048.game import Action, Board, Game2048, apply_action
from rl_2048.ntuple.config import NTupleConfig
from rl_2048.ntuple.network import NTupleNetwork
from rl_2048.profiler import Profiler

logger = logging.getLogger(__name__)


def train(
    config: NTupleConfig,
    run_dir: str = "runs",
    run_name: str | None = None,
    resume: str | None = None,
    offset: int = 0,
):
    """
    Main n-tuple network training loop.

    Parameters
    ----------
    resume : str | None
        Path to a checkpoint .npz file to resume training from
    offset : int
        Episode offset to start from when resuming
    """
    logger.info("Training started")

    group_name = run_name or "ntuple"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = Path(run_dir) / group_name / timestamp

    writer = SummaryWriter(log_dir=str(run_path))
    logger.info("TensorBoard log dir: %s", writer.log_dir)
    hparams_subdir = _log_hyperparams(writer, config)
    checkpoint_path = run_path / hparams_subdir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    game = Game2048()

    if resume:
        resume_path = Path(resume)
        network = NTupleNetwork.load(resume_path)
        logger.info("Loaded checkpoint from %s", resume_path)
    else:
        network = NTupleNetwork(config.patterns, v_init=config.v_init)

    _log_network(writer, network)
    profiler = Profiler()

    last_eval: dict = {}

    try:
        for episode in range(1 + offset, config.max_episodes + 1):
            _run_episode(game, network, config, profiler)

            max_tile = max(game.board)
            writer.add_scalar("train/score", game.score, episode)
            writer.add_scalar("train/max_tile", max_tile, episode)

            if episode % 100 == 0:
                logger.info(
                    "Ep %d | Score: %d | Max tile: %d",
                    episode,
                    game.score,
                    max_tile,
                )
                profiler.log_and_reset(episode, writer)

            if episode % config.eval_interval == 0:
                profiler.begin()
                last_eval = evaluate(game, network, config.eval_episodes)
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
                    pct = count / config.eval_episodes
                    writer.add_scalar(f"eval/tile_pct{tile_val}", pct, episode)

                profiler.begin()
                ep_str = str(episode).zfill(len(str(config.max_episodes)))
                network.save(checkpoint_path / f"checkpoint_ep{ep_str}.npz")
                profiler.record("checkpoint_save")

        network.save(checkpoint_path / "final.npz")
        logger.info(
            "Training complete. Model saved to %s",
            checkpoint_path / "final.npz",
        )
    finally:
        writer.add_scalar("final/mean_score", last_eval.get("mean_score", 0.0))
        writer.add_scalar("final/max_score", last_eval.get("max_score", 0))
        writer.close()


def _run_episode(
    game: Game2048,
    network: NTupleNetwork,
    config: NTupleConfig,
    profiler: Profiler,
):
    """Play one episode with online TD(0) updates."""
    game.reset()

    while True:
        profiler.begin()
        action, afterstate = _select_action(network, game.board)
        profiler.record("select_action")
        if action is None or afterstate is None:
            break

        current_value = network.evaluate(afterstate)

        profiler.begin()
        game.step(action)
        profiler.record("env_step")

        # TD target from post-spawn board
        profiler.begin()
        target = _best_afterstate_value(network, game.board)
        td_error = target - current_value
        network.update(afterstate, config.lr * td_error)
        profiler.record("td_update")


def evaluate(game: Game2048, network: NTupleNetwork, num_episodes: int) -> dict:
    """Run evaluation episodes with greedy policy (no updates)."""
    scores: list[int] = []
    max_tiles: list[int] = []
    for _ in range(num_episodes):
        game.reset()
        while True:
            action, _ = _select_action(network, game.board)
            if action is None:
                break
            game.step(action)
        scores.append(game.score)
        max_tiles.append(max(game.board))

    tile_counts = Counter(max_tiles)
    return {
        "mean_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "tile_distribution": dict(sorted(tile_counts.items())),
    }


def _best_afterstate_value(network: NTupleNetwork, board: Board) -> float:
    """
    Compute max_a [reward(a) + V(afterstate(a))] from a board state. Returns 0 if no
    valid actions (terminal).
    """
    best = 0.0
    found_valid = False
    for action in Action:
        afterstate, reward = apply_action(board, action)
        if afterstate == board:
            continue
        value = reward + network.evaluate(afterstate)
        if not found_valid or value > best:
            best = value
            found_valid = True
    return best


def _select_action(
    network: NTupleNetwork,
    board: Board,
) -> tuple[Action | None, Board | None]:
    """Select best action, returning (action, afterstate). (None, None) if terminal."""
    best_action: Action | None = None
    best_afterstate: Board | None = None
    best_value = float("-inf")

    for action in Action:
        afterstate, reward = apply_action(board, action)
        if afterstate == board:
            continue
        value = reward + network.evaluate(afterstate)
        if value > best_value:
            best_action = action
            best_afterstate = afterstate
            best_value = value

    return best_action, best_afterstate


def _log_hyperparams(writer: SummaryWriter, config: NTupleConfig) -> str:
    """Log hyperparameters for TensorBoard. Returns hparams subdirectory name."""
    hparam_dict = {
        k: v
        for k, v in asdict(config).items()
        if isinstance(v, (int, float, str, bool))
    }
    hparam_dict["num_patterns"] = len(config.patterns)
    hparam_dict["tuple_size"] = len(config.patterns[0]) if config.patterns else 0
    metric_dict = {
        "final/mean_score": 0.0,
        "final/max_score": 0,
    }
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer.add_hparams(hparam_dict, metric_dict, run_name=run_name)
    return run_name


def _log_network(writer: SummaryWriter, network: NTupleNetwork):
    """Log network info to TensorBoard."""
    total_entries = sum(len(lut) for lut in network.luts)
    memory_mb = total_entries * 4 / (1024 * 1024)
    info = (
        f"N-Tuple Network: {network.num_patterns} patterns, "
        f"{total_entries:,} total LUT entries, "
        f"{memory_mb:.0f} MB"
    )
    logger.info(info)
    writer.add_text("network/architecture", info)
    writer.add_text(
        "network/patterns",
        "\n".join(f"Pattern {i}: {p}" for i, p in enumerate(network.patterns)),
    )

"""DQN training loop."""

import logging
from collections import Counter
from pathlib import Path

import torch

from rl_2048.dqn.agent import DQNAgent
from rl_2048.dqn.config import DQNConfig
from rl_2048.dqn.replay_buffer import ReplayBuffer, Transition
from rl_2048.game import Action, Game2048, encode_state

logger = logging.getLogger(__name__)


def train(config: DQNConfig, checkpoint_dir: str = "checkpoints"):
    """Main DQN training loop."""
    logger.info("Training started")
    checkpoint_path = Path(checkpoint_dir)
    game = Game2048()
    agent = DQNAgent(config)
    logger.info("Device: %s", agent.device)
    buffer = ReplayBuffer(config.buffer_capacity)

    global_step = 0
    recent_losses: list[float] = []

    for episode in range(1, config.max_episodes + 1):
        global_step = _run_episode(
            game=game,
            agent=agent,
            buffer=buffer,
            config=config,
            global_step=global_step,
            recent_losses=recent_losses,
        )

        max_tile = max(game.board)
        avg_loss = (
            sum(recent_losses[-100:]) / len(recent_losses[-100:])
            if recent_losses
            else 0.0
        )

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

        if episode % config.eval_interval == 0:
            eval_result = evaluate(game, agent, config.eval_episodes)
            logger.info(
                "EVAL Ep %d | Mean score: %.0f | Max: %d | Tiles: %s",
                episode,
                eval_result["mean_score"],
                eval_result["max_score"],
                eval_result["tile_distribution"],
            )
            ep_str = str(episode).zfill(len(str(config.max_episodes)))
            agent.save(checkpoint_path / f"checkpoint_ep{ep_str}.pt", global_step)

    agent.save(checkpoint_path / "final.pt", global_step)
    logger.info("Training complete. Model saved to %s", checkpoint_path / "final.pt")


def _run_episode(
    game: Game2048,
    agent: DQNAgent,
    buffer: ReplayBuffer,
    config: DQNConfig,
    global_step: int,
    recent_losses: list[float],
) -> int:
    """
    Play one training episode (one game). Put each transition in the replay buffer.
    Returns updated global_step.
    """
    game.reset()
    valid_actions = game.get_valid_actions()
    state = encode_state(game.board)

    while valid_actions:
        epsilon = config.epsilon_at(global_step)
        action = agent.select_action(state, valid_actions, epsilon)
        reward = game.step(action)
        next_valid = game.get_valid_actions()
        next_state = encode_state(game.board)
        done = len(next_valid) == 0

        buffer.push(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                valid_mask=_actions_to_mask(next_valid),
            )
        )

        state = next_state
        valid_actions = next_valid
        global_step += 1

        if len(buffer) >= config.train_start:
            if global_step % config.train_freq == 0:
                loss = agent.train_step(buffer.sample(config.batch_size))
                recent_losses.append(loss)

            if global_step % config.target_sync_interval == 0:
                agent.sync_target_network()
                logger.info("Step %d: target network synced", global_step)

    return global_step


def evaluate(game: Game2048, agent: DQNAgent, num_episodes: int) -> dict:
    """Run evaluation episodes with epsilon=0 (greedy)."""
    scores: list[int] = []
    max_tiles: list[int] = []
    for _ in range(num_episodes):
        game.reset()
        valid_actions = game.get_valid_actions()
        state = encode_state(game.board)
        while valid_actions:
            action = agent.select_action(state, valid_actions, epsilon=0.0)
            game.step(action)
            valid_actions = game.get_valid_actions()
            state = encode_state(game.board)
        scores.append(game.score)
        max_tiles.append(max(game.board))

    tile_counts = Counter(max_tiles)
    return {
        "mean_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "tile_distribution": dict(sorted(tile_counts.items())),
    }


def _actions_to_mask(valid_actions: list[Action]) -> torch.Tensor:
    """Convert a list of valid actions to a boolean mask tensor of shape (4,)."""
    mask = torch.zeros(4, dtype=torch.bool)
    for a in valid_actions:
        mask[a] = True
    return mask

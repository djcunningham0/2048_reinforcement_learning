"""DQN Agent — action selection, training, target sync."""

import copy
import random
from pathlib import Path

import torch
from torch import nn

from rl_2048.dqn.config import DQNConfig
from rl_2048.dqn.network import QNetwork
from rl_2048.dqn.replay_buffer import BatchedTransitions
from rl_2048.game import Action


class DQNAgent:
    """DQN agent with online + target networks."""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.online_net = QNetwork().to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(
        self,
        state: torch.Tensor,
        valid_actions: list[Action],
        epsilon: float,
    ) -> Action:
        """Epsilon-greedy action selection, restricted to valid actions."""
        if random.random() < epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            q_values = self.online_net(state.unsqueeze(0).to(self.device)).squeeze(0)
            mask = torch.full_like(q_values, float("-inf"))
            for a in valid_actions:
                mask[a] = 0.0
            q_values = q_values + mask
            return Action(q_values.argmax().item())

    def train_step(self, batch: BatchedTransitions) -> float:
        """One gradient step on a batch. Returns loss value."""
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device)

        # Current Q-values for taken actions
        q_values = self.online_net(states)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values with vectorized masking
        with torch.no_grad():
            next_q = self.target_net(next_states)
            valid_masks = batch.valid_masks.to(self.device)
            next_q = next_q.masked_fill(~valid_masks, float("-inf"))
            next_q_max = next_q.max(dim=1).values
            next_q_max = torch.clamp(next_q_max, min=0.0)
            targets = rewards + self.config.gamma * next_q_max * (1.0 - dones)

        loss = self.loss_fn(q_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()
        return float(loss.item())

    def sync_target_network(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: Path, global_step: int = 0):
        """Save agent checkpoint (network weights, optimizer state, step)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": global_step,
            },
            path,
        )

    def load(self, path: Path) -> int:
        """Load agent checkpoint. Returns the saved global_step."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("global_step", 0)

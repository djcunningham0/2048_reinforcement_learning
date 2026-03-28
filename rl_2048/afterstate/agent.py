"""Afterstate agent — action selection via V(afterstate), TD training."""

import copy
import random
from pathlib import Path

import torch
from torch import nn

from rl_2048.afterstate.config import AfterstateConfig
from rl_2048.afterstate.replay_buffer import (
    AfterstateInfo,
    BatchedAfterstateTransitions,
)
from rl_2048.game import Action
from rl_2048.network import ConvNetwork


class AfterstateAgent:
    """TD-afterstate agent with online + target value networks."""

    def __init__(self, config: AfterstateConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.online_net = ConvNetwork(output_dim=1).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, info: AfterstateInfo, epsilon: float) -> Action:
        """Select action using pre-computed AfterstateInfo."""
        valid_indices = info.valid_mask.nonzero(as_tuple=False).view(-1).tolist()

        if random.random() < epsilon:
            return Action(random.choice(valid_indices))

        # Evaluate all 4 afterstates in one forward pass, mask invalid
        encoded = info.encoded.to(self.device, non_blocking=True)
        rewards = info.rewards.to(self.device, non_blocking=True)
        valid_mask = info.valid_mask.to(self.device, non_blocking=True)
        with torch.inference_mode():
            values = self.online_net(encoded).squeeze(-1)

        action_values = rewards + self.config.gamma * values
        action_values = action_values.masked_fill(~valid_mask, float("-inf"))
        return Action(action_values.argmax().item())

    def train_step(self, batch: BatchedAfterstateTransitions) -> float:
        """One gradient step on a batch. Returns loss value."""
        afterstates = batch.afterstates.to(self.device)
        next_afterstates = batch.next_afterstates.to(self.device)
        next_rewards = batch.next_rewards.to(self.device)
        next_valid_masks = batch.next_valid_masks.to(self.device)
        dones = batch.dones.to(self.device)

        # afterstate value prediction for actions that were actually taken
        v_pred = self.online_net(afterstates).squeeze(-1)  # (B,)

        # Best action value from the next state: reward + discounted future value
        # TD target: max_a' [r(s',a') + gamma * V_target(afterstate(s',a'))]
        B = next_afterstates.shape[0]
        with torch.inference_mode():
            flat = next_afterstates.view(B * 4, 16, 4, 4)
            next_v = self.target_net(flat).squeeze(-1).view(B, 4)  # (B, 4)
            action_values = next_rewards + self.config.gamma * next_v
            action_values = action_values.masked_fill(~next_valid_masks, float("-inf"))
            targets = action_values.max(dim=1).values
            targets = torch.clamp(targets, min=0.0)
            targets = targets * (1.0 - dones)

        loss = self.loss_fn(v_pred, targets)
        self.optimizer.zero_grad(set_to_none=True)
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
        """Save agent checkpoint."""
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

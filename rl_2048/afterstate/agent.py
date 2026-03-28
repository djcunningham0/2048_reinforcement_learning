"""Afterstate agent — action selection via V(afterstate), TD training."""

import copy
import random
from pathlib import Path

import torch
from torch import nn

from rl_2048.afterstate.config import AfterstateConfig
from rl_2048.afterstate.replay_buffer import BatchedAfterstateTransitions
from rl_2048.game import Action, Board, apply_action, encode_state
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

    def select_action(
        self,
        board: Board,
        valid_actions: list[Action],
        epsilon: float,
    ) -> tuple[Action, Board]:
        """Select action by evaluating V(afterstate) for each valid action."""
        # Compute afterstates for all valid actions
        afterstates: list[tuple[Board, float]] = []
        for a in valid_actions:
            afterstates.append(apply_action(board, a))

        if random.random() < epsilon:
            idx = random.randrange(len(valid_actions))
            return valid_actions[idx], afterstates[idx][0]

        # Batch-evaluate V(afterstate) for all valid actions
        encoded = torch.stack([encode_state(s) for s, _ in afterstates])
        with torch.no_grad():
            values = self.online_net(encoded.to(self.device)).squeeze(-1)

        # Action value = r + gamma * V(afterstate)
        rewards_t = torch.tensor([r for _, r in afterstates], device=self.device)
        action_values = rewards_t + self.config.gamma * values
        best_idx = action_values.argmax().item()

        return valid_actions[best_idx], afterstates[best_idx][0]

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
        with torch.no_grad():
            flat = next_afterstates.view(B * 4, 16, 4, 4)
            next_v = self.target_net(flat).squeeze(-1).view(B, 4)  # (B, 4)
            action_values = next_rewards + self.config.gamma * next_v
            action_values = action_values.masked_fill(~next_valid_masks, float("-inf"))
            targets = action_values.max(dim=1).values
            targets = torch.clamp(targets, min=0.0)
            targets = targets * (1.0 - dones)

        loss = self.loss_fn(v_pred, targets)
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

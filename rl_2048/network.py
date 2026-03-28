import torch
from torch import nn


class ConvNetwork(nn.Module):
    def __init__(self, output_dim: int = 4):
        """
        Convolutional network for 2048 state evaluation.

        Input: (batch_size, 16, 4, 4) one-hot encoding of tile values.
        Output:
        - If output_dim=4: Q-values for each of the 4 actions (e.g., for DQN)
        - If output_dim=1: single scalar value (e.g., for TD-afterstate learning)
        """
        super().__init__()

        # input: 16 channels (up to 32,768 tile)
        self.net = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (batch_size, 16, 4, 4), Output: (batch_size, output_dim)."""
        return self.net(x)

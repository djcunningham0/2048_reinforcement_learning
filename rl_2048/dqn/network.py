import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # input: 16 channels (up to 32,768 tile)
        # output: 4 actions (up, right, down, left)
        self.net = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (batch_size, 16, 4, 4), Output: (batch_size, 4)."""
        return self.net(x)

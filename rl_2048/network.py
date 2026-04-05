from typing import Literal

import torch
from torch import nn

NetworkType = Literal["cnn", "transformer"]


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


class TransformerNetwork(nn.Module):
    def __init__(self, output_dim: int = 4):
        """
        Transformer network for 2048 state evaluation.

        Treats each board cell as a token with learned tile-value and positional
        embeddings. Input/output contract matches ConvNetwork.

        Input: (batch_size, 16, 4, 4) one-hot encoding of tile values.
        Output: (batch_size, output_dim)
        - If output_dim=4: Q-values for each of the 4 actions (e.g., for DQN)
        - If output_dim=1: single scalar value (e.g., for TD-afterstate learning)
        """
        super().__init__()
        embed_dim = 96
        num_heads = 4
        num_layers = 2
        ff_dim = 192
        hidden_dim = 128

        self.tile_embed = nn.Embedding(16, embed_dim)
        self.pos_embed = nn.Embedding(16, embed_dim)
        self.register_buffer("positions", torch.arange(16))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (batch_size, 16, 4, 4), Output: (batch_size, output_dim)."""
        tile_indices = x.argmax(dim=1).reshape(x.size(0), 16)
        tokens = self.tile_embed(tile_indices) + self.pos_embed(self.positions)
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)
        return self.head(self.norm(pooled))


def make_network(network_type: NetworkType = "cnn", output_dim: int = 4) -> nn.Module:
    """Factory for creating network architectures."""
    if network_type == "cnn":
        return ConvNetwork(output_dim=output_dim)
    if network_type == "transformer":
        return TransformerNetwork(output_dim=output_dim)
    raise NotImplementedError(f"Unknown network type: {network_type}")

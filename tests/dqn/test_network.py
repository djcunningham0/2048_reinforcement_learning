"""Tests for Q-Network."""

import torch

from rl_2048.dqn.network import QNetwork


class TestQNetwork:
    def test_output_shape_single(self):
        net = QNetwork()
        x = torch.randn(1, 16, 4, 4)
        out = net(x)
        assert out.shape == (1, 4)

    def test_output_shape_batch(self):
        net = QNetwork()
        x = torch.randn(32, 16, 4, 4)
        out = net(x)
        assert out.shape == (32, 4)

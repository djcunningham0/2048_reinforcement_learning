"""Tests for ConvNetwork."""

import torch

from rl_2048.network import ConvNetwork


class TestConvNetwork:
    def test_output_shape_single(self):
        net = ConvNetwork()
        x = torch.randn(1, 16, 4, 4)
        out = net(x)
        assert out.shape == (1, 4)

    def test_output_shape_batch(self):
        net = ConvNetwork()
        x = torch.randn(32, 16, 4, 4)
        out = net(x)
        assert out.shape == (32, 4)

    def test_output_dim_1(self):
        net = ConvNetwork(output_dim=1)
        x = torch.randn(8, 16, 4, 4)
        out = net(x)
        assert out.shape == (8, 1)

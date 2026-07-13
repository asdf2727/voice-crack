"""
Causal ConvNeXt-style 2D blocks over (time, frequency) maps.

Layout: (B?, C, T, F), channels first (Conv2d's convention), time at dim -2.
Time is strictly causal: blocks use valid time convs (output frame i sees
inputs [i, i + latency]); Downsample/Upsample use (1, stride) kernels, so
they never mix time at all (latency 0). Frequency is same-padded in blocks
and pooled only by the dedicated Downsample/Upsample stages. The self-test
enforces causality.

Note: Large parts of this file were copied over and modified from the open ConvNeXt implementation,
found at: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

import math
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. Adapted to be causal over time (dim -2): frequency is
    same-padded, time is valid, and the residual drops the first `latency`
    frames to match.

    - Parameters: `channels * (kernel[0] * kernel[1] + 2 * bottleneck)`
    - Macs per timestep: `freq * parameters`

    Args:
        channels (int): Number of input and output channels.
        kernel_size (int | tuple(int, int)): Size of the convolving kernel. Assumed square if given int.
        bottleneck (int): Hidden width of the pointwise MLP. Default: `4 * sqrt(in * out channels)`
        layer_scale (float): Init scale of the residual branch. Default: 1e-6.
    """
    def __init__(
            self,
            channels: int,
            kernel_size: int | tuple[int, int] = 7,
            bottleneck: int | None = None,
            layer_scale: float = 1e-6):
        bottleneck = bottleneck or int(4 * channels)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size,
                                padding=(0, (kernel_size[1] - 1) // 2),
                                groups=channels)  # depthwise conv
        self.norm = nn.LayerNorm(channels)
        # pointwise/1x1 convs, implemented with linear layers (channels-last)
        self.up_proj = nn.Linear(channels, bottleneck)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(bottleneck, channels)
        with torch.no_grad():
            self.down_proj.weight *= layer_scale
            self.down_proj.bias *= layer_scale

    @property
    def latency(self) -> int:
        return self.dwconv.kernel_size[0] - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B?, C, T, F) -> (B?, C, T - latency, F)"""
        add = self.dwconv(x)
        add = add.movedim(-3, -1)  # (B?, C, T, F) -> (B?, T, F, C)
        add = self.down_proj(self.act(self.up_proj(self.norm(add))))
        add = add.movedim(-1, -3)  # (B?, T, F, C) -> (B?, C, T, F)
        return x[..., self.latency:, :] + add


class ChannelFirstNorm(nn.Module):
    r"""
    LayerNorm that supports inputs with shape (batch_size?, channels, height, width)
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-3, keepdim=True)
        s = (x - u).square().mean(-3, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[..., :, None, None] * x + self.bias[..., :, None, None]
        return x


class Downsample(nn.Module):
    """Dedicated norm + strided pointwise-in-time conv: halves F, doubles C
    by default. No time mixing (latency 0)."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int | None = None,
            stride: int = 2):
        super().__init__()
        out_channels = out_channels or in_channels * stride
        self.norm = ChannelFirstNorm(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, (1, stride), stride=(1, stride))

    @property
    def latency(self) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B?, C_in, T, F) -> (B?, C_out, T, F // stride)"""
        return self.conv(self.norm(x))


class Upsample(nn.Module):
    """Mirror of Downsample: multiplies F by stride, divides C by default.
    No time mixing (latency 0)."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int | None = None,
            stride: int = 2):
        super().__init__()
        out_channels = out_channels or in_channels // stride
        self.norm = ChannelFirstNorm(in_channels)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, (1, stride), stride=(1, stride))

    @property
    def latency(self) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B?, C_in, T, F) -> (B?, C_out, T, F * stride)"""
        return self.conv(self.norm(x))


# ---------------------------------------------------------------------------
# Self-test: shapes, layer-scale wiring, strict time causality.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, C, T, F, K = 2, 4, 50, 32, 3

    # Block: frequency preserved (same-padded), time valid.
    x = torch.randn(B, C, T, F)
    blk = ConvNeXtBlock(C, kernel_size=K)
    assert blk(x).shape == (B, C, T - K + 1, F), blk(x).shape
    assert blk.latency == K - 1
    assert blk(x[0]).shape == (C, T - K + 1, F)  # unbatched

    # layer_scale must reach the parameters: at 0 the residual branch is dead
    # and the block is exactly the time-crop identity.
    ident = ConvNeXtBlock(C, kernel_size=K, layer_scale=0.0)
    assert torch.equal(ident(x), x[..., ident.latency:, :]), "layer_scale not applied"
    live = ConvNeXtBlock(C, kernel_size=K, layer_scale=0.1)
    assert not torch.equal(live(x), x[..., live.latency:, :])

    # Down/Upsample: pure frequency pooling, channels mirror, no time change.
    down, up = Downsample(C), Upsample(2 * C)
    y = down(x)
    assert y.shape == (B, 2 * C, T, F // 2), y.shape
    assert up(y).shape == (B, C, T, F), up(y).shape
    assert down(x[0]).shape == (2 * C, T, F // 2)

    # Causality through a full stage chain: perturb the input from frame t on
    # -> outputs before t - total latency are bit-identical.
    chain = nn.Sequential(ConvNeXtBlock(C, kernel_size=K), Downsample(C),
                          ConvNeXtBlock(2 * C, kernel_size=K), Upsample(2 * C),
                          ConvNeXtBlock(C, kernel_size=K))
    lat = sum(m.latency for m in chain)
    y = chain(x)
    t = 29
    x2 = x.clone()
    x2[..., t:, :] += torch.randn(B, C, T - t, F)
    y2 = chain(x2)
    assert torch.equal(y[..., :t - lat, :], y2[..., :t - lat, :]), \
        "future leaked into the past"
    assert not torch.allclose(y[..., t - lat:, :], y2[..., t - lat:, :]), \
        "perturbation had no effect"

    print(f"tcn selftest OK: chain latency {lat} frames, causality holds at t={t}")


if __name__ == "__main__":
    _selftest()

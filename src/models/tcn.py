"""
Everything here follows the tensor shape contract `(B?, T, C)` where:

- B: optional batch size
- T: time axis
- C: channels

Note: This is the transpose of the usual CNN shape convention `(B, C, T)`.

Causal TCN building blocks.

Everything here is strictly causal in the time axis (dim -2):
output frame t depends only on input frames <= t, which the
streaming requirement makes non-negotiable. The self-test enforces it.
"""

import torch
import torch.nn as nn

class ManualConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deltas: list[int]):
        super().__init__()
        self._deltas = tuple(deltas)
        self._latency = max(deltas)
        self.lin = nn.Linear(in_channels * len(self._deltas), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B?, T, C_in) -> (B?, T - latency, C_out)"""
        xc = x.contiguous() # (B?, T, C): GEMM-friendly (no-op if already contiguous)
        gathered = torch.cat( # (B?, T - latency, K*C_in)
            [xc[..., self._latency - d : x.shape[-2] - d, :] for d in self._deltas],
            dim=-1)
        return self.lin(gathered)

    @property
    def latency(self):
        return self._latency

class TorchConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        self._latency = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C_in) -> (B, T - latency, C_out)"""
        return super().forward(x.transpose(-1, -2)).transpose(-1, -2)

    @property
    def latency(self):
        return self._latency

class DepthwiseConv1d(nn.Conv1d):
    """Valid depthwise conv (groups == channels): mixes time only, per channel;
    channel mixing is left to the block's MLP."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        self._latency = (kernel_size - 1) * dilation
        super().__init__(channels, channels, kernel_size, dilation=dilation, groups=channels)

    @property
    def latency(self):
        return self._latency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(-1, -2)).transpose(-1, -2)


class TCNBlock(nn.Module):
    """ConvNeXt-style block (the Vocos backbone): depthwise conv mixes time,
    LayerNorm + pointwise MLP mix channels, layer-scale keeps the branch
    near-identity at init so deep stacks train stably."""

    def __init__(self, channels: int, bottleneck: int | None = None,
                 kernel: int = 7, layer_scale: float = 1e-2):
        super().__init__()
        bottleneck = bottleneck or 4 * channels
        self.time_conv = DepthwiseConv1d(channels, kernel)
        self.norm = nn.LayerNorm(channels)
        self.up_proj = nn.Linear(channels, bottleneck)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(bottleneck, channels)
        with torch.no_grad():
            self.down_proj.weight *= layer_scale
            self.down_proj.bias *= layer_scale

    @property
    def latency(self):
        return self.time_conv.latency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        add = self.down_proj(self.act(self.up_proj(self.norm(self.time_conv(x)))))
        return x[..., self.latency:, :] + add

# ---------------------------------------------------------------------------
# Self-test: shapes + strict causality.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, T, C, K = 2, 100, 32, 5

    # ManualConv1d with contiguous deltas must equal a plain valid Conv1d.
    # Conv term k multiplies x[i + k]; manual tap j multiplies x[i + latency - j],
    # so kernel index k maps to tap j = K - 1 - k.
    manual = ManualConv1d(C, C, list(range(K)))
    conv = TorchConv1d(C, C, K)
    with torch.no_grad():
        w = manual.lin.weight.reshape(C, K, C)   # (C_out, tap j, C_in)
        for j in range(K):
            conv.weight[:, :, K - 1 - j] = w[:, j]
        conv.bias.copy_(manual.lin.bias)
    x = torch.randn(B, T, C)
    ym, yc = manual(x), conv(x)
    assert ym.shape == yc.shape == (B, T - K + 1, C), (ym.shape, yc.shape)
    assert (ym - yc).abs().max() < 1e-5, (ym - yc).abs().max()
    assert manual.latency == conv.latency == K - 1

    # Unbatched (T, C) input works too.
    assert manual(x[0]).shape == (T - K + 1, C)

    # layer_scale must actually reach the parameters: at 0 the residual branch
    # is dead and the block is exactly the crop-identity.
    ident = TCNBlock(C, kernel=3, layer_scale=0.0)
    assert torch.equal(ident(x), x[..., ident.latency:, :]), "layer_scale not applied"
    live = TCNBlock(C, kernel=3, layer_scale=0.1)
    assert not torch.equal(live(x), x[..., live.latency:, :])

    # Stacked blocks: time shrinks by the summed latency.
    blocks = nn.Sequential(*[TCNBlock(C, kernel=3) for _ in range(3)])
    lat = sum(b.latency for b in blocks)
    y = blocks(x)
    assert y.shape == (B, T - lat, C), y.shape

    # Causality: output frame i covers input frames [i, i + lat]. Perturb the
    # input from frame t on -> outputs before t - lat must be bit-identical.
    t = 57
    x2 = x.clone()
    x2[..., t:, :] += torch.randn(B, T - t, C)
    y2 = blocks(x2)
    assert torch.equal(y[..., :t - lat, :], y2[..., :t - lat, :]), \
        "future leaked into the past"
    assert not torch.allclose(y[..., t - lat:, :], y2[..., t - lat:, :]), \
        "perturbation had no effect"

    print(f"tcn selftest OK: manual == Conv1d, 3-block stack latency {lat}, "
          f"causality holds at t={t}")


if __name__ == "__main__":
    _selftest()

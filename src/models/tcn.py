"""
Causal TCN building blocks. Everything here is strictly causal in the time
axis (dim 2): output frame t depends only on input frames <= t, which the
streaming requirement makes non-negotiable. The self-test enforces it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """Conv1d padded on the left only, so frame t never sees frames > t."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel, dilation=dilation)
        self._left_pad = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self._left_pad, 0)))


class ChannelNorm(nn.Module):
    """LayerNorm over channels only, each time step independently.

    BatchNorm/GroupNorm normalize across the time axis too, mixing future
    frames into past statistics -- not streamable. This is.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    """Pre-norm residual block: norm -> dilated causal conv -> GELU -> 1x1."""

    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.conv = CausalConv1d(channels, channels, kernel, dilation)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.act(self.conv(self.norm(x))))


class TCN(nn.Module):
    """Stack of TCNBlocks with dilations 1, g, g^2, ... (B, C, T) -> (B, C, T)."""

    def __init__(self, channels: int, blocks: int = 6, kernel: int = 3,
                 dilation_growth: int = 2):
        super().__init__()
        dilations = [dilation_growth ** i for i in range(blocks)]
        self.blocks = nn.Sequential(
            *[TCNBlock(channels, kernel, d) for d in dilations])
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)  # in frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Self-test: shapes + strict causality.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, C, T = 2, 32, 100
    tcn = TCN(C, blocks=4, kernel=3).eval()

    x = torch.randn(B, C, T)
    y = tcn(x)
    assert y.shape == (B, C, T), y.shape

    # Causality: perturb the input from frame `t` on; outputs before `t`
    # must be bit-identical.
    t = 57
    x2 = x.clone()
    x2[..., t:] += torch.randn(B, C, T - t)
    y2 = tcn(x2)
    assert torch.equal(y[..., :t], y2[..., :t]), "future leaked into the past"
    assert not torch.allclose(y[..., t:], y2[..., t:]), "perturbation had no effect"

    print(f"tcn selftest OK: receptive field {tcn.receptive_field} frames, "
          f"causality holds at t={t}")


if __name__ == "__main__":
    _selftest()

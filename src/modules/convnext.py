import torch
from torch import nn, Tensor

class FeedForward(nn.Module):
    def __init__(self,
                 channels: int,
                 hidden: int,
                 scale: float):
        super().__init__()
        self.up_down = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels)
        )
        with torch.no_grad():
            self.up_down[-1].bias *= scale
            self.up_down[-1].weight *= scale

    def forward(self, x: Tensor) -> Tensor:
        return self.up_down(x)

class ConvNeXt2D(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel: tuple[int, int] | int = 7,
                 hidden: int | None = None,
                 scale: float = 1e-6):
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if hidden is None:
            hidden = 3 * channels
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel, padding=(kernel[0] // 2, 0), groups=channels)
        self.ffwd = FeedForward(channels, hidden, scale)
        self.latency = kernel[1] - 1

    def forward(self, x: Tensor) -> Tensor:
        """(B?, T, F, K)"""
        return x[..., self.latency:, :, :] + self.ffwd(self.dw_conv(x.movedim(-1, -3)).movedim(-3, -1))

class ConvNeXt1D(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel: int = 7,
                 hidden: int | None = None,
                 scale: float = 1e-6):
        if hidden is None:
            hidden = 3 * channels
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel, groups=channels)
        self.ffwd = FeedForward(channels, hidden, scale)
        self.latency = kernel - 1

    def forward(self, x: Tensor) -> Tensor:
        """(B?, T, K)"""
        return x[..., self.latency:, :] + self.ffwd(self.dw_conv(x.movedim(-2, -1)).movedim(-1, -2))

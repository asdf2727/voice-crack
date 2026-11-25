from sympy.printing.pytorch import torch
from torch import nn, Tensor

class GLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU())
        self.value = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.value(x).mul(self.gate(x))

class GatedFFN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int | None = None,
                 scale: float | None = None):
        if out_dim is None:
            out_dim = in_dim
        super().__init__()
        self.up = GLU(in_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, out_dim)
        if scale is None: return
        with torch.no_grad():
            self.down.weight *= scale
            self.down.bias *= scale

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.up(x))

class StandardFFN(nn.Sequential):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int | None = None,
                 scale: float | None = None):
        if out_dim is None:
            out_dim = in_dim
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        if scale is None: return
        with torch.no_grad():
            self[-1].weight *= scale
            self[-1].bias *= scale
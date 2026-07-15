import math
import torch
from torch import nn

_golden = (1 + 5 ** 0.5) / 2

class MLP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            squeeze: float = _golden):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        layers = []
        self.dims = [in_dim]
        hidden = round(abs(math.log(in_dim / out_dim) / math.log(squeeze)))
        for i in range(hidden):
            # round: layer widths must be ints for nn.Linear
            next_dim = round(in_dim * (out_dim / in_dim) ** ((i + 1) / (hidden + 1)))
            layers.append(nn.Linear(self.dims[-1], next_dim))
            self.dims.append(next_dim)
            layers.append(nn.GELU())
        layers.append(nn.Linear(self.dims[-1], out_dim))
        self.dims.append(out_dim)
        self.seq = nn.Sequential(*layers)
        print(self.dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
import torch
from torch import nn, Tensor

class VAE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self._latent_dim = min(in_dim, out_dim)
        self.to_latent = nn.Linear(in_dim, 2 * self._latent_dim)
        self.from_latent = nn.Linear(self._latent_dim, out_dim)

    @property
    def latent_dim(self) -> int: return self._latent_dim

    @staticmethod
    def _reparam(mean: Tensor, log_var: Tensor) -> Tensor:
        # z = mean + std * eps keeps z differentiable w.r.t. mean/log_var while
        # the randomness lives in eps, which needs no gradient.
        return mean + (0.5 * log_var).exp() * torch.randn_like(mean)

    def split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_var = self.to_latent(x).chunk(2, dim=-1)
        return mean, log_var

    def sample(self, mean: Tensor, log_var: Tensor) -> Tensor:
        return self.from_latent(self._reparam(mean, log_var))

    def forward(self, x: Tensor) -> Tensor:
        return self.sample(*self.split(x))

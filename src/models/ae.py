"""
Causal TCN autoencoder over STFT feature maps with a variational bottleneck.

Encoder: 1x1 in-projection -> causal TCN -> per-frame mean/log_var heads.
Decoder: mirror -- 1x1 latent projection -> causal TCN -> 1x1 out-projection.

The bottleneck is per-frame: z is (B, latent_dim, T), one L-dim Gaussian per
STFT frame. Training samples via the reparameterization trick; eval uses
z = mean (deterministic). The ARD prior over the latent axes lives in
loss.ard_prior -- the model itself is a plain diagonal-Gaussian VAE.
"""
from typing import NamedTuple

import torch
import torch.nn as nn

from models.tcn import TCN


class VAEOutput(NamedTuple):
    recon: torch.Tensor    # (B, in_ch, T)
    mean: torch.Tensor     # (B, latent_dim, T)
    log_var: torch.Tensor  # (B, latent_dim, T)
    z: torch.Tensor        # (B, latent_dim, T)


class TCNAutoencoder(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int = 64, hidden: int = 384,
                 blocks: int = 6, kernel: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_in = nn.Conv1d(in_ch, hidden, 1)
        self.enc_tcn = TCN(hidden, blocks, kernel)
        self.mean_head = nn.Conv1d(hidden, latent_dim, 1)
        self.log_var_head = nn.Conv1d(hidden, latent_dim, 1)
        self.dec_in = nn.Conv1d(latent_dim, hidden, 1)
        self.dec_tcn = TCN(hidden, blocks, kernel)
        self.dec_out = nn.Conv1d(hidden, in_ch, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_tcn(self.enc_in(x))
        # Clamp keeps exp(log_var) finite while the fresh heads are still wild;
        # [-12, 6] is std in [~2.5e-3, ~20], far wider than anything useful.
        return self.mean_head(h), self.log_var_head(h).clamp(-12.0, 6.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # z = mean + std * eps keeps z differentiable w.r.t. mean/log_var while
        # the randomness lives in eps, which needs no gradient.
        return mean + (0.5 * log_var).exp() * torch.randn_like(mean)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_out(self.dec_tcn(self.dec_in(z)))

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mean, log_var = self.encode(x)
        # Sample at train time, z = mean at inference.
        z = self.reparameterize(mean, log_var) if self.training else mean
        return VAEOutput(self.decode(z), mean, log_var, z)


# ---------------------------------------------------------------------------
# Self-test: shapes, train/eval sampling behavior, gradient flow.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, F2, T, L = 2, 66, 40, 16
    model = TCNAutoencoder(in_ch=F2, latent_dim=L, hidden=48, blocks=3)

    x = torch.randn(B, F2, T)
    out = model(x)
    assert out.recon.shape == (B, F2, T), out.recon.shape
    assert out.mean.shape == out.log_var.shape == out.z.shape == (B, L, T)

    # train mode samples (z != mean), eval mode is deterministic (z == mean).
    assert not torch.equal(out.z, out.mean)
    model.eval()
    out_e = model(x)
    assert torch.equal(out_e.z, out_e.mean)
    assert torch.equal(model(x).recon, out_e.recon)

    # Both heads must receive gradient through the reparameterized sample.
    model.train()
    out = model(x)
    out.recon.square().sum().backward()
    assert model.mean_head.weight.grad is not None
    assert model.log_var_head.weight.grad.abs().sum() > 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ae selftest OK: {n_params/1e3:.0f}K params, "
          f"encoder receptive field {model.enc_tcn.receptive_field} frames")


if __name__ == "__main__":
    _selftest()

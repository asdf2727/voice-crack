"""
Everything here follows the tensor shape contract `(B?, T, C)` where:

- B: optional batch size
- T: time axis
- C: channels

Note: This is the transpose of the usual CNN shape convention `(B, C, T)`.

Causal TCN autoencoder over STFT feature maps with a variational bottleneck.

Encoder: linear in-projection -> causal TCN stack -> per-frame mean/log_var heads.
Decoder: mirror -- linear latent projection -> causal TCN stack -> linear out-projection.

The bottleneck is per-frame: z is (B, T, L), one L-dim Gaussian per
STFT frame. Training samples via the reparameterization trick; eval uses
z = mean (deterministic). The ARD prior over the latent axes lives in
loss.ard_prior -- the model itself is a plain diagonal-Gaussian VAE.
"""
import torch
import torch.nn as nn

from models.tcn import TCNBlock

class TCNEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int | None = None, latent_dim: int = 64,
                 blocks: int = 6, kernel: int = 3):
        super().__init__()
        hidden = hidden or in_ch
        self.latent_dim = latent_dim
        self.enc_in = nn.Linear(in_ch, hidden)
        self.enc_tcn = nn.Sequential(
            *[TCNBlock(hidden, kernel=kernel, layer_scale=1/blocks) for _ in range(blocks)]
        )
        self.mean_head = nn.Linear(hidden, latent_dim)
        self.log_var_head = nn.Linear(hidden, latent_dim)

    @property
    def latency(self) -> int:
        return sum(block.latency for block in self.enc_tcn)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.encode(x)
        # Sample at train time, z = mean at inference.
        return self.reparameterize(mean, log_var) if self.training else mean

class TCNDecoder(nn.Module):
    def __init__(self, out_ch: int, latent_dim: int = 64, hidden: int | None = None,
                 blocks: int = 6, kernel: int = 3):
        super().__init__()
        hidden = hidden or out_ch
        self.latent_dim = latent_dim
        self.dec_in = nn.Linear(latent_dim, hidden)
        self.dec_tcn = nn.Sequential(
            *[TCNBlock(hidden, kernel=kernel, layer_scale=1/blocks) for _ in range(blocks)]
        )
        self.dec_out = nn.Linear(hidden, out_ch)

    @property
    def latency(self) -> int:
        return sum(block.latency for block in self.dec_tcn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_out(self.dec_tcn(self.dec_in(z)))

# ---------------------------------------------------------------------------
# Self-test: shapes, train/eval sampling behavior, gradient flow.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, T, F2, L = 2, 60, 66, 16
    enc = TCNEncoder(F2, hidden=48, latent_dim=L, blocks=3, kernel=3)
    dec = TCNDecoder(F2, latent_dim=L, hidden=48, blocks=3, kernel=3)

    # Valid convs eat frames off the front: recon frame 0 corresponds to
    # input frame enc.latency + dec.latency (the reconstruction target must
    # be cropped accordingly in the training loop).
    x = torch.randn(B, T, F2)
    mean, log_var = enc.encode(x)
    assert mean.shape == log_var.shape == (B, T - enc.latency, L), mean.shape
    recon = dec(mean)
    assert recon.shape == (B, T - enc.latency - dec.latency, F2), recon.shape

    # train mode samples per call, eval mode returns z = mean deterministically.
    enc.train()
    assert not torch.equal(enc(x), enc(x))
    enc.eval()
    assert torch.equal(enc(x), mean)

    # Both heads must receive gradient through the reparameterized sample.
    enc.train()
    dec(enc(x)).square().sum().backward()
    assert enc.mean_head.weight.grad is not None
    assert enc.log_var_head.weight.grad.abs().sum() > 0

    n_params = sum(p.numel() for m in (enc, dec) for p in m.parameters())
    print(f"ae selftest OK: {n_params/1e3:.0f}K params, "
          f"latency enc {enc.latency} + dec {dec.latency} frames")


if __name__ == "__main__":
    _selftest()

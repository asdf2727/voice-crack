"""
Causal ConvNeXt autoencoder over complex STFT maps with a variational
bottleneck.

Spectra are (B?, 2, T, F) channel maps (STFTEncoder.to_real). The encoder is
a ConvNeXt trunk: pointwise stem to `base` channels, then `depths[i]` blocks
per stage with a dedicated Downsample between stages (F // stride, C * stride
each transition). A final LayerNorm + Linear maps each frame's flattened
(C_last, F_last) stack to the per-frame feature vector the (mean, log_var)
heads consume. The decoder mirrors it: Linear from the latent onto the coarse
grid, blocks with Upsample transitions, pointwise head back to 2 channels.

F is floor-divided per stage, so the decoder rebuilds freq_out =
(F // stride^(stages-1)) * stride^(stages-1) <= F bins -- crop the loss
target to match (at most stride^(stages-1) - 1 top bins are lost).

The bottleneck is per-frame: z is (B?, T, L), one L-dim Gaussian per frame.
The ARD prior over the latent axes lives in loss.vae_loss.
"""
import torch
import torch.nn as nn

from models.tcn import ConvNeXtBlock, Downsample, Upsample

class TCNEncoder(nn.Module):
    def __init__(
            self,
            in_freq: int,
            latent_dim: int = 64,
            depths: tuple[int, ...] = (2, 2, 2),
            base: int = 8,
            kernel: int | tuple[int, int] = 7,
            stride: int = 2):
        super().__init__()
        stages: list[nn.Module] = [nn.Conv2d(2, base, 1)]
        dims = base
        for depth in depths:
            stages += [ConvNeXtBlock(dims, kernel) for _ in range(depth)]
            stages.append(Downsample(dims, stride=stride))
            dims *= stride
        stages.pop()
        dims //= stride
        self.enc_tcn = nn.Sequential(*stages)

        feats = in_freq // (stride ** (len(depths) - 1)) * dims
        self.mean_head = nn.Linear(feats, latent_dim)
        self.log_var_head = nn.Linear(feats, latent_dim)

    @property
    def latency(self) -> int:
        return sum(getattr(m, "latency", 0) for m in self.enc_tcn)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_tcn(x)                # (B?, C_last, T', F_last)
        h = h.movedim(-3, -1).flatten(-2)  # per-frame features
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
        return self.reparameterize(mean, log_var)


class TCNDecoder(nn.Module):
    def __init__(
            self,
            out_freq: int,
            latent_dim: int = 64,
            depths: tuple[int, ...] = (2, 2, 2),
            base: int = 8,
            kernel: int | tuple[int, int] = 7,
            stride: int = 2):
        super().__init__()
        upscale = stride ** (len(depths) - 1)
        dims = base * upscale
        self._c0 = dims
        self._f0 = out_freq // upscale
        self.out_freq = self._f0 * upscale
        self.dec_in = nn.Linear(latent_dim, self._f0 * self._c0)

        stages: list[nn.Module] = []
        for depth in reversed(depths):
            stages += [ConvNeXtBlock(dims, kernel) for _ in range(depth)]
            stages.append(Upsample(dims, stride=stride))
            dims //= stride
        stages.pop()
        dims *= stride

        stages.append(nn.Conv2d(base, 2, 1))
        self.dec_tcn = nn.Sequential(*stages)

    @property
    def latency(self) -> int:
        return sum(getattr(m, "latency", 0) for m in self.dec_tcn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z)                                   # (B?, T, C0*F0)
        h = h.unflatten(-1, (-1, self._c0)).movedim(-1, -3)  # (B?, C0, T, F0)
        return self.dec_tcn(h)                               # (B?, 2, T', freq_out)


# ---------------------------------------------------------------------------
# Self-test: shapes (incl. non-divisible F), sampling behavior, gradient flow.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, T, F, L = 2, 40, 33, 16  # odd F: exercises the floor/crop path
    enc = TCNEncoder(F, latent_dim=L, depths=(1, 2), base=4, kernel=3)
    dec = TCNDecoder(F, latent_dim=L, depths=(1, 2), base=4, kernel=3)

    # Valid time convs eat frames off the front: recon frame 0 corresponds to
    # input frame enc.latency + dec.latency.
    x = torch.randn(B, 2, T, F)
    mean, log_var = enc.encode(x)
    assert mean.shape == log_var.shape == (B, T - enc.latency, L), mean.shape
    recon = dec(mean)
    assert 0 < dec.out_freq <= F, dec.out_freq
    assert recon.shape == (B, 2, T - enc.latency - dec.latency, dec.out_freq), recon.shape

    # Unbatched path end to end.
    assert dec(enc.encode(x[0])[0]).shape == (2, T - enc.latency - dec.latency, dec.out_freq)

    # encode is deterministic; forward always samples (reparameterization on,
    # collapsed dims double as the decoder's noise source).
    assert torch.equal(enc.encode(x)[0], mean)
    assert not torch.equal(enc(x), enc(x))

    # Both heads must receive gradient through the reparameterized sample.
    dec(enc(x)).square().sum().backward()
    assert enc.mean_head.weight.grad is not None
    assert enc.log_var_head.weight.grad.abs().sum() > 0

    n_params = sum(p.numel() for m in (enc, dec) for p in m.parameters())
    print(f"ae selftest OK: {n_params/1e3:.0f}K params, "
          f"latency enc {enc.latency} + dec {dec.latency} frames, "
          f"freq {F} -> {dec.out_freq}")


if __name__ == "__main__":
    _selftest()

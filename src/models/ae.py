"""
Causal ConvNeXt autoencoder over complex STFT maps with a variational
bottleneck.

Spectra are (B?, 2, T, F) channel maps (STFTEncoder.to_real). One
list[BlockParams] defines both sides of a symmetric AE: the encoder runs
Downsample (F // stride, -> channels) then `depth` ConvNeXt blocks per entry
(the first Downsample doubles as the stem from the 2 re/im channels), and a
final Linear maps each frame's flattened (C_last, F_last) stack to
(mean, log_var). The decoder mirrors the list in reverse.

F must be divisible by the product of block strides -- crop the spectrum to
the nearest multiple before the model (spectra have n_fft//2 + 1 bins, so
this drops the Nyquist end: harmless).

The bottleneck is per-frame: z is (B?, T, L), one L-dim Gaussian per frame.
The ARD prior over the latent axes lives in loss.vae_loss.
"""

import math
import sys

import torch
import torch.nn as nn

from models.mlp import MLP
from models.tcn import ConvNeXtBlock, Downsample, Upsample

class BlockParams:
    def __init__(self, channels: int, depth: int = 1, kernel: int | tuple[int, int] = 5, stride: int = 2):
        self.channels = channels
        self.depth = depth
        if isinstance(kernel, int): kernel = (kernel, kernel)
        self.kernel = kernel
        self.stride = stride

class TCNEncoder(nn.Module):
    def __init__(
            self,
            in_freq: int,
            blocks: list[BlockParams],
            latent_dim: int = 64):
        full_down = math.prod(b.stride for b in blocks)
        out_freq = in_freq // full_down
        self.in_freq = out_freq * full_down
        if in_freq != self.in_freq:
            print(f"cropping {in_freq} -> {self.in_freq} for input freq bin count", file=sys.stderr)

        super().__init__()
        stages: list[nn.Module] = []
        prev_channels = 2
        scale = 1 / sum(block.depth for block in blocks)
        for block in blocks:
            stages = (stages +
                      [Downsample(prev_channels, block.channels, block.stride)] +
                      [ConvNeXtBlock(block.channels, block.kernel, layer_scale=scale) for _ in range(block.depth)])
            prev_channels = block.channels
        self.enc_tcn = nn.Sequential(*stages)

        feats = out_freq * prev_channels
        self.enc_mlp = MLP(feats, latent_dim * 2)

    @property
    def latency(self) -> int:
        return sum(getattr(m, "latency", 0) for m in self.enc_tcn)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_tcn(x)                # (B?, C_last, T', F_last)
        h = h.movedim(-3, -1).flatten(-2)  # per-frame features
        z = self.enc_mlp(h)
        mean, log_var = torch.split(z, self.latent_dim, dim=-1)
        # Clamp keeps exp(log_var) finite while the fresh heads are still wild;
        # [-12, 6] is std in [~2.5e-3, ~20], far wider than anything useful.
        return mean, log_var.clamp(-12.0, 6.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # z = mean + std * eps keeps z differentiable w.r.t. mean/log_var while
        # the randomness lives in eps, which needs no gradient.
        return mean + (0.5 * log_var).exp() * torch.randn_like(mean)

    @property
    def latent_dim(self) -> int: return self.enc_mlp.out_dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.encode(x)
        return self.reparameterize(mean, log_var)


class TCNDecoder(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            blocks: list[BlockParams],
            out_freq: int):
        full_up = math.prod(b.stride for b in reversed(blocks))
        in_freq = out_freq // full_up
        self.out_freq = in_freq * full_up
        if out_freq != self.out_freq:
            print(f"cropping {out_freq} -> {self.out_freq} for output freq bin count", file=sys.stderr)

        super().__init__()
        stages: list[nn.Module] = []
        next_channels = 2
        scale = 1 / sum(block.depth for block in blocks)
        for block in blocks:
            stages = ([ConvNeXtBlock(block.channels, block.kernel, layer_scale=scale) for _ in range(block.depth)] +
                      [Upsample(block.channels, next_channels, block.stride)] +
                      stages)
            next_channels = block.channels
        self.dec_tcn = nn.Sequential(*stages)

        self._in_channels = next_channels
        feats = in_freq * next_channels
        self.dec_mlp = MLP(latent_dim, feats)

    @property
    def latency(self) -> int:
        return sum(getattr(m, "latency", 0) for m in self.dec_tcn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_mlp(z)                                           # (B?, T, C0*F0)
        h = h.unflatten(-1, (-1, self._in_channels)).movedim(-1, -3)  # (B?, C0, T, F0)
        return self.dec_tcn(h)                                        # (B?, 2, T', freq_out)


# ---------------------------------------------------------------------------
# Self-test: symmetric shapes, divisibility guard, sampling, gradient flow.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, T, F, L = 2, 40, 32, 16
    blocks = [
        BlockParams(4, depth=1, kernel=3),
        BlockParams(8, depth=2, kernel=3)]
    enc = TCNEncoder(F, blocks, latent_dim=L)
    dec = TCNDecoder(L, blocks, F)
    assert enc.latent_dim == L

    # Valid time convs eat frames off the front: recon frame 0 corresponds to
    # input frame enc.latency + dec.latency. Frequency is fully symmetric.
    x = torch.randn(B, 2, T, F)
    mean, log_var = enc.encode(x)
    assert mean.shape == log_var.shape == (B, T - enc.latency, L), mean.shape
    recon = dec(mean)
    assert recon.shape == (B, 2, T - enc.latency - dec.latency, F), recon.shape

    # Unbatched path end to end.
    assert dec(enc.encode(x[0])[0]).shape == (2, T - enc.latency - dec.latency, F)

    # Non-divisible F auto-crops to the nearest stride multiple (Nyquist end).
    assert TCNEncoder(33, blocks, latent_dim=L).in_freq == 32
    assert TCNDecoder(L, blocks, 33).out_freq == 32

    # encode is deterministic; forward always samples (reparameterization on,
    # collapsed dims double as the decoder's noise source).
    assert torch.equal(enc.encode(x)[0], mean)
    assert not torch.equal(enc(x), enc(x))

    # The shared head must receive gradient through the reparameterized sample.
    dec(enc(x)).square().sum().backward()
    assert enc.enc_mlp.seq[-1].weight.grad is not None
    assert enc.enc_mlp.seq[-1].weight.grad.abs().sum() > 0

    n_params = sum(p.numel() for m in (enc, dec) for p in m.parameters())
    print(f"ae selftest OK: {n_params/1e3:.0f}K params, "
          f"latency enc {enc.latency} + dec {dec.latency} frames, freq {F} -> {F}")


if __name__ == "__main__":
    _selftest()

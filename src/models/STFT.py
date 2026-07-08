"""
Everything here follows the tensor shape contract `(B?, T, C)` where:

- B: optional batch size
- T: time axis
- C: channels

Note: This is the transpose of the usual CNN shape convention `(B, C, T)`.
"""

from typing import Callable

import torch
import torch.nn as nn


def rms_normalize(wave: torch.Tensor, target: float = 0.1,
                  min_rms: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale each waveform (..., S) to RMS `target`; undo with `out / gain`.
    min_rms caps the boost so near-silence isn't amplified to speech level."""
    rms = wave.square().mean(dim=-1, keepdim=True).sqrt()
    gain = target / rms.clamp_min(min_rms)
    return wave * gain, gain


class STFTEncoder(nn.Module):
    def __init__(self, hop: int, win_chunks: int = 1,
                 window: Callable[[int], torch.Tensor] = torch.hamming_window):
        super().__init__()
        self.hop = hop
        self.n_fft = win_chunks * hop
        self.register_buffer("window", window(self.n_fft))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x,
            self.n_fft,
            self.hop,
            self.n_fft,
            self.window,
            center=False,
            onesided=True,
            return_complex=True
        )

    @staticmethod
    def to_real(x: torch.Tensor) -> torch.Tensor:
        """complex (B?, F, T) -> real (B?, T, 2F), re/im interleaved per bin."""
        return torch.view_as_real(x.transpose(-1, -2)).flatten(-2)

    @staticmethod
    def compress(c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Magnitude r -> log1p(r), phase kept. Slope 1 at r = 0 (silent bins
        map to ~themselves, no noise-floor amplification like power < 1) and
        log at the top (dynamic-range squash); knee at r ~ 1."""
        mag = c.abs()
        return c * mag.log1p() / (mag + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_real(self.compress(self.stft(x)))

class STFTDecoder(nn.Module):
    def __init__(self, enc: STFTEncoder):
        super().__init__()
        self.hop = enc.hop
        self.n_fft = enc.n_fft
        self.register_buffer("window", enc.window.clone())

    @staticmethod
    def to_complex(x: torch.Tensor) -> torch.Tensor:
        """real (B?, T, 2F) -> complex (B?, F, T). The dtype view must happen
        while the interleaved re/im pairs sit contiguously in the last dim --
        transposing first would pair floats along the wrong axis."""
        return x.view(torch.cfloat).transpose(-1, -2)

    @staticmethod
    def decompress(c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Exact inverse of STFTEncoder.compress: magnitude m -> expm1(m)."""
        mag = c.abs()
        return c * mag.expm1() / (mag + eps)

    def istft(self, x: torch.Tensor, length: int | None = None) -> torch.Tensor:
        return torch.istft(
            x,
            self.n_fft,
            self.hop,
            self.n_fft,
            self.window,
            center=False,
            onesided=True,
            length=length
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.istft(self.decompress(self.to_complex(x)))

from matplotlib import pyplot as plt
import numpy as np

def show_spec(spec: torch.Tensor):
    spec_np = STFTDecoder.to_complex(spec).numpy(force=True)
    mag = np.abs(spec_np)
    phase = np.angle(spec_np)
    sin_p = (np.sin(phase) + 1) / 2
    cos_p = (np.cos(phase) + 1) / 2
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min())
    phase_rgb = np.stack([sin_p * mag_norm, mag_norm, cos_p * mag_norm], axis=-1)

    h, w = phase_rgb.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(phase_rgb, origin="lower", interpolation="none")
    plt.show()


# ---------------------------------------------------------------------------
# Self-test: shapes, NaN-safety on silence, exact round-trips.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    sr = 16000
    # hamming, not the default hann: center=False istft needs a strictly
    # positive overlap-add envelope, and hann's zero endpoints violate it.
    enc = STFTEncoder(hop=128, win_chunks=4, window=torch.hamming_window)  # n_fft 512
    dec = STFTDecoder(enc)

    t = torch.arange(sr) / sr
    wave = torch.stack([0.5 * torch.sin(2 * torch.pi * 440.0 * t),
                        0.1 * torch.randn(sr)])

    spec = enc(wave)
    bins = enc.n_fft // 2 + 1
    n_frames = (sr - enc.n_fft) // enc.hop + 1
    assert spec.shape == (2, n_frames, 2 * bins), spec.shape
    assert spec.dtype == torch.float32

    # Silence must not produce NaNs (compression at mag = 0).
    assert not enc(torch.zeros(1, sr)).isnan().any(), "NaN on silent input"

    # to_real / to_complex are exact inverses (pure layout changes).
    c = enc.stft(wave)
    assert torch.equal(dec.to_complex(enc.to_real(c)), c)

    # compress / decompress round-trip.
    cc = dec.decompress(enc.compress(c))
    assert (cc - c).abs().max() < 1e-3 * c.abs().max(), (cc - c).abs().max()

    # Full wave round-trip; istft(center=False) yields n_fft + (T-1)*hop samples.
    n = enc.n_fft + (n_frames - 1) * enc.hop
    err = (dec(spec) - wave[:, :n]).abs().max().item()
    assert err < 1e-3, err

    # Unbatched (S,) path works end to end.
    assert dec(enc(wave[0])).shape == (n,)

    # Normalization: rows land on the target RMS, near-silence isn't boosted.
    norm, gain = rms_normalize(wave)
    assert (norm.square().mean(-1).sqrt() - 0.1).abs().max() < 1e-4
    assert rms_normalize(torch.full((100,), 1e-6))[1].item() <= 100.0

    print(f"stft selftest OK: {tuple(wave.shape)} -> {tuple(spec.shape)} "
          f"-> roundtrip max err {err:.2e}")


if __name__ == "__main__":
    _selftest()
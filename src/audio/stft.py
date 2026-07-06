"""
Complex-STFT frontend for the autoencoder.

Waveform (B, S) <-> real feature map (B, 2F, T): the complex STFT is power-law
compressed (magnitude^power, phase kept), then real and imaginary parts are
stacked along the channel axis. Compression flattens the ~40 dB dynamic range
of speech so an MSE reconstruction loss doesn't spend everything on the few
loudest bins; the map is exactly invertible, so `inverse` gives listenable
audio back.

Uses center=True padding for a clean istft round-trip; that leaks half a
window (n_fft/2 samples) of lookahead at the STFT level, which is a latency
cost, not a training-correctness issue -- the TCN itself stays causal in
frames. Revisit when wiring the streaming path.
"""
import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, n_fft: int = 1024, hop: int = 256,
                 power: float = 0.3, eps: float = 1e-8):
        super().__init__()
        self.n_fft, self.hop, self.power, self.eps = n_fft, hop, power, eps
        # Buffer (not parameter): follows .to(device), never trained.
        self.register_buffer("window", torch.hann_window(n_fft))

    @property
    def bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def channels(self) -> int:
        """Channel count of the feature map fed to the model: 2F (re + im)."""
        return 2 * self.bins

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, S) float waveform -> (B, 2F, T) compressed complex STFT."""
        spec = torch.stft(wave, self.n_fft, self.hop, window=self.window,
                          center=True, return_complex=True)
        # mag^power * e^(i*phase)  ==  spec * mag^(power-1)
        comp = spec * (spec.abs() + self.eps).pow(self.power - 1.0)
        return torch.cat([comp.real, comp.imag], dim=1)

    def inverse(self, feats: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """(B, 2F, T) -> (B, S) waveform; pass `length` to trim istft padding."""
        F = self.bins
        comp = torch.complex(feats[:, :F], feats[:, F:])
        spec = comp * (comp.abs() + self.eps).pow(1.0 / self.power - 1.0)
        return torch.istft(spec, self.n_fft, self.hop, window=self.window,
                           center=True, length=length)


# ---------------------------------------------------------------------------
# Self-test: round-trip accuracy on synthetic audio.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    sr, secs = 16000, 1.0
    t = torch.arange(int(sr * secs)) / sr
    wave = torch.stack([
        0.5 * torch.sin(2 * torch.pi * 440.0 * t),
        (0.1 * torch.randn(len(t))).clamp(-1, 1),
    ])

    stft = STFT(n_fft=512, hop=128)
    feats = stft(wave)
    assert feats.shape[:2] == (2, stft.channels), feats.shape
    assert feats.dtype == torch.float32

    back = stft.inverse(feats, length=wave.shape[1])
    err = (back - wave).abs().max().item()
    assert err < 1e-4, err

    print(f"stft selftest OK: {tuple(wave.shape)} -> {tuple(feats.shape)} "
          f"-> roundtrip max err {err:.2e}")


if __name__ == "__main__":
    _selftest()

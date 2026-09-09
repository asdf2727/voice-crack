import torch
from torch import nn, Tensor
from typing import Callable
import numpy as np

def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi).remainder(2 * torch.pi) - torch.pi

class STFT(nn.Module):
    def __init__(self, hop: int, win_chunks: int = 1,
                 window: Callable[[int], Tensor] = torch.hamming_window):
        super().__init__()
        self.hop = hop
        self.n_fft = win_chunks * hop
        self.register_buffer("window", window(self.n_fft))

    @property
    def out_bins(self) -> int:
        return self.n_fft // 2 + 1

    def stft(self, x: Tensor) -> Tensor:
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
    def to_feats(c: Tensor, eps = 1e-8) -> Tensor:
        """complex (B?, F, T) -> real (B?, T, F, 8)"""
        mag = c.abs().clamp_min(eps)
        log_mag = mag.log1p()
        unit_c = c / mag
        phs = c.angle()
        init_phs = torch.zeros_like(phs[..., :1])
        phs_diff = _wrap_angle(phs.diff(prepend=init_phs))
        return torch.stack((
            c.real, c.imag,
            unit_c.real, unit_c.imag,
            mag, log_mag,
            phs, phs_diff),
            dim=-1).movedim(-2, -3)

    @staticmethod
    def feats_to_log_polar(f: torch.Tensor) -> tuple[Tensor, Tensor]:
        return f[..., 5], f[..., 6]

    def forward(self, x: Tensor) -> Tensor:
        return self.to_feats(self.stft(x))

class ISTFT(nn.Module):
    def __init__(self, enc: STFT):
        super().__init__()
        self.hop = enc.hop
        self.n_fft = enc.n_fft
        self.register_buffer("window", enc.window.clone())

    @property
    def in_freq(self) -> int:
        return self.n_fft // 2 + 1

    @staticmethod
    def feats_to_log_polar(f: torch.Tensor) -> tuple[Tensor, Tensor]:
        return f[..., 0], torch.atan2(f[..., 1], f[..., 2])

    @staticmethod
    def from_feats(f: torch.Tensor) -> torch.Tensor:
        """real (B?, T, F, 3) -> complex (B?, F, T)"""
        f = f.movedim(-3, -2)
        log_mag, phs = ISTFT.feats_to_log_polar(f)
        mag = log_mag.expm1()
        return torch.polar(mag, phs)

    @staticmethod
    def feats_to_hsv(f: torch.Tensor) -> np.ndarray:
        log_mag = f.select(-2, 0).numpy()
        phs = torch.atan2(f.select(-2, 1), f.select(-2, 2))
        init_phs = torch.zeros_like(phs[..., :1])
        phs_diff = _wrap_angle(phs.diff(prepend=init_phs)).numpy()
        h = ((phs_diff + np.pi) / (2 * np.pi) + 0.5) % 1
        s = np.ones_like(h)
        v = log_mag / log_mag.max().item()
        return np.stack([h, s, v], axis=-1)

    def istft(self, x: Tensor, length: int | None = None) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        return self.istft(self.from_feats(x))

from matplotlib import pyplot, colors

def show_hsv(hsv: np.ndarray):
    rgb = colors.hsv_to_rgb(hsv)
    h, w = rgb.shape[:2]
    dpi = 100
    fig = pyplot.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(rgb, origin="lower", interpolation="none")
    pyplot.show()

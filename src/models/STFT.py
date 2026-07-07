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

class STFTEncoder(nn.Module):
    def __init__(self, hop: int, win_chunks: int = 1,
                 window: Callable[[int], torch.Tensor] = torch.hann_window):
        super().__init__()
        self.hop = hop
        self.n_fft = win_chunks * hop
        self.register_buffer("window", window(self.n_fft))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(torch.stft(
            x,
            self.n_fft,
            self.hop,
            self.n_fft,
            self.window,
            center=False,
            onesided=True,
            return_complex=True
        ).transpose(-1, -2))

class STFTDecoder(nn.Module):
    def __init__(self, enc: STFTEncoder):
        super().__init__()
        self.hop = enc.hop
        self.n_fft = enc.n_fft
        self.window = enc.window

    def forward(self, x: torch.Tensor, length: int | None = None) -> torch.Tensor:
        return torch.istft(
            torch.view_as_complex(x).transpose(-1, -2),
            self.n_fft,
            self.hop,
            self.n_fft,
            self.window,
            center=False,
            onesided=True,
            length=length
        )

from matplotlib import pyplot as plt
import numpy as np

def show_spec(spec: torch.Tensor):
    spec_np = spec.numpy(force=True)
    spec_np = spec_np.transpose(1, 0, 2)
    spec_complex = spec_np[..., 0] + 1j * spec_np[..., 1]
    mag = np.abs(spec_complex)
    phase = np.angle(spec_complex)
    sin_p = (np.sin(phase) + 1) / 2
    cos_p = (np.cos(phase) + 1) / 2
    log_mag = np.log(mag + 1e-8)
    log_mag_norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min())
    phase_rgb = np.stack([sin_p * log_mag_norm, log_mag_norm, cos_p * log_mag_norm], axis=-1)

    h, w = phase_rgb.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(phase_rgb, origin="lower", interpolation="none")
    plt.show()
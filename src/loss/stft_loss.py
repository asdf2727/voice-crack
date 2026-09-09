import torch
from torch import Tensor

from modules.stft import STFT, ISTFT

def mag_phs_loss(target: Tensor, output: Tensor) -> tuple[Tensor, Tensor]:
    """
    Decouples magnitude and phase loss.
    Minimum is achieved when both losses are 0.
    However, if phase is unpredictable, magnitude is still optimized.
    Note that the standard ||c1-c2||^2 formulation does not have this property.
    """
    t_mag, t_phs = STFT.feats_to_log_polar(target)
    o_mag, o_phs = ISTFT.feats_to_log_polar(output)
    mag_loss = (t_mag - o_mag).square()
    phs_loss = 2 * t_mag.square() * (1.0 - torch.cos(t_phs - o_phs))
    return mag_loss.sum(dim=-1).mean(), phs_loss.sum(dim=-1).mean()

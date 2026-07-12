import torch

def autoencoder_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Squared error summed over all non-batch dims, averaged over the batch."""
    return (x - recon).flatten(1).square().sum(dim=1).mean()


def spectral_loss(p: torch.Tensor, t: torch.Tensor, floor: float = 0.3,
                  eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Amplitude + phase-alignment loss on channel-map spectra, (2, T, F) or
    (B, 2, T, F) with re/im at dim -3 (STFTEncoder.to_real layout); unbatched
    input behaves as a batch of one.

    Per complex bin:  mag   = (|p| - |t|)^2
                      phase = |t|^2 - |t| * dot(p, t) / (|p| + floor)
    Combine as mag + K * phase; returned separately for logging.

    Random-phase optimum of |p| stays |t| (no MSE-style collapse); aligned
    bins overshoot by ~K*floor/2, additive, so relatively vanishing for loud
    bins. `floor` is an absolute predicted-energy floor: below it, phase
    supervision becomes a pull toward the target direction, and phase of
    near-silent targets is unsupervised. The absolute scale is meaningful
    because waveforms are rms_normalize'd before the STFT.
    """
    mp = (p.square().sum(-3) + eps).sqrt()  # eps keeps |p| differentiable at 0
    mt2 = t.square().sum(-3)
    mt = (mt2 + eps).sqrt()
    dot = (p * t).sum(-3)                   # (B?, T, F)
    mag = (mp - mt).square()
    phase = mt2 - mt * dot / (mp + floor)
    return mag.flatten(-2).sum(-1).mean(), phase.flatten(-2).sum(-1).mean()

# ---------------------------------------------------------------------------
# Self-test: reductions and the spectral loss's analytic properties.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, L, T = 5, 7, 11

    # Reconstruction: summed squares over features, mean over batch -- for any
    # trailing shape, incl. the 4D (B, T, F, 2) complex-as-real STFT layout.
    x, y = torch.randn(B, T, 3), torch.randn(B, T, 3)
    assert torch.allclose(autoencoder_loss(x, y),
                          (x - y).square().sum() / B, atol=1e-5)
    x4, y4 = torch.randn(B, T, 3, 2), torch.randn(B, T, 3, 2)
    assert torch.allclose(autoencoder_loss(x4, y4),
                          (x4 - y4).square().sum() / B, atol=1e-5)

    # spectral_loss: under random phase the total is minimized at |p| = |t|
    # (plain MSE would collapse to 0), and the pull at p = 0 stays finite.
    thetas = torch.rand(4096) * 2 * torch.pi
    ts = torch.stack([5 * thetas.cos(), 5 * thetas.sin()], 1).reshape(-1, 2, 1, 1)
    def total(mp):
        ps = torch.zeros_like(ts)
        ps[:, 0] = mp
        m, ph = spectral_loss(ps, ts)
        return (m + ph).item()
    assert min((0.0, 2.5, 5.0, 7.5), key=total) == 5.0
    p0 = torch.zeros(2, 1, 1, requires_grad=True)
    spectral_loss(p0, torch.tensor([3.0, 4.0]).reshape(2, 1, 1))[1].backward()
    assert p0.grad.isfinite().all()

    # Unbatched (2, T, F) == batch of one.
    pb, tb = torch.randn(4, 2, 9, 6), torch.randn(4, 2, 9, 6)
    m_b, ph_b = spectral_loss(pb, tb)
    per = [spectral_loss(pb[b], tb[b]) for b in range(4)]
    assert torch.allclose(m_b, torch.stack([m for m, _ in per]).mean(), atol=1e-5)
    assert torch.allclose(ph_b, torch.stack([ph for _, ph in per]).mean(), atol=1e-5)

    print("stft_loss selftest OK")

if __name__ == "__main__":
    _selftest()
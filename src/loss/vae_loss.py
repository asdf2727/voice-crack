"""
ARD-VAE loss terms (arXiv:2501.10901), ported to PyTorch from the reference
TensorFlow implementation (github.com/Surojit-Utah/ARD-VAE, loss/vae_loss.py).

The bottleneck KL is taken against a zero-mean Gaussian prior whose per-axis
variance `var` is estimated from the encoded data (see loss.ard_prior.ARDPrior)
instead of the fixed N(0, I) of a vanilla VAE. Axes the model doesn't need get
their prior variance driven toward zero, which is what makes the effective
dimensionality readable.

Shape convention: batch-first, latent axis last -- mean/log_var are (B, L) or
(B, T, L); var is (L,). Losses are summed over all non-batch dims and
averaged over the batch, so with per-frame latents the KL sums over time just
like the reconstruction sums over frames -- the two terms stay on comparable
scales regardless of crop length.
"""
import torch


def kld_loss(mean: torch.Tensor, log_var: torch.Tensor, var: torch.Tensor,
             eps: float = 1e-8) -> torch.Tensor:
    """KL( N(mean, exp(log_var)) || N(0, var) ), per-axis diagonal Gaussians.

    The reference implementation adds `var` where the exact KL has `log(var)`;
    both are constant w.r.t. the encoder so gradients are identical -- we keep
    the exact form so the reported value is a true KL. The eps clamp keeps a
    fully collapsed axis (var -> 0) from turning the loss into inf/NaN.
    """
    v = var.clamp_min(eps).view(*([1] * (mean.dim() - 1)), -1)
    kld = 0.5 * (v.log() - log_var - 1.0 + (mean.square() + log_var.exp()) / v)
    return kld.flatten(1).sum(dim=1).mean()


def kld_loss_wo_const(mean: torch.Tensor, log_var: torch.Tensor, var: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """kld_loss minus the terms constant w.r.t. the encoder (same gradients)."""
    v = var.clamp_min(eps).view(*([1] * (mean.dim() - 1)), -1)
    kld = 0.5 * (-log_var + (mean.square() + log_var.exp()) / v)
    return kld.flatten(1).sum(dim=1).mean()


def autoencoder_loss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Squared error summed over all non-batch dims, averaged over the batch."""
    return (x - recon).flatten(1).square().sum(dim=1).mean()


def spectral_loss(p: torch.Tensor, t: torch.Tensor, floor: float = 0.3,
                  eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Amplitude + phase-alignment loss on interleaved re/im spectra (B, T, 2F).

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
    p2, t2 = p.unflatten(-1, (-1, 2)), t.unflatten(-1, (-1, 2))
    mp = (p2.square().sum(-1) + eps).sqrt()  # eps keeps |p| differentiable at 0
    mt2 = t2.square().sum(-1)
    mt = (mt2 + eps).sqrt()
    dot = (p2 * t2).sum(-1)
    mag = (mp - mt).square()
    phase = mt2 - mt * dot / (mp + floor)
    return mag.flatten(1).sum(-1).mean(), phase.flatten(1).sum(-1).mean()


# ---------------------------------------------------------------------------
# Self-test: checks the KL against hand-derived closed forms.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    B, L, T = 5, 7, 11

    # var = 1 -> unit-variance prior -> must equal the textbook
    # KL(N(mu, s^2) || N(0, 1)) = 0.5 * sum(mu^2 + s^2 - 1 - log s^2).
    mean, log_var = torch.randn(B, L), torch.randn(B, L)
    got = kld_loss(mean, log_var, torch.ones(L))
    want = (0.5 * (mean.square() + log_var.exp() - 1.0 - log_var)).sum(1).mean()
    assert torch.allclose(got, want, atol=1e-5), (got, want)

    # KL of the prior against itself is zero.
    var = torch.rand(L) + 0.1
    zero = kld_loss(torch.zeros(1, L), var.log().unsqueeze(0), var)
    assert abs(zero.item()) < 1e-5, zero.item()

    # A collapsed axis must stay finite (eps clamp).
    assert kld_loss(mean, log_var, torch.zeros(L)).isfinite()

    # (B, T, L) reduces like T stacked (B, L) problems, summed over time.
    mean3, log_var3 = torch.randn(B, T, L), torch.randn(B, T, L)
    got3 = kld_loss(mean3, log_var3, var)
    per_frame = torch.stack([kld_loss(mean3[:, t], log_var3[:, t], var)
                             for t in range(T)]).sum()
    assert torch.allclose(got3, per_frame, atol=1e-4), (got3, per_frame)

    # wo_const differs from the exact KL by a constant -> identical gradients.
    m1 = mean3.clone().requires_grad_(True)
    lv1 = log_var3.clone().requires_grad_(True)
    kld_loss(m1, lv1, var).backward()
    m2 = mean3.clone().requires_grad_(True)
    lv2 = log_var3.clone().requires_grad_(True)
    kld_loss_wo_const(m2, lv2, var).backward()
    assert torch.allclose(m1.grad, m2.grad, atol=1e-6)
    assert torch.allclose(lv1.grad, lv2.grad, atol=1e-6)

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
    ts = torch.stack([5 * thetas.cos(), 5 * thetas.sin()], -1)  # (n, 2)
    def total(mp):
        ps = torch.zeros_like(ts)
        ps[:, 0] = mp
        m, ph = spectral_loss(ps, ts)
        return (m + ph).item()
    assert min((0.0, 2.5, 5.0, 7.5), key=total) == 5.0
    p0 = torch.zeros(1, 2, requires_grad=True)
    spectral_loss(p0, torch.tensor([[3.0, 4.0]]))[1].backward()
    assert p0.grad.isfinite().all()

    print("vae_loss selftest OK")


if __name__ == "__main__":
    _selftest()

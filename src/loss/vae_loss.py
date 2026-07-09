"""
Variational bottleneck priors (ARD-VAE, arXiv:2501.10901).

Each prior owns its closed-form diagonal-Gaussian KL and an EMA estimate of
the aggregate posterior, updated every step from (mean, log_var) -- the exact
per-sample second moment mu^2 + sigma^2, i.e. the paper's sampled-z update
with the sampling noise integrated out. `halflife` is measured in samples
(latent frames): the prior must move on a much slower timescale than SGD or
axes ratchet into premature collapse.

Shape convention: latent axis last, time axis before it -- mean/log_var are
(T, L) or (B, T, L). KLs are summed over T and L and averaged over the batch;
unbatched input behaves as a batch of one.
"""
import torch
from torch import nn


def _top_frac(stat: torch.Tensor, frac: float) -> torch.Tensor:
    """Indices holding `frac` of stat's total, largest first (all if degenerate)."""
    order = stat.argsort(descending=True)
    total = stat.sum()
    if total <= 0:
        return order
    csum = stat[order].cumsum(0) / total
    return order[:int((csum < frac).sum().item()) + 1]


class LatentPrior(nn.Module):
    def kld_loss(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL( N(mean, exp(log_var)) || prior )"""
        ...
    def kld_rel_loss(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """kld_loss minus the terms constant w.r.t. the encoder (same gradients)."""
        ...
    def update(self, mean: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-8):
        """Adapt the prior / relevance statistics to the encoded data."""
        ...
    def relevant_dims(self, frac: float = 0.99) -> torch.Tensor:
        """Indices of the axes explaining `frac` of the relevance statistic,
        most relevant first."""
        ...
    @property
    def var(self) -> torch.Tensor:
        """Per-axis variance summary, for logging."""
        ...


class UnitPrior(LatentPrior):
    """Fixed N(0, I) prior; tracks E[mu^2] per axis (the signal part of the
    aggregate posterior variance) for relevance reporting only."""

    def __init__(self, latent_dim: int, halflife: float = 1000000.0):
        super().__init__()
        self._latent_dim = latent_dim
        self.halflife = halflife
        self.register_buffer("mean_sq", torch.ones(latent_dim))  # EMA of E[mu^2]

    def kld_loss(self, mean: torch.Tensor, log_var: torch.Tensor):
        kld = 0.5 * (mean.square() + log_var.exp() - 1.0 - log_var)
        return kld.flatten(-2).sum(dim=-1).mean()

    def kld_rel_loss(self, mean: torch.Tensor, log_var: torch.Tensor):
        kld = 0.5 * (mean.square() + log_var.exp() - log_var)
        return kld.flatten(-2).sum(dim=-1).mean()

    @torch.no_grad()
    def update(self, mean: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-8):
        keep = 0.5 ** (mean[..., 0].numel() / self.halflife)
        sq = mean.square().flatten(0, -2).mean(dim=0)
        self.mean_sq = (keep * self.mean_sq + (1 - keep) * sq).clamp_min(eps)

    def relevant_dims(self, frac: float = 0.99) -> torch.Tensor:
        return _top_frac(self.mean_sq, frac)

    @property
    def var(self) -> torch.Tensor:
        return self.mean_sq


class ScaledPrior(LatentPrior):
    """ARD prior N(0, v) with v tracking the aggregate posterior second moment
    E[mu^2 + sigma^2] (the conjugate update's beta/alpha, Rao-Blackwellized).
    Relevance per axis is v / E[sigma^2] - 1 = E[mu^2] / E[sigma^2], a
    signal-to-posterior-noise ratio: 0 for collapsed axes."""

    def __init__(self, latent_dim: int, halflife: float = 1000000.0):
        super().__init__()
        self._latent_dim = latent_dim
        self.halflife = halflife
        self.register_buffer("prior_var", torch.ones(latent_dim))     # EMA of E[mu^2 + sigma^2]
        self.register_buffer("post_var_mean", torch.ones(latent_dim)) # EMA of E[sigma^2]

    def kld_loss(self, mean: torch.Tensor, log_var: torch.Tensor):
        v = self.prior_var.view(*([1] * (mean.dim() - 1)), -1)
        kld = 0.5 * (v.log() - log_var - 1.0 + (mean.square() + log_var.exp()) / v)
        return kld.flatten(-2).sum(dim=-1).mean()

    def kld_rel_loss(self, mean: torch.Tensor, log_var: torch.Tensor):
        v = self.prior_var.view(*([1] * (mean.dim() - 1)), -1)
        kld = 0.5 * (-log_var + (mean.square() + log_var.exp()) / v)
        return kld.flatten(-2).sum(dim=-1).mean()

    @torch.no_grad()
    def update(self, mean: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-8):
        keep = 0.5 ** (mean[..., 0].numel() / self.halflife)
        sq = mean.square().flatten(0, -2).mean(dim=0)
        var = log_var.exp().flatten(0, -2).mean(dim=0)
        self.prior_var = (keep * self.prior_var + (1 - keep) * (sq + var)).clamp_min(eps)
        self.post_var_mean = (keep * self.post_var_mean + (1 - keep) * var).clamp_min(eps)

    def relevant_dims(self, frac: float = 0.99) -> torch.Tensor:
        return _top_frac((self.prior_var / self.post_var_mean - 1.0).clamp_min(0), frac)

    @property
    def var(self) -> torch.Tensor:
        return self.prior_var


# ---------------------------------------------------------------------------
# Self-test: closed forms, rel-loss gradients, EMA updates, relevance order.
# ---------------------------------------------------------------------------

def _selftest():
    import math
    torch.manual_seed(0)
    B, T, L = 5, 11, 7
    mean, log_var = torch.randn(B, T, L), torch.randn(B, T, L)

    # UnitPrior == textbook KL(N(mu, s^2) || N(0, 1)); ScaledPrior agrees
    # while its variance is still 1.
    unit, scaled = UnitPrior(L), ScaledPrior(L)
    want = (0.5 * (mean.square() + log_var.exp() - 1.0 - log_var)).flatten(1).sum(1).mean()
    assert torch.allclose(unit.kld_loss(mean, log_var), want, atol=1e-5)
    assert torch.allclose(scaled.kld_loss(mean, log_var), want, atol=1e-4)

    # Unbatched (T, L) == batch of one: batched KL is the mean of per-item KLs.
    per_item = torch.stack([unit.kld_loss(mean[b], log_var[b]) for b in range(B)])
    assert torch.allclose(per_item.mean(), want, atol=1e-5)

    # KL of the prior against itself is zero.
    scaled.prior_var = torch.rand(L) + 0.1
    zero = scaled.kld_loss(torch.zeros(1, 1, L),
                           scaled.prior_var.log().expand(1, 1, L))
    assert abs(zero.item()) < 1e-5, zero.item()

    # rel_loss differs from the full KL by a constant -> identical gradients.
    for prior in (unit, scaled):
        m1 = mean.clone().requires_grad_(True)
        lv1 = log_var.clone().requires_grad_(True)
        prior.kld_loss(m1, lv1).backward()
        m2 = mean.clone().requires_grad_(True)
        lv2 = log_var.clone().requires_grad_(True)
        prior.kld_rel_loss(m2, lv2).backward()
        assert torch.allclose(m1.grad, m2.grad, atol=1e-6)
        assert torch.allclose(lv1.grad, lv2.grad, atol=1e-6)

    # update() tracks E[mu^2 + sigma^2] without retaining the autograd graph.
    true_std = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.01, 0.01, 0.01])
    scaled = ScaledPrior(L, halflife=500.0)
    for _ in range(80):
        m = (torch.randn(64, L) * true_std).requires_grad_(True)
        lv = torch.full((64, L), -2.0, requires_grad=True)
        scaled.update(m, lv)
    assert scaled.prior_var.grad_fn is None, "update leaked the graph"
    want_var = true_std.square() + math.exp(-2.0)
    assert ((scaled.prior_var - want_var).abs() / want_var).max() < 0.2

    # Relevance: signal axes first (largest E[mu^2]/E[sigma^2]), noise-only
    # axes excluded below frac.
    dims = scaled.relevant_dims(0.995)
    assert dims[0].item() == 0, dims
    assert set(dims.tolist()) == {0, 1, 2, 3}, dims

    unit = UnitPrior(L, halflife=500.0)
    for _ in range(80):
        unit.update(torch.randn(64, L) * true_std, torch.zeros(64, L))
    assert set(unit.relevant_dims(0.995).tolist()) == {0, 1, 2, 3}

    print("vae_loss selftest OK")


if __name__ == "__main__":
    _selftest()

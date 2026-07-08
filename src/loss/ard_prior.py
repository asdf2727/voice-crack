"""
ARD prior over the latent axes (ARD-VAE, arXiv:2501.10901 sec. 3): each axis
gets a zero-mean Gaussian prior whose variance is estimated from the encoded
data, not trained by SGD. With uninformative hyperpriors the paper's conjugate
Gamma update reduces to the per-axis second moment of the latents; here that
moment is tracked with a per-batch exponential moving average instead of the
paper's per-epoch pooled recompute -- same estimator, exponential window,
no collect/update choreography:

    for batch in loader:
        z = ...
        prior.collect_and_update(z)     # every step, z detached internally
        kld uses prior.var as the per-axis target variance

`halflife` is measured in samples (latent *frames*, not batches): past data
loses half its weight every `halflife` samples. It is the stability knob --
the prior sits inside the loss the encoder is descending, so it must move on
a much slower timescale than SGD or axes ratchet into premature collapse.
Size it to at least a few hundred optimizer steps' worth of frames (e.g.
batch 16 x ~370 frames ~ 6K samples/step -> halflife of ~1e6 samples).

Axes whose variance collapses toward zero are irrelevant; `relevant_dims`
applies the paper's 99%-of-variance criterion.
"""
import torch
import torch.nn as nn


class ARDPrior(nn.Module):
    def __init__(self, latent_dim: int, halflife: float = 1000000.0):
        super().__init__()
        self._latent = latent_dim
        self.halflife = halflife
        self.register_buffer("var", torch.ones(latent_dim))

    @property
    def get_var(self) -> torch.Tensor:
        return self.var

    @torch.no_grad()
    def collect_and_update(self, z: torch.Tensor) -> None:
        """EMA step over latents, (..., L) with the latent axis last; with
        per-frame latents every frame counts as one sample."""
        z = z.reshape(-1, z.shape[-1])
        # Weight of past data halves every `halflife` samples, independent of
        # how those samples are batched: keep^(halflife/n batches) == 0.5.
        keep = 0.5 ** (z.shape[0] / self.halflife)
        self.var *= keep
        self.var += (1 - keep) * z.square().mean(dim=0)

    def relevant_dims(self, frac: float = 0.99) -> torch.Tensor:
        """Indices of the axes explaining `frac` of total prior variance, most
        relevant first (paper's criterion, minus the decoder-Jacobian
        weighting)."""
        order = self.var.argsort(descending=True)
        csum = self.var[order].cumsum(0) / self.var.sum()
        k = int((csum < frac).sum().item()) + 1
        return order[:k]


# ---------------------------------------------------------------------------
# Self-test: recovers a known anisotropic variance profile.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    L = 8
    true_std = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.01, 0.01, 0.01, 0.01])

    prior = ARDPrior(L, halflife=500.0)
    assert torch.allclose(prior.var, torch.ones(L)), "starts at N(0, I)"

    # Halflife semantics: after exactly `halflife` all-zero samples the
    # initial estimate must decay to 0.5, however the samples are batched.
    decay = ARDPrior(L, halflife=1000.0)
    for _ in range(4):
        decay.collect_and_update(torch.zeros(250, L))
    assert torch.allclose(decay.var, torch.full((L,), 0.5)), decay.var

    # Convergence: feed ~40 halflives of anisotropic data (2D and 3D shapes);
    # the EMA must land on the true per-axis second moment.
    for _ in range(80):
        prior.collect_and_update(torch.randn(64, L) * true_std)
        prior.collect_and_update(torch.randn(4, 50, L) * true_std)
    rel = (prior.var - true_std.square()).abs() / true_std.square()
    assert rel.max() < 0.15, rel  # EMA window ~1.4K effective samples

    dims = prior.relevant_dims(0.99)
    assert set(dims.tolist()) == {0, 1, 2, 3}, dims  # tiny axes excluded
    assert dims[0].item() == 0, dims  # sorted by variance, largest first

    print(f"ard_prior selftest OK: var {prior.var.numpy().round(3)} "
          f"| relevant dims {dims.tolist()}")


if __name__ == "__main__":
    _selftest()

"""
Conjugate Gamma hyperprior over per-axis latent precisions (ARD-VAE,
arXiv:2501.10901 sec. 3). The prior over each latent axis is N(0, beta/alpha);
alpha/beta are re-estimated in closed form from encoded samples instead of
being trained by SGD.

Per-epoch protocol (matches the reference trainer):

    for batch in loader:
        ...
        prior.collect(z)                # stash sampled latents as they appear
    prior.update()                      # conjugate update, once per epoch
    kld = kld_loss(mean, log_var, prior.alpha, prior.beta)

Axes whose beta/alpha collapses toward zero are irrelevant; `relevant_dims`
applies the paper's 99%-of-variance criterion.
"""
import torch
import torch.nn as nn


class ARDPrior(nn.Module):
    def __init__(self, latent_dim: int, a0: float = 0.0, b0: float = 0.0,
                 max_samples: int = 10_000, min_var: float = 1e-6):
        super().__init__()
        self.a0, self.b0 = a0, b0
        self.max_samples = max_samples
        self.min_var = min_var
        # Buffers are module state that moves with .to(device) and is saved in
        # state_dict() but is invisible to the optimizer -- alpha/beta must only
        # ever change through update(), never through gradients.
        self.register_buffer("alpha", torch.ones(latent_dim))
        self.register_buffer("beta", torch.ones(latent_dim))
        self._pool: list[torch.Tensor] = []
        self._pooled = 0

    @property
    def latent_dim(self) -> int:
        return self.alpha.numel()

    @property
    def target_var(self) -> torch.Tensor:
        """Per-axis prior variance beta/alpha (paper's sigma_hat^2)."""
        return (self.beta / self.alpha).clamp_min(self.min_var)

    @torch.no_grad()
    def collect(self, z: torch.Tensor) -> None:
        """Stash sampled latents, (B, L) or (B, L, T), for the next update().

        With per-frame latents every frame counts as one sample of the
        aggregate posterior. Stops once max_samples are pooled (the paper uses
        ~10K), so calling it every batch is fine.
        """
        if self._pooled >= self.max_samples:
            return
        if z.dim() == 3:
            z = z.transpose(1, 2).reshape(-1, z.shape[1])
        take = min(z.shape[0], self.max_samples - self._pooled)
        self._pool.append(z[:take].detach().clone())
        self._pooled += take

    @torch.no_grad()
    def update(self) -> None:
        """Closed-form conjugate update from the pooled samples:
        alpha = a0 + n/2, beta = b0 + sum(z^2)/2, so beta/alpha becomes the
        per-axis second moment of the encoded data about zero. No-op if
        nothing was collected."""
        if not self._pool:
            return
        z = torch.cat(self._pool)
        self._pool.clear()
        self._pooled = 0
        self.alpha.fill_(self.a0 + z.shape[0] / 2)
        self.beta.copy_(self.b0 + z.square().sum(dim=0) / 2)

    def relevant_dims(self, frac: float = 0.99) -> torch.Tensor:
        """Indices of the axes explaining `frac` of total prior variance, most
        relevant first (paper's criterion, minus the decoder-Jacobian
        weighting)."""
        var = self.target_var
        order = var.argsort(descending=True)
        csum = var[order].cumsum(0) / var.sum()
        k = int((csum < frac).sum().item()) + 1
        return order[:k]


# ---------------------------------------------------------------------------
# Self-test: recovers a known anisotropic variance profile.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    L = 8
    true_std = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.01, 0.01, 0.01, 0.01])

    prior = ARDPrior(L)
    assert torch.allclose(prior.target_var, torch.ones(L)), "starts at N(0, I)"

    # Feed both 2D and 3D batches (264 samples/iter x 50 > cap); cap must hold exactly.
    for _ in range(50):
        prior.collect(torch.randn(64, L) * true_std)
        prior.collect(torch.randn(4, L, 50) * true_std.view(1, -1, 1))
    assert prior._pooled == prior.max_samples, prior._pooled

    prior.update()
    rel = (prior.target_var - true_std.square()).abs() / true_std.square()
    assert rel.max() < 0.15, rel  # second moment of 10K samples ~ few % off

    dims = prior.relevant_dims(0.99)
    assert set(dims.tolist()) == {0, 1, 2, 3}, dims  # tiny axes excluded
    assert dims[0].item() == 0, dims  # sorted by variance, largest first

    # update() with an empty pool must not touch alpha/beta.
    a, b = prior.alpha.clone(), prior.beta.clone()
    prior.update()
    assert torch.equal(a, prior.alpha) and torch.equal(b, prior.beta)

    print(f"ard_prior selftest OK: target_var {prior.target_var.numpy().round(3)} "
          f"| relevant dims {dims.tolist()}")


if __name__ == "__main__":
    _selftest()

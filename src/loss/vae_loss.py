import torch
from torch import nn, Tensor


class LatentPrior(nn.Module):
    """
    Interface for a module managing and updating latent priors in VAE models.

    This contract provides functionality for computing the Kullback-Leibler divergence loss,
    adjusting latent priors based on encoded data, and determining relevant dimensions
    for relevance detection. The purpose is to facilitate latent space manipulations
    and statistical modeling in probabilistic frameworks.
    """
    def kld_loss(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """KL( N(mean, exp(log_var)) || prior )"""
        ...
    def kld_rel_loss(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """kld_loss minus the terms constant w.r.t. the encoder (same gradients)."""
        ...
    def update(self, mean: Tensor, log_var: Tensor, halflife: float = 1e6):
        """Adapt the prior and relevance statistics to the encoded data."""
        ...
    @property
    def snr(self) -> Tensor:
        """Per-axis weighed variance for relevancy detection."""
        ...

    @staticmethod
    def relevant_dims(var: Tensor, frac: float | None = 0.99) -> Tensor:
        """Indices of the axes explaining `frac` of the relevance statistic, most relevant first."""
        order = var.argsort(descending=True)
        if frac is None: return order
        csum = var[order].cumsum(0)
        stop = csum[-1].item() * frac
        significant = int((csum < stop).sum().item()) + 1
        return order[:significant]


class UnitPrior(LatentPrior):
    """
    Fixed N(0, I) prior; tracks E[mu^2] for relevance reporting only.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self._latent_dim = latent_dim
        self.register_buffer("mean_sq", torch.zeros(latent_dim))  # EMA of E[mu^2]

    def kld_loss(self, mean: Tensor, log_var: Tensor):
        """KL( N(mean, exp(log_var)) || N(0, I) )"""
        kld = 0.5 * (mean.square() + log_var.exp() - 1.0 - log_var)
        return kld.sum(dim=-1).mean()

    def kld_rel_loss(self, mean: Tensor, log_var: Tensor):
        kld = mean.square() + log_var.exp() - log_var
        return kld.sum(dim=-1).mean() * 0.5

    @torch.no_grad()
    def update(self, mean: Tensor, log_var: Tensor, halflife: float = 1e6):
        keep = 0.5 ** (mean[..., 0].numel() / halflife)
        sq = mean.square().flatten(0, -2).mean(dim=0)
        self.mean_sq = keep * self.mean_sq + (1 - keep) * sq

    @property
    def snr(self) -> Tensor:
        return self.mean_sq


class ScaledPrior(LatentPrior):
    """
    ARD prior N(0, prior_var) with prior_var tracking the aggregate
    posterior second moment E[mu^2 + sigma^2]. post_var tracks
    E[sigma^2] for relevance reporting only. Relevance per axis
    is prior_var / E[sigma^2] - 1 = E[mu^2] / E[sigma^2],
    tending to 0 for irrelevant axes.

    This is an EMA-based implementation of the original ARD-VAE paper
    (https://arxiv.org/abs/2501.10901), simplifying implementation and
    providing a more up-to-date relevance estimate than the original.
    This implementation also removes the need for computing the jacobian
    of the output w.r.t. the latent, seeing as SNR can be used as a direct
    estimate of relevance.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self._latent_dim = latent_dim
        self.register_buffer("prior_var", torch.ones(latent_dim)) # EMA of E[mu^2 + sigma^2]
        self.register_buffer("post_var", torch.ones(latent_dim))  # EMA of E[sigma^2]

    def kld_loss(self, mean: Tensor, log_var: Tensor):
        """KL( N(mean, exp(log_var)) || N(0, prior_var) )"""
        prior = self.prior_var.log().expand_as(mean)
        kld = 0.5 * (prior.log() - log_var - 1.0 + (mean.square() + log_var.exp()) / prior)
        return kld.sum(dim=-1).mean()

    def kld_rel_loss(self, mean: Tensor, log_var: Tensor):
        prior = self.prior_var.expand_as(mean)
        kld = -log_var + (mean.square() + log_var.exp()) / prior
        return kld.sum(dim=-1).mean() * 0.5

    @torch.no_grad()
    def update(self, mean: Tensor, log_var: Tensor, halflife: float = 1e6):
        keep = 0.5 ** (mean[..., 0].numel() / halflife)
        sq = mean.square().flatten(0, -2).mean(dim=0)
        var = log_var.exp().flatten(0, -2).mean(dim=0)
        self.prior_var = keep * self.prior_var + (1 - keep) * (sq + var)
        self.post_var =  keep * self.post_var +  (1 - keep) * var

    @property
    def snr(self) -> Tensor:
        return self.prior_var / self.post_var - 1.0

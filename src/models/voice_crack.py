import torch
from torch import nn, Tensor

from loss import *
from modules import *

class VoiceCrack(nn.Module):
    def __init__(self,
                 fft_bins,
                 enc_layers: int = 8,
                 enc_feats: int = 3,
                 dec_layers: int = 8,
                 dec_feats: int = 3):
        super().__init__()
        self.fft_bins = fft_bins
        self.enc_layers = enc_layers
        self.enc_feats = enc_feats
        self.dec_layers = dec_layers
        self.dec_feats = dec_feats
        self.step_cnt = 0
        in_feats = self.enc_feats * self.fft_bins
        out_feats = self.dec_feats * self.fft_bins
        self.in_filter = nn.Linear(8, self.enc_feats)
        self.enc_blocks = Vocos(in_feats, self.enc_layers)
        self.bottleneck = VAE(in_feats, out_feats)
        self.dec_blocks = Vocos(out_feats, self.dec_layers)
        self.out_filter = nn.Linear(self.dec_feats, 3)
        self.vae_prior = vae_loss.UnitPrior(self.bottleneck.latent_dim)

    def _config_dict(self):
        return {
            "fft_bins": self.fft_bins,
            "enc_layers": self.enc_layers,
            "enc_feats": self.enc_feats,
            "dec_layers": self.dec_layers,
            "dec_feats": self.dec_feats,
        }

    def save_model(self, path):
        torch.save({
            "config": self._config_dict(),
            "state": self.state_dict(),
            "step_cnt": self.step_cnt,
        }, path)

    @staticmethod
    def load_model(path) -> VoiceCrack:
        data = torch.load(path, weights_only=True)
        model = VoiceCrack(**data["config"])
        model.load_state_dict(data["state"])
        model.step_cnt = data["step_cnt"]
        return model

    def load_checkpoint(self, path):
        data = torch.load(path, weights_only=True)
        assert data["config"] == self._config_dict()
        self.load_state_dict(data["state"])
        self.step_cnt = data["step_cnt"]

    @property
    def latency(self):
        return self.enc_blocks.latency + self.dec_blocks.latency

    def _encode_spec(self, spec: Tensor) -> Tensor:
        filtered = self.in_filter(spec)  # select features
        flattened = torch.flatten(filtered, -2, -1)  # flatten frequency and feature dimensions
        return self.dec_blocks(flattened)  # run encoder vocos

    def _decode_spec(self, latent: Tensor) -> Tensor:
        decoded = self.dec_blocks(latent)  # run decoder vocos
        split = torch.unflatten(decoded, -1, (-1, self.dec_feats))  # unflatten into frequency bins with features
        return split

    def spec_to_latent(self, spec: Tensor, halflife: float | None = 1e6) -> tuple[Tensor, Tensor]:
        encoded = self._encode_spec(spec)
        mean, log_var = self.bottleneck.split(encoded)
        if halflife: self.vae_prior.update(mean, log_var, halflife)
        return mean, log_var

    def latent_to_spec(self, mean: Tensor, log_var: Tensor) -> Tensor:
        sample = self.bottleneck.sample(mean, log_var)
        return self._decode_spec(sample)

    def forward(self, spec: Tensor) -> Tensor:
        encoded = self._encode_spec(spec)
        sample = self.bottleneck(encoded)
        return self._decode_spec(sample)

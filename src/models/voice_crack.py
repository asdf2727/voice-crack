import torch
from torch import nn, Tensor

from loss import *
from modules import *

class VoiceCrack(nn.Module):
    def __init__(self,
                 n_fft: int,
                 win_chunks: int,
                 vocos_dim: int,
                 freq_bin_feats: int = 3,
                 conv_layers: int = 12):
        super().__init__()
        self.n_fft = n_fft
        self.win_chunks = win_chunks
        self.vocos_dim = vocos_dim
        self.freq_bin_feats = freq_bin_feats
        self.conv_layers = conv_layers
        self.step_cnt = 0

        self.enc = STFT(n_fft, win_chunks)
        self.dec = ISTFT(self.enc)
        in_feats = self.freq_bin_feats * self.enc.out_freq
        out_feats = 3 * self.dec.in_freq
        self.freq_bin_filter = nn.Linear(8, self.freq_bin_feats)
        self.enc_map = nn.Linear(in_feats, self.vocos_dim)
        self.enc_vocos = Vocos(self.vocos_dim, self.conv_layers)
        self.bottleneck = VAE(self.vocos_dim, self.vocos_dim)
        self.dec_vocos = Vocos(self.vocos_dim, self.conv_layers)
        self.dec_map = nn.Linear(self.vocos_dim, out_feats)
        self.vae_prior = vae_loss.UnitPrior(self.bottleneck.latent_dim)

    def _config_dict(self):
        return {
            "n_fft": self.n_fft,
            "win_chunks": self.win_chunks,
            "vocos_dim": self.vocos_dim,
            "freq_bin_feats": self.freq_bin_feats,
            "conv_layers": self.conv_layers,
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
        return self.enc_vocos.latency + self.dec_vocos.latency

    def _encode_spec(self, spec: Tensor) -> Tensor:
        filtered = self.freq_bin_filter(spec)  # select features
        flattened = torch.flatten(filtered, -2, -1)  # flatten frequency and feature dimensions
        return self.enc_vocos(self.enc_map(flattened))  # run encoder vocos

    def _decode_spec(self, latent: Tensor) -> Tensor:
        decoded = self.dec_map(self.dec_vocos(latent))  # run decoder vocos
        split = torch.unflatten(decoded, -1, (-1, 3))  # unflatten into frequency bins with features
        return split

    def spec_to_latent(self, spec: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self._encode_spec(spec)
        mean, log_var = self.bottleneck.split(encoded)
        return mean, log_var

    def latent_to_spec(self, mean: Tensor, log_var: Tensor) -> Tensor:
        sample = self.bottleneck.sample(mean, log_var)
        return self._decode_spec(sample)

    def forward(self, spec: Tensor) -> Tensor:
        encoded = self._encode_spec(spec)
        sample = self.bottleneck(encoded)
        return self._decode_spec(sample)

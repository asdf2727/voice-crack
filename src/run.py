import torch
import torch.nn.functional as F

from loss.vae_loss import LatentPrior, ScaledPrior
from models.STFT import STFTEncoder, STFTDecoder
from models.ae import BlockParams, TCNDecoder, TCNEncoder


def _compute_gain(wave: torch.Tensor, target: float = 0.1, min_rms: float = 1e-3):
    return target / wave.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(min_rms)


class Runner:
    config: dict
    stft_enc: STFTEncoder
    stft_dec: STFTDecoder
    enc: TCNEncoder
    dec: TCNDecoder
    prior: LatentPrior
    device: torch.device

    @staticmethod
    def load_from_file(path, device: torch.device = torch.device("cpu")):
        data = torch.load(path, map_location=device)
        out = Runner.new_from_config(data["config"], device)
        out.load_data(data)
        return out

    @staticmethod
    def new_sym_model(
            blocks: list[BlockParams],
            latent_dim: int,
            stft_hop: int = 256,
            stft_win_chunks: int = 4,
            device: torch.device = torch.device("cpu"),
    ) -> Runner:
        return Runner.new_from_config({
            "blocks": [vars(b) for b in blocks],
            "latent_dim": latent_dim,
            "stft_hop": stft_hop,
            "stft_win_chunks": stft_win_chunks,
        }, device)

    @staticmethod
    def new_from_config(config: dict, device = torch.device("cpu")):
        out = Runner()
        out.config = config
        blocks = [BlockParams(**b) for b in config["blocks"]]
        out.stft_enc = STFTEncoder(config["stft_hop"], config["stft_win_chunks"]).to(device)
        out.stft_dec = STFTDecoder(out.stft_enc).to(device)
        out.enc = TCNEncoder(out.stft_enc.out_freqs, blocks, config["latent_dim"]).to(device)
        out.dec = TCNDecoder(config["latent_dim"], blocks, out.stft_enc.out_freqs).to(device)
        out.prior = ScaledPrior(config["latent_dim"]).to(device)
        out.device = device
        print(f"model: {out.enc.in_freq}/{out.stft_enc.out_freqs} bins, "
              f"latent {config['latent_dim']} ({len(out.prior.relevant_dims())} relevant), "
              f"latency {out.enc.latency + out.dec.latency} frames")
        return out

    def load_data(self, data: dict):
        self.enc.load_state_dict(data["enc"])
        self.dec.load_state_dict(data["dec"])
        self.prior.load_state_dict(data["prior"])

    def save_model(self, path) -> None:
        torch.save({
            "config": self.config,
            "enc": self.enc.state_dict(),
            "dec": self.dec.state_dict(),
            "prior": self.prior.state_dict(),
        }, path)

    def wave_to_spec(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gain = _compute_gain(wave)
        wave = (wave * gain).to(self.device)
        return gain, self.stft_enc(wave)[..., :self.enc.in_freq]

    def relevant_latent(self, mean: torch.Tensor, frac: float = 0.99) -> torch.Tensor:
        return mean[..., self.prior.relevant_dims(frac)]

    def full_latent(self, rel_latent: torch.Tensor, frac: float = 0.99, add_noise: bool = True) -> torch.Tensor:
        rel_dims = self.prior.relevant_dims(frac)
        latent_dim = self.config["latent_dim"]
        if rel_latent.shape[-1] != rel_dims.numel():
            raise ValueError(f"latent has {rel_latent.shape[-1]} dims, but {rel_dims.numel()} of {latent_dim} relevant")
        new_shape = (*rel_latent.shape[:-1], latent_dim)
        # Collapsed dims are sampled from the prior: the decoder's noise source.
        new_latent = (torch.randn(new_shape, device=rel_latent.device) * self.prior.var.sqrt()
                      if add_noise else torch.zeros(new_shape, device=rel_latent.device))
        new_latent[..., rel_dims] = rel_latent
        return new_latent

    def spec_to_wave(self, spec: torch.Tensor) -> torch.Tensor:
        pad = (0, self.stft_dec.in_freq - spec.shape[-1])
        out_spec = F.pad(spec, pad)
        out_wave = self.stft_dec(out_spec)
        return out_wave

    def run_input(self, wave: torch.Tensor) -> torch.Tensor:
        gain, spec = self.wave_to_spec(wave)
        out_spec = self.dec(self.enc(spec))
        out_wave = self.spec_to_wave(out_spec)
        return out_wave / gain

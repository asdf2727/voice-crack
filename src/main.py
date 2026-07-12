import torch

from audio.device_stream import *
from audio.file_stream import FileSource
from datasets.vctk import VCTKDataset

from models.STFT import *
from models.ae import *
from loss.vae_loss import ScaledPrior, LatentPrior


def rand_phase(spec: torch.Tensor) -> torch.Tensor:
    mag = STFTDecoder.to_complex(spec).abs()
    return STFTEncoder.to_real(mag * torch.exp((2j * np.pi) * torch.rand_like(mag)))

def run_encoder(enc: TCNEncoder, prior: LatentPrior, spec: torch.Tensor) -> torch.Tensor:
    # Hybrid inference: deterministic mean on the relevant dims (content),
    # the collapsed dims are re-sampled from the prior in run_decoder.
    return enc.encode(spec)[0][..., prior.relevant_dims()]

def run_decoder(dec: TCNDecoder, prior: LatentPrior, latent: torch.Tensor) -> torch.Tensor:
    rel = prior.relevant_dims()
    mask = torch.ones_like(prior.var)
    mask[rel] = 0
    mean = torch.zeros((latent.shape[0], prior.var.shape[0]))
    mean[..., rel] = latent
    # reparameterize wants LOG variance; masked (relevant) dims get ~zero std,
    # collapsed dims get their prior variance -- the decoder's noise source.
    return dec(TCNEncoder.reparameterize(mean, (prior.var * mask + 1e-12).log()))

def check_strides(x: torch.Tensor):
    print(f"shape {tuple(x.shape)} - stride - {x.stride()} - cont? {x.is_contiguous()}")

def build_from_checkpoint(ckpt: str, in_freq: int) -> tuple[TCNEncoder, TCNDecoder, LatentPrior]:
    ckpt = torch.load(ckpt, map_location='cpu')

    enc = TCNEncoder(in_freq, **ckpt["config"])
    dec = TCNDecoder(in_freq, **ckpt["config"])
    prior = ScaledPrior(enc.latent_dim)

    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    prior.load_state_dict(ckpt["prior"])

    rel = prior.relevant_dims()
    print(f"model: {in_freq} bins, latent {enc.latent_dim} "
          f"({len(rel)} relevant: {rel.tolist()}), "
          f"latency {enc.latency + dec.latency} frames | epoch {ckpt['stats']['epoch']}")

    enc.eval()
    dec.eval()
    return enc, dec, prior

def main():
    chunk_size = 256

    stft = STFTEncoder(chunk_size, 4, window=torch.hamming_window)
    istft = STFTDecoder(stft)

    enc, dec, prior = build_from_checkpoint("../models/tcn_ardvae.pt", stft.n_fft // 2 + 1)

    ds = VCTKDataset("../datasets/VCTK-Corpus-0.92")
    idx = np.random.randint(len(ds))
    print(f"ID {idx}")

    print("playing original...")
    file = FileSource(ds.get_path(idx), 0.05)
    # Same normalization as training; the model works in normalized space and
    # the gain is re-applied to its output.
    wav, gain = rms_normalize(torch.from_numpy(file.get_wav()))
    spec = stft.forward(wav)
    check_strides(spec)
    show_spec(spec[..., enc.latency + dec.latency :, :])
    DeviceSink.dump_source(file)

    print("running model...")
    latent = run_encoder(enc, prior, spec)
    print(latent.shape)
    out_spec = run_decoder(dec, prior, latent)
    # The valid-conv top band isn't reconstructed; zero it for the istft.
    out_spec = torch.nn.functional.pad(out_spec, (0, spec.shape[-1] - out_spec.shape[-1]))

    print("playing reconstructed...")
    check_strides(out_spec)
    show_spec(out_spec)
    out_wav = (istft.forward(out_spec) / gain).numpy(force=True)
    out_wav = out_wav[:len(out_wav)//chunk_size * chunk_size].reshape(-1, chunk_size)
    with DeviceSink(chunk_size, sr=file.sample_rate()) as sink:
        for chunk in out_wav:
            sink.put_chunk(chunk)    #chunk_data = dec(spec)
    #print(chunk_data.shape)

if __name__ == "__main__":
    main()

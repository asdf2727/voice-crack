from audio.device_stream import *
from audio.file_stream import FileSource
from datasets.vctk import VCTKDataset

from models.STFT import *
from models.ae import *
from loss.vae_loss import ScaledPrior


def rand_phase(spec: torch.Tensor) -> torch.Tensor:
    mag = STFTDecoder.to_complex(spec).abs()
    return STFTEncoder.to_real(mag * torch.exp((2j * np.pi) * torch.rand_like(mag)))

def run_model(enc, dec, spec: torch.Tensor) -> torch.Tensor:
    #return spec
    #return rand_phase(spec)
    return dec(enc(spec))

def check_strides(x: torch.Tensor):
    print(f"shape {tuple(x.shape)} - stride - {x.stride()} - cont? {x.is_contiguous()}")

def build_from_checkpoint(ckpt: str, hop: int) -> tuple[TCNEncoder, TCNDecoder]:
    ckpt = torch.load(ckpt, map_location='cpu')

    enc_sd, dec_sd = ckpt["enc"], ckpt["dec"]
    hidden, in_ch = enc_sd["enc_in.weight"].shape
    latent = enc_sd["mean_head.weight"].shape[0]
    enc_blocks = len({k.split(".")[1] for k in enc_sd if k.startswith("enc_tcn.")})
    dec_blocks = len({k.split(".")[1] for k in dec_sd if k.startswith("dec_tcn.")})
    kernel = enc_sd["enc_tcn.0.time_conv.weight"].shape[-1]

    enc = TCNEncoder(in_ch, hidden=hidden, latent_dim=latent,
                     blocks=enc_blocks, kernel=kernel)
    dec = TCNDecoder(in_ch, latent_dim=latent, hidden=dec_sd["dec_in.weight"].shape[0],
                     blocks=dec_blocks, kernel=kernel)
    prior = ScaledPrior(enc.latent_dim)

    enc.load_state_dict(enc_sd)
    dec.load_state_dict(dec_sd)
    prior.load_state_dict(ckpt["prior"])

    rel = prior.relevant_dims()
    print(f"model: in_ch {enc.enc_in.weight.shape[1]}, latent {enc.latent_dim} "
          f"({len(rel)} relevant: {rel.tolist()}), "
          f"latency {enc.latency + dec.latency} frames | epoch {ckpt['stats']['epoch']}")

    enc.eval()  # z = mean, deterministic
    dec.eval()
    return enc, dec

def main():
    chunk_size = 256

    enc, dec = build_from_checkpoint("../models/tcn_ardvae.pt", chunk_size)

    stft = STFTEncoder(chunk_size, 4, window=torch.hamming_window)
    istft = STFTDecoder(stft)

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
    show_spec(spec[enc.latency + dec.latency :, :])
    DeviceSink.dump_source(file)

    print("running model...")
    out_spec = run_model(enc, dec, spec)

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

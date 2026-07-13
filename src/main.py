from audio.device_stream import *
from audio.file_stream import FileSource
from datasets.vctk import VCTKDataset

from models.STFT import *
from models.ae import *
from run import Runner


def rand_phase(spec: torch.Tensor) -> torch.Tensor:
    mag = STFTDecoder.to_complex(spec).abs()
    return STFTEncoder.to_real(mag * torch.exp((2j * np.pi) * torch.rand_like(mag)))

def check_strides(x: torch.Tensor):
    print(f"shape {tuple(x.shape)} - stride - {x.stride()} - cont? {x.is_contiguous()}")

def main():
    chunk_size = 256

    stft = STFTEncoder(chunk_size, 4, window=torch.hamming_window)
    istft = STFTDecoder(stft)

    runner = Runner.load_from_file("../models/ConvNeXt_ae.pt", torch.device("cpu"))

    ds = VCTKDataset("../datasets/VCTK-Corpus-0.92")
    idx = np.random.randint(len(ds))
    print(f"ID {idx}")

    print("playing original...")
    file = FileSource(ds.get_path(idx), 0.05)
    # Same normalization as training; the model works in normalized space and
    # the gain is re-applied to its output.
    gain, spec = runner.wave_to_spec(torch.from_numpy(file.get_wav()))
    check_strides(spec)
    show_spec(spec[..., runner.enc.latency + runner.dec.latency :, :])
    DeviceSink.dump_source(file)

    print("running model...")
    latent = runner.relevant_latent(runner.enc(spec))
    print(latent.shape)
    out_spec = runner.dec(runner.full_latent(latent))
    check_strides(out_spec)
    show_spec(out_spec)
    out_wav = runner.spec_to_wave(out_spec) / gain

    print("playing reconstructed...")
    out_wav = out_wav[:len(out_wav)//chunk_size * chunk_size].reshape(-1, chunk_size)
    with DeviceSink(chunk_size, sr=file.sample_rate()) as sink:
        for chunk in out_wav:
            sink.put_chunk(chunk)    #chunk_data = dec(spec)
    #print(chunk_data.shape)

if __name__ == "__main__":
    main()

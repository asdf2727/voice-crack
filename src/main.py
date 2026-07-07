from audio.device_stream import *
from audio.file_stream import FileSource
from datasets.vctk import VCTKDataset

from models.STFT import *

def rand_phase(spec: torch.Tensor) -> torch.Tensor:
    mag = spec.square().sum(dim=2).sqrt()
    phase = torch.rand_like(mag) * 2 * np.pi
    return torch.stack([mag * phase.sin(), mag * phase.cos()], -1)

def run_model(spec: torch.Tensor) -> torch.Tensor:
    #return drop_phase(spec)
    return rand_phase(spec)

def check_strides(x: torch.Tensor):
    print(f"shape {tuple(x.shape)} - stride - {x.stride()} - cont? {x.is_contiguous()}")

def main():
    chunk_size = 256

    enc = STFTEncoder(chunk_size, 4, window=torch.hamming_window)
    dec = STFTDecoder(enc)

    ds = VCTKDataset("../datasets/VCTK-Corpus-0.92")
    idx = np.random.randint(len(ds))
    print(f"ID {idx}")

    file = FileSource(ds.get_path(idx), 0.05)
    wav = torch.from_numpy(file.get_wav())
    check_strides(wav)
    spec = enc.forward(wav)
    check_strides(spec)
    show_spec(spec)

    print("playing original...")
    DeviceSink.dump_source(file)

    out_spec = run_model(spec)
    check_strides(out_spec)

    print("playing reconstructed...")
    out_wav = dec.forward(out_spec).numpy(force=True)
    out_wav = out_wav[:len(out_wav)//chunk_size * chunk_size].reshape(-1, chunk_size)
    with DeviceSink(chunk_size, sr=file.sample_rate()) as sink:
        for chunk in out_wav:
            sink.put_chunk(chunk)    #chunk_data = dec(spec)
    #print(chunk_data.shape)

if __name__ == "__main__":
    main()
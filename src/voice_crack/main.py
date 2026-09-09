import torch

from audio.file_stream import FileSource
from audio.device_stream import DeviceSink
from time import sleep
from modules import stft

dataset_root = "../datasets/VCTK-Corpus-0.92/wav48_silence_trimmed/"

def check_strides(x: torch.Tensor):
    print(f"shape {tuple(x.shape)} - stride - {x.stride()} - cont? {x.is_contiguous()}")

def main():
    chunk_len = 128

    source = FileSource(dataset_root + "p225/p225_003_mic1.flac", 0.05)
    wav = torch.from_numpy(source.get_wav())
    check_strides(wav)

    encode = stft.STFT(256, 4)
    decode = stft.ISTFT(encode)

    spec = encode(wav)
    check_strides(spec)
    out_spec = torch.stack([spec[..., 5], spec[..., 3], spec[..., 2]], dim=-1)
    check_strides(out_spec)

    stft.show_hsv(decode.feats_to_hsv(out_spec.movedim(-1, -2)))

    out_wav = decode(out_spec).numpy()
    print(f"out wav shape {out_wav.shape}")
    with DeviceSink(chunk_len, sr=source.sample_rate()) as sink:
        for i in range(len(out_wav) // chunk_len):
            start = i * chunk_len
            sink.put_chunk(out_wav[start:start+chunk_len])
        sleep(0.5)

if __name__ == "__main__":
    main()

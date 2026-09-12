import torch
from torch import Tensor
from time import sleep
import random

from audio.file_stream import FileSource
from audio.device_stream import DeviceSink

from datasets.vctk import VCTK_092
from models import VoiceCrack
from modules import stft

def check_strides(x: Tensor):
    print(f"shape {tuple(x.shape)} - stride - {x.stride()} - cont? {x.is_contiguous()}")

def main():
    chunk_len = 128

    dataset = VCTK_092("../../datasets/")
    wav = random.choice(dataset)[0]
    check_strides(wav)

    encode = stft.STFT(256, 4)
    decode = stft.ISTFT(encode)
    model = VoiceCrack.load_model("../../models/v0/epoch_1.pt").eval()
    print(model.vae_prior.snr.sum())
    print(str(model.in_filter.weight.detach().numpy()))
    print(str(model.in_filter.bias.detach().numpy()))

    spec = encode(wav)
    check_strides(spec)
    stft.show_hsv(encode.feats_to_hsv(spec[model.latency:, ...]))

    out_spec = model(spec).detach()
    check_strides(out_spec)
    stft.show_hsv(decode.feats_to_hsv(out_spec))

    out_wav = decode(out_spec).numpy()
    print(f"out wav shape {out_wav.shape}")
    with DeviceSink(chunk_len, sr=48000) as sink:
        for i in range(len(out_wav) // chunk_len):
            start = i * chunk_len
            sink.put_chunk(out_wav[start:start+chunk_len])
        sleep(0.5)

if __name__ == "__main__":
    main()

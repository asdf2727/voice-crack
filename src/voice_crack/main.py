import os.path
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
    dataset = VCTK_092("../../datasets/")
    id = random.randrange(len(dataset))
    print(id)
    wav = dataset[id][0]

    encode = stft.STFT(1024, 4)
    decode = stft.ISTFT(encode)

    spec = encode(wav)
    stft.show_hsv(encode.feats_to_hsv(spec))

    run_model = True and os.path.exists("../../models/v0.2/latest.pt")
    if run_model:
        model = VoiceCrack.load_model("../../models/v0.2/latest.pt").eval()
        print(model.vae_prior.snr.sum())
        print(str(model.in_filter.weight.detach().numpy()))
        print(str(model.in_filter.bias.detach().numpy()))
        out_spec = model(spec).detach()
        stft.show_hsv(decode.feats_to_hsv(out_spec))
    else:
        out_spec = stft.to_out_feats(spec)

    out_wav = decode(out_spec).numpy()
    chunk_len = 128
    with DeviceSink(chunk_len, sr=48000) as sink:
        for i in range(len(out_wav) // chunk_len):
            start = i * chunk_len
            sink.put_chunk(out_wav[start:start+chunk_len])
        sleep(0.5)

if __name__ == "__main__":
    main()

import os.path
from torch import Tensor
import torch.nn.functional as F
import random


from audio.device_stream import DeviceSink
from audio.file_stream import FileSource, FileSink
from audio.tensor_stream import ArraySource
from datasets.vctk import VCTK_092
from models import VoiceCrack
from modules import stft

def check_strides(x: Tensor):
    print(f"shape {tuple(x.shape)} - stride - {x.stride()} - cont? {x.is_contiguous()}")


encode = stft.STFT(1024, 4)
decode = stft.ISTFT(encode)
model = None
if os.path.exists("../../models/v0.2/latest.pt"):
    model = VoiceCrack.load_model("../../models/v0.2/latest.pt").eval()
    print(model.vae_prior.snr[model.vae_prior.relevant_dims()])
    print(str(model.freq_bin_filter.weight.detach().numpy()))
    print(str(model.freq_bin_filter.bias.detach().numpy()))

dataset = None
if os.path.exists("../../datasets/VCTK-Corpus-0.92"):
    dataset = VCTK_092("../../datasets/")

def load(cmd: list[str]) -> Tensor:
    if cmd[0] == "vctk":
        assert dataset is not None
        wav_id = int(cmd[1]) if len(cmd) > 1 else random.randrange(0, len(dataset))
        print(f"Loaded sample {wav_id}")
        wav = dataset[wav_id][0]
    else:
        wav = Tensor(FileSource(cmd[0], 128, 48000).get_wav())
    spec = encode(wav)
    stft.show_hsv(encode.feats_to_hsv(spec))
    return spec

def main():
    in_spec = None
    out_wav = None
    while True:
        cmd = input("> ").split(" ")
        if cmd[0] == "exit":
            break
        elif cmd[0] == "load":
            in_spec = load(cmd[1:])
            out_wav = decode(stft.to_out_feats(in_spec))
        elif cmd[0] == "run":
            padded = F.pad(in_spec, (0, 0, 0, 0, model.latency, 0))
            out_spec = model(padded).detach()
            stft.show_hsv(decode.feats_to_hsv(out_spec))
            out_wav = decode(out_spec)
        elif cmd[0] == "play":
            DeviceSink.dump_source(ArraySource(out_wav, 0.05, 48000))
        elif cmd[0] == "save":
            FileSink.dump_source("../../output.wav", ArraySource(out_wav, 0.05, 48000))
        elif cmd[0] == "test":
            in_spec = load(["vctk"])
            padded = F.pad(in_spec, (0, 0, 0, 0, model.latency, 0))
            out_spec = model(padded).detach()
            stft.show_hsv(decode.feats_to_hsv(out_spec))
            out_wav = decode(out_spec)
            DeviceSink.dump_source(ArraySource(out_wav, 0.05, 48000))


if __name__ == "__main__":
    main()

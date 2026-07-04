import random
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr
import torch
from torch.utils.data import Dataset

class VCTKDataset(Dataset):
    def __init__(self, root: str, sample_rate: int = 48000,
                 segment_seconds: float = 1.0):
        wav_dir = Path(root) / "wav48_silence_trimmed"
        self.files = sorted(wav_dir.glob(f"p*/*.flac"))   # the index — paths only
        if not self.files:
            raise FileNotFoundError(f"No flac under {wav_dir}")
        speakers = sorted({f.parent.name for f in self.files})
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_seconds)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        wav, src_sr = sf.read(path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)                       # your _to_mono, inline
        if src_sr != self.sample_rate:
            wav = soxr.resample(wav, src_sr, self.sample_rate)
        wav = self._fix_length(wav)                      # → exactly segment_len samples
        speaker = self.speaker_to_id[path.parent.name]
        return torch.from_numpy(wav), speaker

from torch.utils.data import DataLoader
ds = VCTKDataset("/path/to/VCTK-Corpus-0.92", sample_rate=16000, segment_seconds=1.0)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

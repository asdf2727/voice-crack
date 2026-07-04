from pathlib import Path
import torch
from torch.utils.data import Dataset

from audio.file_stream import FileSource


class VCTKDataset(Dataset):
    def __init__(
            self,
            root: str,
            mic_id: str = 'any',
            download: bool = False,
            url: str = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip',
            audio_ext='flac'):
        self._root = Path(root) / "wav48_silence_trimmed"
        if not self._root.exists():
            if not download: raise FileNotFoundError(f"{root} not a valid VCTK dataset")
            # TODO download

        if mic_id is "any": self.files = sorted(self._root.glob(f"p*/*.{audio_ext}"))
        else: self.files = sorted(self._root.glob(f"p*/*_{mic_id}.{audio_ext}"))
        if not self.files: raise FileNotFoundError(f"No {audio_ext} under {self._root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str, str]:
        path = self.files[idx]
        source = FileSource(path, 0)
        wav = source.get_file()
        ids = path.name.split("_")
        return torch.from_numpy(wav), source.sample_rate(), ids[0], ids[1]


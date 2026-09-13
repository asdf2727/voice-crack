import os
from pathlib import Path

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip

"""
Copied over from https://docs.pytorch.org/audio/main/_modules/torchaudio/datasets/vctk.html#VCTK_092

changed a few things
"""

URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
CHECKSUM = "f96258be9fdc2cbff6559541aae7ea4f59df3fcaf5cf963aae5ca647357e359c"

SampleType = tuple[Tensor, int, str, str, str]

class VCTK_092(Dataset):
    """*VCTK 0.92* :cite:`yamagishi2019vctk` dataset

    Args:
        root (str): Root directory where the dataset's top level directory is found.
        mic_id (str, optional): Microphone ID. Either ``"mic1"`` or ``"mic2"``. (default: ``"mic2"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"``)

    Note:
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
    """

    def __init__(
            self,
            root: Path | str,
            mic_id: str = "any",
            download: bool = False,
            url: str = URL,
    ):
        root = Path(root)
        self._path = root / "VCTK-Corpus-0.92"
        self._txt_dir = self._path / "txt"
        self._audio_dir = self._path / "wav48_silence_trimmed"
        self._mic_id = mic_id

        if not os.path.isdir(self._path):
            archive = root / "VCTK-Corpus-0.92.zip"
            if not os.path.isfile(archive):
                if not download:
                    raise RuntimeError("Dataset not found. Please use `download=True` to download it.")
                print(f"Downloading VCTK-Corpus-0.92.zip to {archive}")
                download_url_to_file(url, str(archive), hash_prefix=CHECKSUM)
            _extract_zip(str(archive), str(self._path))

        if mic_id == "any":
            files = self._audio_dir.glob(f"p*/*.*")
        else:
            files = self._audio_dir.glob(f"p*/*_{mic_id}.*")
        files = sorted(files, key=lambda x: os.path.getsize(x))
        self._samples = []
        for file in files:
            name, ext = os.path.splitext(file.name)
            self._samples.append((*name.split("_"), ext))

    def __getitem__(self, n: int) -> tuple[Tensor, int, str | None, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor: Waveform
            int:
                Sample rate
            str:
                Transcript
            str:
                Speaker ID
            str:
                Utterance ID
        """
        speaker_id, utterance_id, mic_id, audio_ext = self._samples[n]
        transcript_path = self._txt_dir / speaker_id / f"{speaker_id}_{utterance_id}.txt"
        audio_path = self._audio_dir / speaker_id / f"{speaker_id}_{utterance_id}_{mic_id}{audio_ext}"

        transcript = None
        if os.path.isfile(transcript_path):
            with open(transcript_path) as file_path:
                transcript = file_path.readlines()[0]
        waveform, sample_rate =  torchaudio.load(audio_path)
        waveform = waveform.mean(0) # average all channels

        return waveform, sample_rate, transcript, speaker_id, utterance_id

    def __len__(self) -> int:
        return len(self._samples)

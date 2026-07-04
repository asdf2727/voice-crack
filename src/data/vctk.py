"""
VCTK dataset + streaming TBPTT lane manager for the voice-conversion pipeline.

Two pieces, deliberately decoupled:

1. ``VCTKDataset`` -- a plain map-style ``Dataset``. Dumb utterance provider:
   given an index it loads ONE full utterance (mono, resampled) plus its speaker
   id. Variable length on purpose; NO cropping. This is the thing you can also
   hand to a normal ``DataLoader`` for e.g. precomputing speaker vectors.

2. ``LaneManager`` -- an ``IterableDataset`` implementing persistent-lane
   (stateful) batched truncated BPTT. It keeps ``num_lanes`` parallel streams
   alive, chops each utterance into fixed-size ``chunk`` s, packs
   ``chunks_per_window`` of them into a window, and hot-swaps a fresh utterance
   into a lane the moment its current one runs out -- flagging that chunk as a
   ``reset`` so the training loop can zero that lane's recurrent state.
   Trailing partial chunks are DROPPED, never padded, so the model never sees an
   incomplete chunk (this is the "mask before the last chunk" behaviour: a chunk
   is only ever emitted whole, and ``valid`` marks which chunks carry real data).

Everything here is in *chunk units of raw waveform*. Mel/feature extraction and
the recurrent cell live in the model, not here -- compute mel batched on GPU in
the training step. The window a LaneManager yields is raw audio:

    window : float32 [num_lanes, chunks_per_window, chunk]   # the audio
    speaker: int64   [num_lanes, chunks_per_window]          # speaker id per chunk (-1 if invalid)
    reset  : bool    [num_lanes, chunks_per_window]          # True on a lane's first chunk of a new utterance
    valid  : bool    [num_lanes, chunks_per_window]          # True where the chunk is real data (== loss mask)

Model side, per chunk c in lane b, in time order:
    if reset[b, c]: h[:, b, :] = 0        # equivalently  h = h * (~reset)[None, :, None]
    out = cell(window[b, c], h[:, b])     # cell loops the mel frames inside the chunk
    ... accumulate loss only where valid[b, c] ...
Detach h between *windows* (that's what truncates BPTT); the reset above handles
utterance boundaries inside a window.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from torch.utils.data import Dataset, IterableDataset


class VCTKDataset(Dataset):
    """Map-style provider of full VCTK-0.92 utterances (mono, resampled)."""

    def __init__(self, root: str, sample_rate: int = 16000, mic: str = "mic1"):
        wav_dir = Path(root) / "wav48_silence_trimmed"
        # Index paths only -- no audio is decoded until __getitem__.
        self.files = sorted(wav_dir.glob(f"p*/*_{mic}.flac"))
        if not self.files:
            raise FileNotFoundError(f"No {mic} .flac files under {wav_dir}")
        # Speaker string ('p225') -> contiguous int id, for embedding lookup / clustering.
        speakers = sorted({f.parent.name for f in self.files})
        self.speaker_to_id = {s: i for i, s in enumerate(speakers)}
        self.sample_rate = sample_rate

    @property
    def num_speakers(self) -> int:
        return len(self.speaker_to_id)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        wav, src_sr = sf.read(path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)                      # -> mono
        if src_sr != self.sample_rate:
            wav = soxr.resample(wav, src_sr, self.sample_rate, "VHQ")
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        return torch.from_numpy(wav), self.speaker_to_id[path.parent.name]


@dataclass
class _Lane:
    """Mutable per-lane state carried across windows within one epoch."""
    wav: np.ndarray | None = None   # current utterance's samples (None => empty/drained)
    pos: int = 0                    # read cursor, in samples
    speaker: int = -1               # current utterance's speaker id


class LaneManager(IterableDataset):
    """
    Persistent-lane batched TBPTT over an utterance source.

    ``utts`` is anything indexable that returns ``(waveform, speaker_id)`` --
    typically a ``VCTKDataset``, but any ``__len__`` / ``__getitem__`` pair works
    (see ``_SyntheticUtts`` in the self-test). One decode per utterance per epoch,
    so memory stays flat regardless of corpus size.

    NOTE on workers: an IterableDataset is *replicated* into every DataLoader
    worker, and the lane state here is inherently central, so run this with
    ``num_workers=0`` (or don't wrap it in a DataLoader at all -- it already
    yields full batches). Parallelise the raw audio *decode* separately later if
    __getitem__ becomes the bottleneck.
    """

    def __init__(self, utts, num_lanes: int, chunk: int,
                 chunks_per_window: int, seed: int = 0):
        self.utts = utts
        self.num_lanes = num_lanes
        self.chunk = chunk
        self.chunks_per_window = chunks_per_window
        self.seed = seed
        self._epoch = 0

    # --- utterance plumbing -------------------------------------------------

    def _get_wav(self, idx: int):
        wav, spk = self.utts[idx]
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()
        return np.asarray(wav, dtype=np.float32), int(spk)

    def _load_next(self, lane: _Lane, queue: list[int]) -> bool:
        """Pull the next utterance into ``lane``. False if the queue is empty."""
        if not queue:
            lane.wav = None
            return False
        lane.wav, lane.speaker = self._get_wav(queue.pop())
        lane.pos = 0
        return True

    def _take_chunk(self, lane: _Lane, queue: list[int]):
        """
        Advance ``lane`` by one whole chunk.

        Returns ``(chunk, speaker, reset, valid)``. Skips utterances with no full
        chunk left (drops the trailing partial), hot-loading the next one; the
        first chunk of any freshly loaded utterance gets ``reset=True``. When the
        queue is exhausted and the lane has nothing left, returns a drained slot
        (``valid=False``).
        """
        reset = False
        while lane.wav is None or lane.pos + self.chunk > len(lane.wav):
            reset = True
            if not self._load_next(lane, queue):
                return None, -1, False, False          # drained
        chunk = lane.wav[lane.pos:lane.pos + self.chunk]
        lane.pos += self.chunk
        return chunk, lane.speaker, reset, True

    # --- iteration ----------------------------------------------------------

    def __iter__(self):
        # Reshuffle each epoch; __iter__ is called once per epoch by convention.
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        queue = list(range(len(self.utts)))
        rng.shuffle(queue)

        lanes = [_Lane() for _ in range(self.num_lanes)]
        B, C, K = self.num_lanes, self.chunks_per_window, self.chunk

        while True:
            window = np.zeros((B, C, K), dtype=np.float32)
            speaker = np.full((B, C), -1, dtype=np.int64)
            reset = np.zeros((B, C), dtype=bool)
            valid = np.zeros((B, C), dtype=bool)

            for b in range(B):
                for c in range(C):
                    chunk, spk, rst, val = self._take_chunk(lanes[b], queue)
                    reset[b, c] = rst
                    valid[b, c] = val
                    if val:
                        window[b, c] = chunk
                        speaker[b, c] = spk

            if not valid.any():        # queue empty and every lane drained -> epoch done
                return

            yield (
                torch.from_numpy(window),
                torch.from_numpy(speaker),
                torch.from_numpy(reset),
                torch.from_numpy(valid),
            )


# ---------------------------------------------------------------------------
# Self-test: exercises the lane logic on synthetic audio, no VCTK needed.
# ---------------------------------------------------------------------------

class _SyntheticUtts:
    """Random-length noise utterances, incl. some shorter than one chunk."""

    def __init__(self, n: int, sr: int, num_speakers: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._data = []
        for _ in range(n):
            secs = rng.uniform(0.05, 3.0)          # 0.05s clips are < one chunk -> skipped
            length = int(sr * secs)
            wav = (rng.standard_normal(length).astype(np.float32) * 0.1)
            self._data.append((wav, int(rng.integers(num_speakers))))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


def _selftest():
    sr, chunk = 16000, 3200                        # 200 ms chunks
    utts = _SyntheticUtts(n=37, sr=sr, num_speakers=5, seed=1)
    lm = LaneManager(utts, num_lanes=4, chunk=chunk, chunks_per_window=8, seed=0)

    total_valid = total_reset = n_windows = 0
    for window, speaker, reset, valid in lm:
        assert window.shape == (4, 8, chunk)
        assert speaker.shape == reset.shape == valid.shape == (4, 8)
        assert not bool((reset & ~valid).any()), "reset must only land on valid chunks"
        assert bool((speaker[valid] >= 0).all()), "valid chunks must have a speaker"
        assert bool((speaker[~valid] == -1).all()), "invalid chunks must be sentinel -1"
        total_valid += int(valid.sum())
        total_reset += int(reset.sum())
        n_windows += 1

    # Conservation: every whole chunk of every utterance is emitted exactly once,
    # and each utterance that yields >=1 chunk triggers exactly one reset.
    expected_valid = sum(len(utts[i][0]) // chunk for i in range(len(utts)))
    utts_with_chunks = sum(1 for i in range(len(utts)) if len(utts[i][0]) // chunk >= 1)
    assert total_valid == expected_valid, (total_valid, expected_valid)
    assert total_reset == utts_with_chunks, (total_reset, utts_with_chunks)

    print(f"selftest OK: {n_windows} windows | {total_valid} valid chunks "
          f"(expected {expected_valid}) | {total_reset} resets == "
          f"{utts_with_chunks} utterances-with-chunks | "
          f"{len(utts) - utts_with_chunks} sub-chunk utterances skipped")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="VCTK dataset / lane manager")
    ap.add_argument("--root", type=str, default=None,
                    help="VCTK-0.92 root (the dir containing wav48_silence_trimmed)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--mic", type=str, default="mic1")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic lane-logic test (no dataset needed)")
    args = ap.parse_args()

    if args.selftest or args.root is None:
        _selftest()
    else:
        ds = VCTKDataset(args.root, sample_rate=args.sample_rate, mic=args.mic)
        print(f"{len(ds)} utterances, {ds.num_speakers} speakers")
        wav, spk = ds[0]
        print(f"utt0: {tuple(wav.shape)} samples @ {ds.sample_rate} Hz, speaker id {spk}")

        chunk = int(0.2 * ds.sample_rate)
        lm = LaneManager(ds, num_lanes=8, chunk=chunk, chunks_per_window=8)
        window, speaker, reset, valid = next(iter(lm))
        print(f"first window: {tuple(window.shape)} | "
              f"valid {int(valid.sum())}/{valid.numel()} | resets {int(reset.sum())}")

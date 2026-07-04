"""
Streaming TBPTT lane manager for the voice-conversion pipeline.

`LaneManager` is an `IterableDataset` implementing persistent-lane (stateful)
batched truncated BPTT over a map-style utterance dataset (`datasets.vctk`).
It keeps `batch_size` parallel lanes alive, chops each utterance into fixed
`chunk_len` chunks, packs `window_size` chunks per window, and hot-swaps a
fresh utterance into a lane the moment its current one runs out -- flagging that
chunk as a `reset` so the training loop can zero that lane's recurrent state.
Trailing partial chunks are dropped, never padded (the model never sees an
incomplete chunk).

Each window is a *flattened* raw-waveform vector, so whole runs of chunks copy in
one slice and mel can be applied batched downstream. Per epoch it yields:

- **window**: float32[batch_size, window_size * chunk_len] -- raw audio, flattened
- **reset**: bool[batch_size, window_size] -- True on a lane's first chunk of a new utterance
- **valid**: bool[batch_size, window_size] -- True where the chunk is real data (== loss mask)

The dataset yields audio at its native sample rate, so `chunk_len` is expressed
in that rate. Mel/feature extraction, resampling, and the recurrent cell live in
the model, not here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

from vctk import VCTKDataset

class LaneManager(IterableDataset):
    def __init__(
            self,
            dataset,
            chunk_len: int,
            window_size: int,
            batch_size: int = 1,
            seed: int = 0):
        self._dataset = dataset
        self.chunk_len = chunk_len
        self.window_size = window_size
        self.batch_size = batch_size

        self._rng = random.Random(seed)
        self._ids = list(range(len(self._dataset)))
        self._id_pos = 0

    @dataclass
    class _Lane:
        data: np.ndarray = None   # current utterance's samples (None => drained)
        pos: int = 0              # read cursor, in samples

    def _write_lane(self, lane: _Lane, window: np.ndarray, reset: np.ndarray) -> int:
        """
        Writes to window and reset inplace. Skips utterances with no full
        chunk left (drops the trailing partial), hot-loading the next one; the
        first chunk of any freshly loaded utterance gets `reset=True`.

        Returns how many chunks have been written, different from window size
        only when the queue is exhausted and the lane has nothing left.
        """
        C, K = self.window_size, self.chunk_len
        pos = 0
        while pos < C and lane.data is not None:
            chunks_left = (len(lane.data) - lane.pos) // K
            write = min(chunks_left, C - pos)
            # A reset marks the first real chunk of a freshly loaded utterance:
            # lane.pos == 0 means we're at the very start of this utterance, and
            # write > 0 means we actually emit a chunk here (skips too-short utts
            # and never fires on a cross-window continuation, where lane.pos > 0).
            if write > 0 and lane.pos == 0:
                reset[pos] = True
            window[pos * K : (pos + write) * K] = lane.data[lane.pos : lane.pos + write * K]
            pos += write
            lane.pos += write * K
            # Utterance exhausted (its whole chunks consumed) -> load the next one.
            if write != chunks_left:
                continue
            if self._id_pos >= len(self._dataset):
                lane.data = None
                break
            lane.data = self._dataset[self._ids[self._id_pos]][0]
            self._id_pos += 1
            lane.pos = 0
        return pos

    def __iter__(self):
        # Reshuffle each epoch; __iter__ is called once per epoch by convention.
        self._rng.shuffle(self._ids)

        if self.batch_size > len(self._dataset):
            raise ValueError(f"batch_size {self.batch_size} > dataset size {len(self._dataset)}")
        lanes = [self._Lane(data=self._dataset[self._ids[i]][0]) for i in range(self.batch_size)]
        self._id_pos = len(lanes)

        B, C, K = self.batch_size, self.window_size, self.chunk_len

        # Run until every lane is drained -- not just until the queue is assigned,
        # or the last few utterances still buffered in the lanes would be dropped.
        while True:
            window = np.zeros((B, C * K), dtype=np.float32)
            reset = np.zeros((B, C), dtype=bool)
            valid = np.zeros((B, C), dtype=bool)
            any_valid = False

            for b in range(B):
                written = self._write_lane(lanes[b], window[b], reset[b])
                valid[b, :written] = True
                any_valid |= bool(written)

            if not any_valid:
                return

            yield (
                torch.from_numpy(window),
                torch.from_numpy(reset),
                torch.from_numpy(valid),
            )


# ---------------------------------------------------------------------------
# Self-test: exercises the lane logic on synthetic audio, no VCTK needed.
# ---------------------------------------------------------------------------

class _SyntheticUtts:
    """Random-length noise utterances, incl. some shorter than one chunk.

    Mirrors `VCTKDataset`'s contract: `__getitem__` returns a 4-tuple
    `(waveform_tensor, sample_rate, speaker, utt_id)` so LaneManager's `[0]`
    extraction path is identical to production.
    """

    def __init__(self, n: int, sr: int, num_speakers: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._sr = sr
        self._data = []
        for i in range(n):
            secs = rng.uniform(0.05, 3.0)          # 0.05s clips are < one chunk -> skipped
            length = int(sr * secs)
            wav = (rng.standard_normal(length).astype(np.float32) * 0.1)
            spk = f"p{int(rng.integers(num_speakers)):03d}"
            self._data.append((torch.from_numpy(wav), sr, spk, str(i)))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


def _selftest():
    sr, chunk = 16000, 3200                        # 200 ms chunks
    B, C = 4, 8
    utts = _SyntheticUtts(n=37, sr=sr, num_speakers=5, seed=1)
    lm = LaneManager(utts, chunk_len=chunk, window_size=C, batch_size=B)

    total_valid = total_reset = n_windows = 0
    for window, reset, valid in lm:
        assert window.shape == (B, C * chunk), window.shape
        assert reset.shape == valid.shape == (B, C), (reset.shape, valid.shape)
        assert not bool((reset & ~valid).any()), "reset must only land on valid chunks"
        total_valid += int(valid.sum())
        total_reset += int(reset.sum())
        n_windows += 1

    # Conservation: every whole chunk of every utterance is emitted exactly once,
    # and each utterance that yields >= 1 chunk triggers exactly one reset. These
    # three quantities together catch the type/reset/termination bugs: early
    # termination lowers total_valid; a continuation reset raises total_reset; a
    # missing mid-window reset lowers it.
    lengths = [len(utts[i][0]) for i in range(len(utts))]
    expected_valid = sum(n // chunk for n in lengths)
    utts_with_chunks = sum(1 for n in lengths if n // chunk >= 1)
    assert total_valid == expected_valid, (total_valid, expected_valid)
    assert total_reset == utts_with_chunks, (total_reset, utts_with_chunks)

    print(f"selftest OK: {n_windows} windows | {total_valid} valid chunks "
          f"(expected {expected_valid}) | {total_reset} resets == "
          f"{utts_with_chunks} utterances-with-chunks | "
          f"{len(utts) - utts_with_chunks} sub-chunk utterances skipped")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LaneManager over VCTK / synthetic audio")
    ap.add_argument("--root", type=str, default=None,
                    help="VCTK-0.92 root (the dir containing wav48_silence_trimmed)")
    ap.add_argument("--mic", type=str, default="any",
                    help="mic id (e.g. mic1, mic2) or 'any'")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic lane-logic test (no dataset needed)")
    args = ap.parse_args()

    if args.selftest or args.root is None:
        _selftest()
    else:
        ds = VCTKDataset(args.root, mic_id=args.mic)
        print(f"{len(ds)} utterances")
        wav, sr, spk, utt = ds[0]
        print(f"utt0: {tuple(wav.shape)} samples @ {sr} Hz, speaker {spk}, id {utt}")

        chunk = int(0.2 * sr)
        lm = LaneManager(ds, chunk_len=chunk, window_size=8, batch_size=4)
        window, reset, valid = next(iter(lm))
        print(f"first window: {tuple(window.shape)} | "
              f"valid {int(valid.sum())}/{valid.numel()} | resets {int(reset.sum())}")

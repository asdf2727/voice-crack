from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

from datasets.vctk import VCTKDataset

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
        self._id_pos = len(self._dataset)
        self._ids = list(range(self._id_pos))

    @dataclass
    class _Lane:
        data: np.ndarray = None
        pos: int = 0

    def _write_lane(self, lane: _Lane, window: np.ndarray, reset: np.ndarray) -> int:
        """
        Writes to window and reset inplace. Skips utterances with no full
        chunk left (drops the trailing partial), hot-loading the next one; the
        first chunk of any freshly loaded utterance gets ``reset=True``.

        Returns how many chunks have been written, different from window size
        only when the queue is exhausted and the lane has nothing left.
        """
        C, K = self.window_size, self.chunk_len
        pos = 0
        while pos < C and lane.data is not None:
            chunks_left = (len(lane.data) - lane.pos) // K
            write = min(chunks_left, C - pos)
            # Write to window and reset
            if pos == 0: reset[pos] = True
            window[pos * K : (pos + write) * K] = lane.data[lane.pos : lane.pos + write * K]
            pos += write
            lane.pos += write * K
            # Load next utterance if exhausted
            if write != chunks_left: continue
            if self._id_pos >= len(self._dataset):
                lane.data = None
                break
            lane.data = self._dataset[self._ids[self._id_pos]]
            self._id_pos += 1
            lane.pos = 0
        return pos

    def __iter__(self):
        # Reshuffle each epoch; __iter__ is called once per epoch by convention.
        self._rng.shuffle(self._ids)
        lanes = [self._Lane(data=self._dataset[self._ids[i]]) for i in range(self.batch_size)]
        self._id_pos = len(lanes)

        B, C, K = self.batch_size, self.window_size, self.chunk_len

        while self._id_pos < len(self._dataset):
            window = np.zeros((B, C * K), dtype=np.float32)
            reset = np.zeros((B, C), dtype=bool)
            valid = np.zeros((B, C), dtype=bool)

            for b in range(B):
                written = self._write_lane(lanes[b], window[b], reset[b])
                valid[b, :written] = True
            yield (
                torch.from_numpy(window),
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
    lm = LaneManager(utts, chunk_len=chunk, window_size=8, batch_size=4)

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
        ds = VCTKDataset(args.root)
        print(f"{len(ds)} utterances")
        wav, sr, spk, id = ds[0]
        print(f"utt0: {tuple(wav.shape)} samples @ {sr} Hz, speaker id {spk}")

        chunk = int(0.2 * sr)
        lm = LaneManager(ds, chunk_len=chunk, window_size=8, batch_size=4)
        window, speaker, reset, valid = next(iter(lm))
        print(f"first window: {tuple(window.shape)} | "
              f"valid {int(valid.sum())}/{valid.numel()} | resets {int(reset.sum())}")

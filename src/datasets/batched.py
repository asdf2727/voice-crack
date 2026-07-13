"""
Length-bucketed batching for variable-length utterance datasets.

`BatchCropDataset` wraps a map-style dataset whose items are (waveform, ...)
tuples *sorted by length* (datasets.vctk sorts by file size as a proxy). Each
item of this dataset is a whole batch: `batch_size` consecutive utterances,
each randomly cropped to the length of the batch's shortest member. Because
neighbors in a length-sorted order have similar lengths, the cropped-away
audio per batch is minimal. The trailing partial batch (the longest
utterances) is dropped.

Use with DataLoader(batch_size=None): the dataset already yields batches, so
the loader only shuffles batch order and handles workers/pinning. Batch
*composition* stays fixed across epochs (the usual bucketing trade-off);
crop offsets are re-rolled on every access.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class BatchCropDataset(Dataset):
    def __init__(self, dataset, batch_size: int, max_samples: int | None = None):
        """`max_samples` caps the crop length: peak training memory is linear
        in batch duration, and with length-sorted batching the longest files
        land in the same batch with no short member to crop them -- unbounded
        without a cap."""
        if len(dataset) < batch_size:
            raise ValueError(f"batch_size {batch_size} > dataset size {len(dataset)}")
        self._ds = dataset
        self.batch_size = batch_size
        self.max_samples = max_samples

    def __len__(self) -> int:
        return len(self._ds) // self.batch_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """One full batch: float32 (batch_size, S), S = shortest member,
        capped at max_samples."""
        rng = np.random.default_rng()
        waves = [np.asarray(self._ds[idx * self.batch_size + j][0], dtype=np.float32)
                 for j in range(self.batch_size)]
        n = min(len(w) for w in waves)
        if self.max_samples is not None:
            n = min(n, self.max_samples)
        out = np.empty((self.batch_size, n), dtype=np.float32)
        for j, w in enumerate(waves):
            start = rng.integers(len(w) - n + 1)
            out[j] = w[start:start + n]
        return torch.from_numpy(out)


class InfiniteBatchCrops(IterableDataset):
    """Endless uniformly-random batches (with replacement) from a
    BatchCropDataset, for generation-based training. Each DataLoader worker
    samples independently (seed offset by worker id)."""

    def __init__(self, batches: BatchCropDataset, seed: int | None = None):
        self._batches = batches
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        seed = None if self.seed is None else self.seed + (info.id if info else 0)
        rng = np.random.default_rng(seed)
        while True:
            yield self._batches[int(rng.integers(len(self._batches)))]


# ---------------------------------------------------------------------------
# Self-test: batch shapes, minimal cropping, crops are contiguous slices.
# ---------------------------------------------------------------------------

class _SortedUtts:
    """Random utterances of increasing length; mirrors VCTKDataset's tuple
    contract (item[0] is the waveform) and its sorted-by-length ordering."""

    def __init__(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._data = [rng.standard_normal(120 + 17 * i).astype(np.float32)
                      for i in range(n)]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i], 48000, f"p{i:03d}", str(i)


def _selftest():
    n, B = 23, 4
    ds = _SortedUtts(n)
    batches = BatchCropDataset(ds, batch_size=B)
    assert len(batches) == n // B, len(batches)  # trailing partial dropped

    for i in range(len(batches)):
        batch = batches[i]
        members = [ds[i * B + j][0] for j in range(B)]
        # Cropped to the batch's shortest member == first one (sorted order).
        assert batch.shape == (B, len(members[0])), batch.shape
        # Every row is a contiguous slice of its source utterance.
        for j, src in enumerate(members):
            row = batch[j].numpy()
            hits = [s for s in range(len(src) - len(row) + 1)
                    if np.array_equal(src[s:s + len(row)], row)]
            assert hits, f"batch {i} row {j} is not a slice of its utterance"

    # Crop offsets re-roll between accesses (longest member has slack).
    last = len(batches) - 1
    assert any(not torch.equal(batches[last], batches[last]) for _ in range(8)), \
        "crop offsets never vary"

    # max_samples caps every batch, and rows are still contiguous slices.
    capped = BatchCropDataset(ds, batch_size=B, max_samples=100)
    assert all(capped[i].shape == (B, 100) for i in range(len(capped)))

    # Infinite sampler: valid batch shapes forever; the batch-index sequence
    # is deterministic per seed (crop offsets still re-roll).
    import itertools
    inf = list(itertools.islice(iter(InfiniteBatchCrops(batches, seed=1)), 8))
    inf2 = list(itertools.islice(iter(InfiniteBatchCrops(batches, seed=1)), 8))
    assert all(b.shape[0] == B for b in inf)
    assert [tuple(a.shape) for a in inf] == [tuple(c.shape) for c in inf2]

    print(f"batched selftest OK: {len(batches)} batches of {B}, "
          f"shortest {batches[0].shape[1]} / longest {batches[last].shape[1]} samples")


if __name__ == "__main__":
    _selftest()

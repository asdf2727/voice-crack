import numpy as np

class StreamSource:
    _sr: int
    _chunk_size: int

    @staticmethod
    def _to_mono(chunk: np.ndarray):
        if chunk.ndim == 2: return chunk.mean(axis=1)
        elif chunk.ndim != 1: raise ValueError(f"Invalid block shape: {chunk.shape}")
        return chunk

    def sample_rate(self) -> int: return self._sr
    def chunk_size(self) -> int: return self._chunk_size
    def chunk_time(self) -> float: return self._chunk_size / self._sr

    def get_next_chunk(self) -> np.ndarray | None: ...

class StreamSink:
    def put_chunk(self, chunk: np.ndarray) -> None: ...
    def close(self): ...
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close(); return False
    def __del__(self):
        try: self.close()
        except: pass

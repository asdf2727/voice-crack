from .streams import *
import numpy as np

class ArraySource(StreamSource):
    def __init__(self, array: np.ndarray, chunk_time: float, sr: int):
        self._wav = array
        self._sr = sr
        self._chunk_size = int(self._sr * chunk_time)
        self.current_sample = 0

    def get_wav(self) -> np.ndarray: return self._wav

    def get_next_chunk(self) -> np.ndarray | None:
        if self.current_sample >= len(self._wav):
            return None
        out = self._wav[self.current_sample: self.current_sample + self._chunk_size]
        self.current_sample += self._chunk_size
        return out

class ArraySink(StreamSink):
    def __init__(self):
        self._wav = np.empty((0,))

    def get_wav(self) -> np.ndarray: return self._wav

    def put_chunk(self, chunk: np.ndarray) -> None:
        self._wav = np.concatenate([self._wav, chunk], axis=0)

    @staticmethod
    def dump_source(source: StreamSource, max_time: float = 999999999) -> np.ndarray:
        spent_time = 0
        with ArraySink() as sink:
            while (chunk := source.get_next_chunk()) is not None:
                sink.put_chunk(chunk)
                spent_time += source.chunk_time()
                if spent_time > max_time: break
            return sink.get_wav()

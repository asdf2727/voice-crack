from audio.streams import *
import sys
import numpy as np
import soundfile as sf
import soxr

# TODO maybe add lazy file source loading
class FileSource(StreamSource):
    def __init__(self, path: str, chunk_time: float, sr: int | None = None):
        self._raw, src_sr = sf.read(path, dtype='float32')
        self._raw = self._to_mono(self._raw)
        self._sr = src_sr if sr is None else sr
        if src_sr != self._sr:
            print(f"Warning: Resampling file from {src_sr} to {self._sr}", file=sys.stderr)
            self._raw = soxr.resample(self._raw, src_sr, self._sr, 'VHQ')
        self._chunk_size = int(self._sr * chunk_time)
        self.current_sample = 0

    def get_chunk(self) -> np.ndarray | None:
        if self._raw is None: return None
        if self.current_sample >= len(self._raw):
            self._raw = None
            return None
        out = self._raw[self.current_sample: self.current_sample + self._chunk_size]
        self.current_sample += self._chunk_size
        return out

class FileSink(StreamSink):
    def __init__(self, path: str, sr: int):
        self._sf = sf.SoundFile(path, mode='w', samplerate=sr, channels=1)

    def put_chunk(self, chunk: np.ndarray) -> None:
        self._sf.write(chunk)        # float32 in [-1,1] → auto-converted to file's subtype

    def close(self):
        if self._sf is None: return
        self._sf.close()
        self._sf = None

    @staticmethod
    def dump_source(path: str, source: StreamSource, max_time: float = 999999999):
        spent_time = 0
        with FileSink(path, source.sample_rate()) as sink:
            while (chunk := source.get_chunk()) is not None:
                sink.put_chunk(chunk)
                spent_time += source.chunk_time()
                if spent_time > max_time: break

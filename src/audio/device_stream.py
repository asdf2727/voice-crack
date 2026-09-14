from audio.streams import *
import sys
import numpy as np
import sounddevice as sd
import queue
from time import sleep

class DeviceSource(StreamSource):
    def __init__(self, chunk_time: float, sr: int = 44100, device: int | str = "default"):
        self._sr = sr
        self._chunk_size = int(sr * chunk_time)
        self.q = queue.Queue[np.ndarray]()
        if device is str:
            device = sd.query_devices(device, 'input')["index"]
        self._stream = sd.InputStream(
            samplerate=sr,
            blocksize=self._chunk_size,
            device=device,
            channels=1,
            dtype='float32',
            callback=self._callback)
        if self._sr != self._stream.samplerate:
            raise ValueError(f"Sample rate mismatch: {self._sr} != {self._stream.samplerate}")
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
        if status: print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def get_next_chunk(self) -> np.ndarray | None:
        if self._stream is None: return None
        return self._to_mono(self.q.get())

class DeviceSink(StreamSink):
    def __init__(self, chunk_size: int, sr: int = 44100, device: int | str = "default"):
        self.q = queue.Queue[np.ndarray](maxsize=4)
        if device is str:
            device = sd.query_devices(device, 'output')["index"]
        self._stream = sd.OutputStream(
            samplerate=sr,
            blocksize=chunk_size,
            device=device,
            channels=1,
            dtype='float32',
            callback=self._callback)
        if sr != self._stream.samplerate:
            raise ValueError(f"Sample rate mismatch: {sr} != {self._stream.samplerate}")
        self._stream.start()

    def put_chunk(self, chunk: np.ndarray) -> None:
        if self._stream is None: return
        self.q.put(chunk)

    def _callback(self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
        if status: print(status, file=sys.stderr)
        n = 0
        try:
            chunk = self.q.get_nowait()
            n = len(chunk)
            outdata[:n, 0] = chunk
        except queue.Empty:
            pass
        if n < frames:
            outdata[n:, 0] = 0

    def close(self):
        if self._stream is None: return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    @staticmethod
    def dump_source(source: StreamSource):
        with DeviceSink(source.chunk_size(), sr=source.sample_rate()) as sink:
            while (chunk := source.get_next_chunk()) is not None:
                sink.put_chunk(chunk)
            sleep(0.5) # wait for the buffered stream to finish
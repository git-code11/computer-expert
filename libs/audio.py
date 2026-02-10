import typing as tp
import signal
import threading
import queue
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import asyncio
import functools


@dataclass
class AudioStream:
    data: bytes


class AudioStreamGenerator:
    def __init__(self, sample_rate=16000, chunk_duration_ms=5000, overlap_duration_ms=100, device=None):
        self.stop_signal = threading.Event()  # asyncio.Event()
        self.block_size = (sample_rate * chunk_duration_ms) // 1000
        self.overlap_block_size = (overlap_duration_ms * sample_rate) // 1000
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        # Sample_width of 2byte or 16bit means subtype=PCM_16, format=WAV, dtype=int16
        self.sample_width = 2
        self.device = device

    def generate(self) -> tp.Iterable[sr.AudioData]:
        q_in = queue.Queue()

        def callback(indata, frames, time, status):
            q_in.put(indata.copy())

        overlap = None

        def data_rewrite(data):
            nonlocal overlap
            if overlap is not None:
                data = np.concat([overlap, data])
            overlap = data[data.size - self.overlap_block_size:]
            return data

        def to_audio_data(data):
            return sr.AudioData(
                bytes(data),
                self.sample_rate,
                sample_width=self.sample_width)

        # with self.register_stop_handler():
        with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device,
                dtype="int16",
                channels=1,
                callback=callback
        ) as stream:
            print("Listening")
            while True:
                can_stop = self.stop_signal.is_set()
                is_empty = q_in.empty()
                if stream.active and can_stop:
                    stream.abort()
                elif is_empty and can_stop:
                    break
                try:
                    # data = stream.read(self.block_size)[0]
                    data = q_in.get(timeout=self.chunk_duration_ms//2000)
                    data = data_rewrite(data)
                    yield to_audio_data(data)
                except queue.Empty:
                    pass
            print("Stop Listening")

    def run(self, input: tp.Any, **kwargs) -> tp.Iterable[sr.AudioData]:
        with self.register_stop_handler():
            yield from self.generate()

    async def async_run(self, input: tp.Any, **kwargs) \
            -> tp.AsyncIterator[sr.AudioData]:
        loop = asyncio.get_running_loop()
        q_in = asyncio.Queue()
        stop = asyncio.Event()

        def cb():
            for data in self.generate():
                loop.call_soon_threadsafe(q_in.put_nowait, data)
            loop.call_soon_threadsafe(stop.set)
        coro = asyncio.to_thread(cb)
        asyncio.create_task(coro)
        with self.async_register_stop_handler():
            while True:
                can_stop = stop.is_set()
                is_empty = q_in.empty()
                if can_stop and is_empty:
                    break
                try:
                    yield q_in.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(1)

    def stop(self):
        self.stop_signal.set()

    @contextmanager
    def register_stop_handler(self, signum=signal.SIGINT):
        original_handler = signal.getsignal(signum)

        def handler(signum, _):
            self.stop()
            signame = signal.Signals(signum).name
            print(f'Signal handler called with signal {
                  signame} ({signum}) {self.stop_signal.is_set()}')

        signal.signal(signum, handler)
        try:
            yield
        finally:
            signal.signal(signum, original_handler)

    @contextmanager
    def async_register_stop_handler(self, signum=signal.SIGINT):
        loop = asyncio.get_event_loop()

        def audio_abort_handler(signum):
            self.stop()
            signame = signal.Signals(signum).name
            print(f'Signal handler called with signal {signame} ({signum})')

        self.stop_signal.clear()
        loop.add_signal_handler(signum, functools.partial(
            audio_abort_handler, signum=signum))
        try:
            yield
        finally:
            loop.remove_signal_handler(signum)

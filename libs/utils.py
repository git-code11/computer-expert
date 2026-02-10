import typing as tp
import numpy as np
import speech_recognition as sr
import asyncio


T = tp.TypeVar("T")


def async_to_sync_iterable(async_iterator: tp.AsyncIterator[T]) -> tp.Iterator[T]:
    """
    Converts an asynchronous iterator into a synchronous iterator.
    """
    # Manages the event loop for the duration of the iteration
    with asyncio.Runner() as runner:
        while True:
            try:
                # Runs the next iteration of the async iterator synchronously
                result = runner.run(anext(async_iterator))
                yield result
            except StopAsyncIteration:
                break
            except Exception as e:
                print(f"Unkown Exception: {e}")
                break


class AudioDivider:
    def __init__(self, time_split_ms: int | None = None):
        self.sample_width = 2
        self.time_split_ms = time_split_ms

    async def run(self, data: tuple[int, np.ndarray]):
        sample_rate, data = data
        if not self.time_split_ms:
            if data.ndim > 1:
                data = data[:, 1]
            yield sr.AudioData(
                bytes(data), sample_rate, self.sample_width)
            return

        frame_size = self.time_split_ms * sample_rate
        for i in range(0, data.size, frame_size):
            sample_data = data[i:i+frame_size]
            sample_data = sr.AudioData(
                sample_data, sample_rate, self.sample_width)
            yield sample_data

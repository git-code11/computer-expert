import typing as tp
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
import speech_recognition as sr
import sys


class TextStreamGenerator:

    def __init__(self, max_words=0, retries=1):
        self.recognizer = sr.Recognizer()
        self.max_words = max_words
        self.retries = retries

    def run(
        self,
        audio_stream: tp.Iterator[sr.AudioData]
    ) -> tp.Iterator[str]:
        full_text = ""
        with ThreadPoolExecutor(max_workers=3) as executor:
            all_text = executor.map(self.run_stt, audio_stream)
            for text in all_text:
                if self.max_words == 0:
                    yield text
                elif text and \
                        len(full_text.split(" ")) >= self.max_words:
                    yield full_text
                    full_text = ""
            yield full_text

    async def async_run(
        self,
        input: tp.AsyncIterator[sr.AudioData],
        **kwargs
    ) -> tp.AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        q_in = queue.Queue()
        q_out = asyncio.Queue()
        stop = asyncio.Event()

        def cb():
            audio = q_in.get()
            try:
                result = self.audio_to_text(audio)
                loop.call_soon_threadsafe(q_out.put_nowait, result)
            finally:
                q_in.task_done()

        async def action():
            with ThreadPoolExecutor(max_workers=3) as pool:
                async for audio in input:
                    q_in.put(audio)
                    loop.run_in_executor(pool, cb)
                    # await asyncio.sleep(0.1)

            await asyncio.to_thread(q_in.join)
            stop.set()

        coro = action()
        asyncio.create_task(coro)
        full_text = ""
        while True:
            can_stop = stop.is_set()
            is_empty = q_out.empty()
            if can_stop and is_empty:
                break
            elif is_empty:
                await asyncio.sleep(0.5)
                continue
            text = await q_out.get()
            if self.max_words == 0:
                yield text
            elif text and \
                    len(full_text.split(" ")) >= self.max_words:
                yield full_text
                full_text = ""
        yield full_text

    def audio_to_text(self, audio: sr.AudioData):
        # received audio data, now we'll recognize it using Google Speech Recognition
        text = None
        retries = self.retries
        while retries >= 0:
            try:
                # print(audio.frame_data)
                text = self.recognizer.recognize_google(audio)
                break
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio",
                      file=sys.stderr)
                break
            except sr.RequestError as e:
                print(
                    "Could not request results from "
                    "Google Speech Recognition service; {0}"
                    .format(e), file=sys.stderr)
            except Exception as e:
                print(
                    f"Unexpected Exception: {e}", file=sys.stderr)
            print("retrying")
            retries -= 1
        return text

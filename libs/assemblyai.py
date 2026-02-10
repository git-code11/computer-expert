import typing as tp
import os
import asyncio
from speech_recognition import AudioData
from contextlib import asynccontextmanager
import speech_recognition as sr
import websockets
import json
import httpx


class STTResult(tp.Dict):
    message: str
    partial: bool


class AssemblySTT:
    def __init__(self,
                 sample_rate: int,
                 format_true: bool = True,
                 api_key: str | None = None):
        self.config = dict(
            sample_rate=sample_rate,
            format_turns=True
        )
        self.base_url = "https://api.assemblyai.com"
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")

    async def convert(self, audio: sr.AudioData) -> str:
        headers = dict(
            authorization=self.api_key
        )

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers
        ) as client:
            audio_data = audio.get_wav_data()
            response = await client.post("/v2/upload",
                                         data=audio_data)
            audio_url = response.json()
            audio_url = audio_url["upload_url"]

            data = {
                "audio_url": audio_url,
                "speech_models": ["universal"]
            }

            response = await client.post("/v2/transcript", json=data)
            transcript_id = response.json()['id']
            polling_endpoint = "/v2/transcript/" + transcript_id
            while True:
                transcription_result = await client.get(polling_endpoint)
                transcription_result = transcription_result.json()
                transcript_text = transcription_result['text']
                if transcription_result['status'] == 'completed':
                    return transcript_text
                elif transcription_result['status'] == 'error':
                    raise RuntimeError(f"Transcription failed: {
                                       transcription_result['error']}")
                else:
                    await asyncio.sleep(2)

    async def async_run(self, input: tp.AsyncIterator[bytes]) -> tp.AsyncIterator[str]:
        async for audio in input:
            data = await self.convert(audio)
            print(f"In STT RUN {data=}")
            if data is not None:
                yield data


class AssemblySTTStream:
    def __init__(self,
                 sample_rate: int,
                 format_true: bool = True,
                 retries: int = 2,
                 api_key: str | None = None):
        self.config = dict(
            sample_rate=sample_rate,
            format_turns=True
        )
        self.uri = "wss://streaming.assemblyai.com/v3/ws?"
        self._ws: websockets.ClientProtocol | None = None
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.retries = retries

    async def send(self, ws: websockets.ClientProtocol, data: bytes):
        await ws.send(data)

    async def recieve(self, ws: websockets.ClientProtocol) \
            -> tp.AsyncIterator[STTResult]:
        async for raw_message in ws:
            message = json.loads(raw_message)

            if message["type"] == "Turn":
                print(f"{message=}")
                partial = not message.get("turn_is_formatted")
                yield dict(message=message["transcript"], partial=partial)

    @asynccontextmanager
    async def start(self):
        try:
            print("Starting")
            ws = await websockets.connect(
                self.uri,
                additional_headers={"Authorization": self.api_key}
            )
            print("Connected.")
            yield ws
        finally:
            if not ws.close:
                ws.close()

    async def async_run(self, input: tp.AsyncIterator[AudioData], ):
        q_in = asyncio.Queue()
        evt = asyncio.Event()

        async def send_audio():
            try:
                async for audio_chunk in input:
                    print(f"{audio_chunk=}")
                    if evt.is_set():  # break when error occured
                        break
                    await q_in.put(audio_chunk)
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Audio Exception: {e}")
            finally:
                q_in.put_nowait(None)

        async def process(ws: websockets.ClientProtocol, close: tp.Callable[[float], None]):
            while True:
                data = await q_in.get()
                if data is None:
                    # Schedule websocket to close
                    await asyncio.sleep(0.1)
                    close(5)  # after 5 seconds
                    break
                print(f"{data}")
                print(f"{data.frame_data=}")
                try:
                    await self.send(ws, data.frame_data)
                except Exception as e:
                    evt.set()  # Break on error
                    print(f"Audio Exception: {e}")

        async with websockets.connect(
            self.uri,
            additional_headers={
                "Authorization": self.api_key}
        ) as ws:

            # async for ws in websockets.connect(
            #     self.uri,
            #     additional_headers={
            #         "Authorization": self.api_key}
            # ):
            cm = asyncio.timeout(None)

            def close(tsec: float):
                deadline = asyncio.get_running_loop().time() + tsec
                cm.reschedule(deadline)

            task1 = asyncio.create_task(send_audio())
            task2 = asyncio.create_task(process(ws, close))
            try:
                async with cm:
                    async for result in self.recieve(ws):
                        print(f"{result=}")
                        yield result
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                evt.set()  # Stop on connection closed
            await task1
            await task2


if __name__ == "__main__":
    from libs.audio import AudioStreamGenerator
    from langchain_core.runnables import RunnableLambda, RunnableGenerator
    from dotenv import load_dotenv
    load_dotenv()
    runnable_cfg = {"configurable": {"thread_id": "1"}}
    sample_rate = 16000
    audio_gen = AudioStreamGenerator(
        sample_rate=sample_rate, chunk_duration_ms=800)
    stt_gen = AssemblySTTStream(sample_rate=sample_rate)
    runner = \
        RunnableLambda(audio_gen.async_run) | \
        RunnableGenerator(stt_gen.async_run)

    async def main():
        async for result in runner.astream(None, config=runnable_cfg):
            print(result)

    asyncio.run(main())

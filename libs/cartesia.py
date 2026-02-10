import cartesia
import typing as tp
import httpx
import speech_recognition as sr
import io
import os


class CartesiaTTS:
    def __init__(self, api_key: str | None = None, model_id: str | None = 'sonic-3',
                 sample_rate: int = 24000,):
        self.sample_rate = sample_rate
        self.model_id = model_id
        self.api_key = api_key or os.getenv('CARTESIA_API_KEY')
        self.base_url = "https://api.cartesia.ai/tts/"

    def create_payload(self, text: str) -> dict:
        payload = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                # "id": "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
                "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02"
            },
            "output_format": {
                "container": "wav",
                "sample_rate": self.sample_rate,
                "encoding": "pcm_s16le",
            },
            "language": "en",
            "generation_config": {
                "volume": 1,
                "speed": 1,
                "emotion": "neutral"
            },
            "save": False,
            "pronunciation_dict_id": None,
            "speed": "normal"
        }
        return payload

    async def run(self, input: str) \
            -> sr.AudioData:
        headers = {
            "Cartesia-Version": "2025-04-16",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers
        ) as client:
            data = self.create_payload(input)
            response = await client.post(
                '/bytes',
                json=data
            )
            data = await response.aread()
            if response.status_code != 200:
                raise Exception(f"CartesiaTSS: {data}")
            data_byte = io.BytesIO(data)
            # with open('data.wav', 'wb') as file:
            #     file.write(data_byte.getvalue())
            return sr.AudioData.from_file(data_byte)

    async def async_run(self, input: tp.AsyncIterator[str]) \
            -> tp.AsyncIterator[sr.AudioData]:
        async for text in input:
            print(f"AI Text {text}")
            yield await self.run(text)

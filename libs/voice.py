import typing as tp
import asyncio
from google import genai
from google.genai import types
import speech_recognition as sr
import io


class VoiceGenerator:
    def __init__(self):
        self.client = genai.Client()

    def text_to_speech(self, text: str) -> bytes:
        responses = self.client.models.generate_content_stream(
            model="gemini-2.5-flash-preview-tts",
            contents=f"Say cheerfully: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',
                        )
                    )
                ),
            )
        )

        file = io.BytesIO()
        for response in responses:
            data = response.candidates[0].content.parts[0].inline_data.data
            file.write(data)
        return file.getvalue()

    async def run(self, input: str) -> bytes:
        data = await asyncio.to_thread(self.text_to_speech, input)
        yield sr.AudioData(data,  sample_rate=24000, sample_width=2)

    async def async_run(self, input: tp.AsyncIterator[str]) \
            -> tp.AsyncIterator[bytes]:
        async for text in input:
            yield self.run(text)

import typing as tp
import os
from pathlib import Path
import asyncio
from uuid import uuid4
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from libs.audio import AudioStreamGenerator
from libs.text import TextStreamGenerator
from libs.agent import CropAgent

BASE_DIR = Path.cwd()
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
DOC_FILEPATH = BASE_DIR / "crop.xlsx"
STORE_PATH = BASE_DIR / "local_store"
CHECKPOINT_PATH = BASE_DIR / "local_checkpoint.db"


async def main():
    user_thread_id = uuid4()
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}

    crop_agent = CropAgent(DOC_FILEPATH, STORE_PATH, CHECKPOINT_PATH)
    crop_agent.init(DEBUG)

    correction_agent = create_agent(
        crop_agent.llm,
        system_prompt="You are an autogrammar corrector, your goal is to only return the corrected speech in raw text, "
        "if the grammar cannot be structured or completed return the text `NONE`"
        "the grammar is always pertaining to agriculture sector"
    )

    async def correct(input: tp.AsyncIterator[str]) -> str:
        full_text = str.join(' ', [text async for text in input if text is not None])
        result = await correction_agent.ainvoke(
            {"messages": [HumanMessage(content=full_text)]})
        return result['messages'][-1]

    audio_gen = AudioStreamGenerator()
    stt_gen = TextStreamGenerator()

    runner = \
        RunnableGenerator(audio_gen.async_run) | \
        RunnableGenerator(stt_gen.async_run) | \
        RunnableLambda(correct)

    async for result in runner.astream(None):
        print(result)
    # async for audio in audio_gen.async_run([]):
    #     text = stt_gen.audio_to_text(audio)
    #     print(text)
asyncio.run(main())

import os
from pathlib import Path
import asyncio
from uuid import uuid4
from langchain_core.runnables import RunnableGenerator, \
    RunnableLambda, RunnableBranch
from libs.audio import AudioStreamGenerator
from libs.text import TextStreamGenerator
from libs.agent import ComputerAgent
from libs.correction_agent import CorrectionAgent

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path.cwd()
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
DOC_FILEPATH = BASE_DIR / "crop.xlsx"
STORE_PATH = BASE_DIR / "local_store"
CHECKPOINT_PATH = BASE_DIR / "local_checkpoint.db"


async def main():
    user_thread_id = uuid4()
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}

    crop_agent = ComputerAgent(DOC_FILEPATH, STORE_PATH, CHECKPOINT_PATH)
    crop_agent.init(DEBUG)

    correction_agent = CorrectionAgent()
    audio_gen = AudioStreamGenerator()
    stt_gen = TextStreamGenerator()

    agent_runner = RunnableLambda(correction_agent.run_async) | \
        RunnableLambda(crop_agent.run_async)

    runner = \
        RunnableLambda(audio_gen.async_run) | \
        RunnableGenerator(stt_gen.async_run) | \
        RunnableBranch(
            (lambda x: isinstance(x, str) and
                len(x.strip()) > 0,
             agent_runner),
            lambda x: "Failed to capture input",
        )

    async for result in runner.astream(None, config=runnable_cfg):
        print(result)

asyncio.run(main())

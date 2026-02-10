import gradio as gr
import os
import asyncio
from pathlib import Path
from langchain_core.runnables import RunnableGenerator, \
    RunnableLambda, RunnableBranch
from libs.agent import ComputerAgent
from libs.correction_agent import CorrectionAgent
from libs.assemblyai import AssemblySTT
from libs.text import TextStreamGenerator
from libs.utils import AudioDivider
from libs.voice import VoiceGenerator
from libs.cartesia import CartesiaTTS
import speech_recognition as sr
import tempfile
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()


BASE_DIR = Path.cwd()
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
DOC_FILEPATH = BASE_DIR / "data.xlsx"
STORE_PATH = BASE_DIR / "local_store"
CHECKPOINT_PATH = BASE_DIR / "local_checkpoint.db"
sample_rate = 44100

MODEL_NAME = "google_genai:gemini-2.5-flash-lite"

# MODEL_NAME = "deepseek"

crop_agent = ComputerAgent(DOC_FILEPATH, STORE_PATH,
                           CHECKPOINT_PATH, model_name=MODEL_NAME)
crop_agent.init(DEBUG)

correction_agent = CorrectionAgent(model_name=MODEL_NAME)

stt_gen = AssemblySTT(sample_rate=sample_rate)
# stt_gen = TextStreamGenerator()

audio_gen = AudioDivider()
# tts_gen = VoiceGenerator()
tts_gen = CartesiaTTS()

crop_runner = RunnableLambda(crop_agent.run_async)

agent_runner = RunnableLambda(correction_agent.run_async) \
    | crop_runner

stt_runner = RunnableLambda(audio_gen.run) \
    | RunnableGenerator(stt_gen.async_run)

tts_runner = RunnableGenerator(tts_gen.async_run)


def condition_branch_runner(runner):
    return RunnableBranch(
        (lambda x: isinstance(x, str) and
         len(x.strip()) > 0,
         runner),
        lambda x: "Failed to capture input"
        # lambda x: await tts_gen.run("Failed to capture input"),
    )


# AGENT -> TTS
branch_runner = condition_branch_runner(agent_runner | tts_runner)

# AGENT -> TEXT
branch_text_runner = condition_branch_runner(agent_runner)


# STT -> AGENT -> TTS
agent_voice_runner = stt_runner \
    | branch_runner


async def text_chatting(message, history, request: gr.Request):
    user_thread_id = request.session_hash
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}
    full_text = []
    try:
        async for value in branch_text_runner.astream(message['text'], config=runnable_cfg):
            full_text.append(value)
            joined_text = str.join("", full_text)
            yield gr.ChatMessage(
                content=joined_text,
                role="assistant"
            ), joined_text
    except Exception as e:
        yield f"Error Occured {e}", None


async def voice_chatting(audio, request: gr.Request):
    user_thread_id = request.session_hash
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}
    async for audio_out in agent_voice_runner.astream(audio, config=runnable_cfg):
        yield audio_out.get_wav_data()


async def state_update(text, request: gr.Request):
    user_thread_id = request.session_hash
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}
    if text:
        async for audio in tts_runner.astream(text, config=runnable_cfg):
            yield audio.get_wav_data()


with gr.Blocks() as demo:
    state = gr.State([])
    input_audio = gr.Audio(
        label="Input Audio",
        sources=["microphone", "upload"],
        type="numpy",
        streaming=False,
        waveform_options=gr.WaveformOptions(
             waveform_color="#B83A4B",
             sample_rate=44100
        ),
        render=False
    )

    output_audio = gr.Audio(
        label="Output Audio",
        streaming=True,
        autoplay=True,
        render=False
    )
    chat_section = gr.ChatInterface(
        fn=text_chatting,
        multimodal=True,
        # save_history=True,
        chatbot=gr.Chatbot(
            value=[
                gr.ChatMessage(
                    role="assistant",
                    content="Welcome! I’m your Computer Hardware Diagnostic Assistant.",
                    # metadata={"title":  "🧠 Thinking"}
                )
            ],
            height=400),
        # textbox=gr.Textbox(),
        title="Crop Assistant AI",
        description="Crop Assistant chatbot with multimodal input",
        additional_outputs=[state]
    )

    with gr.Row():
        input_audio.render()
        output_audio.render()

    # btn = gr.Button("Check")
    #
    # @btn.click(inputs=state, outputs=output_audio)
    # async def clicked(input):
    #     time_ms = int(1000 * len(english_data.frame_data) /
    #                   english_data.sample_rate)
    #     for i in range(0, time_ms, 1000):
    #         data = english_data.get_segment(i, i+1000).get_wav_data()
    #         yield data
    #         await asyncio.sleep(0.01)
    #
    input_audio.stop_recording(
        fn=voice_chatting,
        inputs=[input_audio],
        outputs=[output_audio]
    ).then(lambda: (None, None), outputs=[input_audio, output_audio])

    state.change(state_update, inputs=[state], outputs=[output_audio])


app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )

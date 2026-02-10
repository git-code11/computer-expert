import os
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain_community.document_transformers.markdownify import MarkdownifyTransformer
# from langchain_classic.agents import AgentExecutor
from libs.document import CropDataLoader
# from langchain_classic.memory import SQLChatMessageHistory, StreamlitChatMessageHistory

load_dotenv()

BASE_DIR = Path.cwd()
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
DOC_FILEPATH = BASE_DIR / "crop.xlsx"

if DEBUG:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    checkpointer = InMemorySaver()
    store = InMemoryStore()
else:
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    # from langchain_community.storage import SQLStore
    from langchain_classic.storage import LocalFileStore
    conn = sqlite3.connect("local_checkpoint.db", check_same_thread=False)
    checkpointer = SqliteSaver()
    store = LocalFileStore(BASE_DIR / "local_store")


loader = CropDataLoader(DOC_FILEPATH)
crop_documents = loader.load()

merged_documents = "\n\n".join(crop.page_content for crop in crop_documents)


system_prompt_template = SystemMessagePromptTemplate.from_template([
    "You are an expert in botany with vast experience in analyzing crops, "
    "you can tell the kind of disease affecting the plant and what might have caused it, "
    "you also know how to manage risk and remedy to fix any issues around the crop, "
    "You give concise information and also ask for question to narrow the question to a specific problem."
    "Also ensure the response provided is minimal, a complete sentence and also direct based on the user issues.",
    "This is a record manual that the botanist is meant to explictly follow and no external knowledge should be used apart from these:\n {documents}"
])

system_prompt = system_prompt_template.format(
    documents=merged_documents)

model_name = "google_genai:gemini-2.5-flash-lite"
llm = model = init_chat_model(model_name)

main_agent = create_agent(
    model=llm,
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    store=store
)

user_thread_id = uuid4()
runnable_cfg = {"configurable": {"thread_id": user_thread_id}}

while True:
    user_query = input("(user)>")
    if user_query:
        inputs = dict(
            messages=[HumanMessage(content=user_query)]
        )

        for chunk in main_agent.stream(inputs, config=runnable_cfg, stream_mode="updates"):
            print(chunk)

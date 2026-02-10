from langchain.tools import tool
import typing as tp
import os
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents.base import Document
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models.base import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langgraph.config import RunnableConfig
from libs.document import ComputerDataLoader
from langchain_community.retrievers import BM25Retriever

SYSTEM_PROMPT = """You are a conversational Computer Hardware Diagnostic Assistant.

Your role is to talk with users in a friendly, clear manner and help identify computer hardware problems based on symptoms they describe.

Workflow

Read the user’s message and identify the key symptoms or device mentioned.

Make a tool call to retrieve relevant documentation based on the user’s query before giving any diagnosis or actions.

Use only the retrieved documents to determine the most likely cause.

Respond to the user following the rules below.

Behavior

Greet the user briefly and ask what problem they are experiencing if the issue is unclear.

Ask short follow-up questions when symptoms are insufficient.

Prioritize the most visible or clearly reported symptoms.

Diagnosis rules

Use confidence words such as likely, possibly, or less likely based only on matches in the retrieved documents.

Do not use outside knowledge or assumptions.

Responses

Provide ACTION STEPS ONLY when giving solutions.

Use short, clear sentences.

Keep language simple and practical.

Do not include explanations, theory, or document references.

Scope

Stay strictly within computer hardware troubleshooting.

Tools

Always retrieve relevant documentation using the available retrieval tool before diagnosing.

Base all conclusions strictly on the retrieved content."""


class ComputerAgent:
    agent: CompiledStateGraph
    llm: BaseChatModel
    store: BaseStore
    checkpoint: BaseCheckpointSaver
    loader: BaseLoader
    docs: list[Document]
    serialized_docs: str
    retriever: BM25Retriever

    system_prompt_template = SystemMessagePromptTemplate.from_template(
        SYSTEM_PROMPT)

    def __init__(self, doc_path: os.PathLike,
                 store_path: os.PathLike | None = None,
                 checkpoint_path: os.PathLike | None = None,
                 model_name: str | None = None
                 ):
        self.model_name = model_name or "google_genai:gemini-2.5-flash-lite"
        self.doc_path = doc_path
        self.store_path = store_path
        self.checkpoint_path = checkpoint_path

    def init(self, debug=False):
        if debug:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.store.memory import InMemoryStore
            self.checkpointer = InMemorySaver()
            self.store = InMemoryStore()
        else:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            # from langchain_community.storage import SQLStore
            from langchain_classic.storage import LocalFileStore
            if self.checkpoint_path:
                raise Exception("Provide Checkpoint path for sqlite DB")
            self.conn = sqlite3.connect(self.checkpoint_path,
                                        check_same_thread=False)
            self.checkpointer = SqliteSaver(self.conn)
            self.checkpointer.setup()
            if self.store_path:
                raise Exception("Provide Store path")
            self.store = LocalFileStore(self.store_path)

        # Load document
        self.loader = ComputerDataLoader(self.doc_path)
        self.docs = self.loader.load()
        self.retriever = BM25Retriever.from_documents(self.docs)
        self.serialized_docs = ""

        # Create an agent
        self.agent = self.setup_agent()

    def search_database_bm25(self):
        @tool
        def search_database_bm25(query: str, limit: int = 10) -> str:
            """Search the computer dianosis manual

            Args:
                query: Search terms to look for
                limit: Maximum number of results to return
            """
            result = self.retriever.invoke(query)
            if len(result) == 0:
                return "No Data Found matching query"
            merged_result = str.join(
                "\n\n", [doc.page_content for doc in result])
            return merged_result
        return search_database_bm25

    def get_system_prompt(self):
        return self.system_prompt_template.format(
            documents=self.serialized_docs)

    def setup_agent(self):
        model_name = "google_genai:gemini-2.5-flash-lite"
        self.llm = init_chat_model(model_name)
        system_prompt = self.get_system_prompt()
        main_agent = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
            checkpointer=self.checkpointer,
            store=self.store,
            tools=[self.search_database_bm25()]
        )

        return main_agent

    async def run_async(self, user_query: str,
                        config: RunnableConfig | None = None,
                        **kwargs) -> tp.AsyncIterator[str]:
        print(f"{user_query=}")
        inputs = dict(
            messages=[HumanMessage(content=user_query)]
        )

        async for message, _ in \
            self.agent.astream(
                inputs,
                config,
                **dict(**kwargs, stream_mode="messages")):
            if message.text:
                yield message.text


if __name__ == "__main__":
    from uuid import uuid4
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()
    BASE_DIR = Path.cwd()
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    DOC_FILEPATH = BASE_DIR / "data.xlsx"
    STORE_PATH = BASE_DIR / "local_store"
    CHECKPOINT_PATH = BASE_DIR / "local_checkpoint.db"

    user_thread_id = uuid4()
    runnable_cfg = {"configurable": {"thread_id": user_thread_id}}
    crop_agent = ComputerAgent(DOC_FILEPATH, STORE_PATH, CHECKPOINT_PATH)
    crop_agent.init(DEBUG)
    app = RunnableLambda(crop_agent.run_async)

    async def main():
        while True:
            user_query = input("(user)>")
            if user_query:
                async for output in app.astream(user_query, runnable_cfg):
                    await asyncio.sleep(0.1)  # fake latency here
                    print(output, end="")
            print()

    asyncio.run(main())

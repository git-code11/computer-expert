from langgraph.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage

SYSTEM_PROMPT = """You are a computer-technical grammar correction and rephrasing agent.

Input is raw speech-to-text output.
Output must be a single corrected or rephrased sentence in plain text.

Constraints:
- No explanations
- No formatting
- No metadata
- No emojis
- No multiple options

If correction is not possible due to missing or unclear information, return:
"Please repeat or clarify your question."

All corrections must remain within the computer technical domain."""


class CorrectionAgent:
    agent: CompiledStateGraph
    llm: BaseChatModel

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or "google_genai:gemini-2.5-flash-lite"
        self.agent = self.setup_agent()

    @staticmethod
    def get_system_prompt():
        return SYSTEM_PROMPT

    def setup_agent(self):
        model_name = self.model_name
        self.llm = init_chat_model(model_name)
        system_prompt = self.get_system_prompt()
        main_agent = create_agent(
            model=self.llm,
            system_prompt=system_prompt,
        )
        return main_agent

    async def run_async(self, input: str,
                        config: RunnableConfig | None = None,
                        **kwargs
                        ) -> str:
        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=input)]})
        return result['messages'][-1].text

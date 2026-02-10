from langchain.agents import create_agent
from langchain_core.prompts import ChatMessagePromptTemplate

main_agent_system_prompt = """
You are in expert in botany with vast experience in analyzing crops,
you can tell the kind of disease affecting the plant and what might have caused it,
you also know how to manage risk and remedy to fix any issues around the crop.
""".strip()


main_agent = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    # tools=[check_weather],
    system_prompt="You are a helpful assistant",
)

inputs = {"messages": [
    {"role": "user", "content": "what is the weather in sf"}]}
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)

# faiss.write_index(index, "index.bin")
# index2 = faiss.read_index("index.bin")  # index2 is identical to index


"""
- context
- state
- store
"""

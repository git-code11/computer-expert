from langchain_google_genai import GoogleGenerativeAIEmbeddings

import dotenv
dotenv.load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    # task_type=None
)
output = embeddings.embed_query("hello, world!")
print(f"The length of the output {len(output)=}")
print(output)

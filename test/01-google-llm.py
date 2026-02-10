from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
dotenv.load_dotenv()
model_name = "gemini-2.5-flash-lite"
llm = ChatGoogleGenerativeAI(model=model_name)
output = llm.invoke("Sing a ballad of LangChain.")
print(output)

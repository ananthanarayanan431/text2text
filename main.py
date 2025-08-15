from dotenv import load_dotenv

load_dotenv()

from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI()
print(llm.invoke("Hello"))

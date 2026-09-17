import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()


model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = model.invoke("What is PostgreSQL?")

print(response.content)
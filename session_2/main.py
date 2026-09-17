import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is PostgreSQL?")
]

response = model.invoke(messages)

print(response.content)
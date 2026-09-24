import os
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[HumanMessage]
    
model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def process(state: AgentState) -> AgentState:
    """This function processes user input and does further process."""    
    response = model.invoke(state["messages"])
    
    print(response.content)
    
    return state
    

graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

chatbot = graph.compile()

user_input = input("Enter your input (type '/bye' to end the session): ")

while user_input != "/bye":
    response = chatbot.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    user_input = input("Enter your input (type '/bye' to end the session): ")
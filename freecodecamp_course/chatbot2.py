import os
from typing import TypedDict, List, Union
from dotenv import load_dotenv

load_dotenv()

from langchain.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def process(state: AgentState) -> AgentState:
    """This function processes user input and does further process."""    
    response = model.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    conversation_history.append(AIMessage(content=response.content))
    
    print(f"\nAI: {response.content}")
    
    return state
    

graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

chatbot = graph.compile()

conversation_history = []

user_input = input("Enter your input (type '/bye' to end the session): ")

while user_input != "/bye":
    conversation_history.append(HumanMessage(content=user_input))
    response = chatbot.invoke({
        "messages": conversation_history
    })
    user_input = input("Enter your input (type '/bye' to end the session): ")
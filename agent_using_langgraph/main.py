from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    

@tool
def calculate_total(price: float, quantity:float):
    """Calculate the total price using price and quantity."""
    return quantity * price
 
    
def  call_llm(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    
    return {
        "messages": [response]
    }
    

tool_registry = {
    "calculate_total": calculate_total,
}


model_with_tools = model.bind_tools([calculate_total])
    
graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm", call_llm)


tool_node = ToolNode([calculate_total])
graph_builder.add_node("tools", tool_node)


graph_builder.add_edge(START, "llm")

graph_builder.add_edge("tools", "llm")


graph_builder.add_conditional_edges(
    "llm",
    tools_condition
)


graph = graph_builder.compile()

result = graph.invoke({
        "messages": [
            HumanMessage(
                content="Calculate total price of 8 units where per unit price is 35$."
            )
        ]
    })

for message in result["messages"]:
    print(type(message).__name__, ":", message.content)
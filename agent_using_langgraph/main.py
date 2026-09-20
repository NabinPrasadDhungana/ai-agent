from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
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

    
def tools_node(state: AgentState):
    print("Tools node reached")
    overall_tool_result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        
        if tool_name not in tool_registry:
            print(f"Unknown tool: {tool_name}")
            continue
        
        result = tool_registry[tool_name].invoke(arguments)
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
        )

        overall_tool_result.append(tool_message)
        
    return {
        "messages": overall_tool_result
    }


def should_continue(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "end"

model_with_tools = model.bind_tools([calculate_total])
    
graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm", call_llm)

graph_builder.add_node("tool", tools_node)

graph_builder.add_edge(START, "llm")

graph_builder.add_edge("tool", "llm")

graph_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tool",
        "end": END
    }
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
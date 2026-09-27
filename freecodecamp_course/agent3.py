import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

@tool
def add(a: int, b: int):
    """This function adds two numbers and returns the result."""
    print("\nTool is called!!")
    return a + b

tools = [add]

model_with_tools = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
).bind_tools(tools=tools)


def model_call(state: AgentState) -> AgentState:
    """This function calls the model and updates the state."""
    system_prompt = SystemMessage(content="You are a helpful assistant having very good problem solving ability.")
    
    response = model_with_tools.invoke([system_prompt] + state["messages"])
    
    print(f"\n AI: {response.content}")
    if response.tool_calls:
        print(f"\tCalled tools are:")
        for tools in response.tool_calls:
            print(f"\t\tName: {tools['name']}", end="")
            print(f"\tArguments: {tools['args']}")
        
    return {
        "messages": [response]
    }


tool_node = ToolNode([add])    

graph = StateGraph(AgentState)

graph.add_node("model", model_call)
graph.add_node("tools", tool_node)


graph.add_edge(START, "model")
graph.add_conditional_edges(
    "model",
    tools_condition,
    
    {
        "tools": "tools",
        "__end__": END
    }
)
graph.add_edge("tools", "model")


app = graph.compile()

app.invoke({
    "messages": [HumanMessage(content="Find the sum of 5 and 6 is equal to 10 or not.")]
})

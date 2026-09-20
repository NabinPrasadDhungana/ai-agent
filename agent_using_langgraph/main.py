from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from agent_loop_using_langchain.main import model

class AgentState(TypedDict):
    messages: list
    

def test_node(state: AgentState):
    print("Inside test_node")
    print(state["messages"])
    
    return {
        "messages": ["Hello from Node!"]
    }
    
    
def  call_llm(state: AgentState):
    response = model.invoke(state["messages"])
    
    return {
        "messages": [response]
    }
    
    
graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm", call_llm)

graph_builder.add_edge(START, "llm")

graph_builder.add_edge("llm", END)

graph = graph_builder.compile()

result = graph.invoke({
        "messages": [
            HumanMessage(
                content="Explain PostgreSQL in one sentence."
            )
        ]
    })

print(result["messages"])

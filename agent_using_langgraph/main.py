from typing import TypedDict

from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: list
    

def test_node(state: AgentState):
    print("Inside test_node")
    print(state["messages"])
    
    return {
        "messages": ["Hello from Node!"]
    }
    
    
graph_builder = StateGraph(AgentState)

graph_builder.add_node("test", test_node)

graph_builder.add_edge(START, "test")

graph_builder.add_edge("test", END)

graph = graph_builder.compile()

result = graph.invoke({
        "messages": ["Hello LangGraph"]
    })

print(result)
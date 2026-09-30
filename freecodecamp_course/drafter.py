import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage ,SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Tools
@tool
def update(content: str) -> str:
    """This tool puts the contents provided by LLM to the document file."""
    global document_content
    document_content = content
    
    print("\n Tool called: update()")
    return f"Document has been updated successfully. The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """This tool saves the document with the given file name."""
    global document_content
    
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
        
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
            
        print("\n Tool called: save()")
        print("\n Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [update, save]


model_with_tools = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
).bind_tools(tools=tools)


def agent(state: AgentState) -> AgentState:
    global document_content
    
    system_prompt = SystemMessage(content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
        - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after modifications.
        
        The current document content is:{document_content}
        """)
    
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
        
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content=user_input)
        
    print(f"\n Current document state: \n {document_content}\n")
    
    response = model_with_tools.invoke([system_prompt] + list(state["messages"]) + [user_message])
    
    print(f"\nAI: {response.content}")
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print (f" USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {
        "messages": list(state["messages"]) + [user_message, response]
    }


def should_continue(state: AgentState) -> str:
    """This function conditionally return the edge of the graph."""  
    messages = state["messages"]
    
    if not messages:
        return "continue"  
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            print(f"\nTool result: {message.content}")
            return "exit"
    
    return "continue"

def print_messages (messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages [-3:]:
        if isinstance (message, ToolMessage):
            print (f" \n TOOL RESULT: {message.content}")
            

tool_node = ToolNode(tools=tools)

graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
        
    {
        "continue": "tools",
        "exit": END
    }
)
graph.add_edge("tools", "agent")

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER ======")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages (step["messages"])
            
    print("\n ===== DRAFTER FINISHED ===== :")


if __name__ == '__main__':
    run_document_agent()
# "Give me an email draft to apply for a leave for tomorrow from my office."

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


@tool
def get_customer(customer_id: int):
    """Get a customer's information using their customer ID."""

    customers = {
        1: {"id": 1, "name": "Ram", "email": "ram@example.com"},
        2: {"id": 2, "name": "Sita", "email": "sita@example.com"},
    }

    if customer_id not in customers:
        return f"Customer {customer_id} not found."

    return customers[customer_id]


@tool
def get_customer_order(customer_id: int):
    """Get the current order for a customer."""

    orders = {
        1: {"customer_id": 1, "items": 3, "price_per_item": 20},
        2: {"customer_id": 2, "items": 8, "price_per_item": 35},
    }

    if customer_id not in orders:
        return f"No order found for customer {customer_id}."

    return orders[customer_id]
 
    
def  call_llm(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    
    return {
        "messages": [response]
    }


model_with_tools = model.bind_tools([calculate_total, get_customer, get_customer_order])
    
graph_builder = StateGraph(AgentState)

graph_builder.add_node("llm", call_llm)


tool_node = ToolNode([calculate_total, get_customer, get_customer_order])
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
                content="Get customer 2's order."
                        "Then calculate the total price of that order."
                        "Finally tell me the customer's name and the total price."
            )
        ]
    })

for message in result["messages"]:
    print(type(message).__name__, ":", message.content)
import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

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

@tool
def get_customer(customer_id: int):
    """Get customer information by customer ID."""
    
    customers = {
        1: {
            "name": "Ram",
            "email": "ram@example.com"
        },
        2: {
            "name": "Sita",
            "email": "sita@example.com"
        }
    }

    return customers.get(customer_id)

@tool
def calculate_total(quantity: float, price: float):
    """Get total price by quantity and price."""
    return quantity * price

tool_registry = {
    "get_customer": get_customer,
    "calculate_total": calculate_total,
}

model_with_tools = model.bind_tools([get_customer, calculate_total])

response = model_with_tools.invoke("Calculate the total for 8 items costing 35 each.")

print(response.content)
print("---------------------------")
print(response.tool_calls)

tools = response.tool_calls

if tools:
    for toool in tools:
        tool_name = toool["name"]
        arguments = toool["args"]
        
        if tool_name in tool_registry:
            result = tool_registry[tool_name].invoke(arguments)    
            
        print(f"Tool call result for tool {tool_name} = {result}")

import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

load_dotenv()


model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


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

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Calculate the total for 8 items costing 35 each."),
]

response = model_with_tools.invoke(messages)

print(response.content)
print("---------------------------")
print(response.tool_calls)

tools = response.tool_calls

if tools:
    for tool_call in tools:
        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        
        if tool_name not in tool_registry:
            print(f"Unknown tool: {tool_name}")
            continue
        
        result = tool_registry[tool_name].invoke(arguments)
        
        tool_message = {
            "content": result,
            "tool_call_id": tool_call["id"],
        }
        messages.append(ToolMessage(**tool_message))
             
            
        print(f"Tool call result for tool {tool_name} = {result}")
        
final_answer = model_with_tools.invoke(messages)

print("---------------------")
print(final_answer.content)

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT_CONTENT = """
Get the information of customer 2 and calculate the total
price of 8 items costing 35 each. Also tell me did you answer according to the tool call results or you made it up, honestly?
"""


def success(data):
    return {
        "success": True,
        "data": data,
    }
    
def error(error):
    return {
        "success": False,
        "error": error,
    }

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

def get_customer(customer_id):
    if customer_id is None:
        return error("You must provide a customer_id.")
    
    if customer_id in customers:
        return success(customers[customer_id])
    
    return error(f"No customer with customer id {customer_id} exist.")

def calculate_total(quantity, price):
    if not isinstance(quantity, (int, float)) or not isinstance(price, (int, float)):
        return error("Both price and quantity should be in int/float types.")
    
    if quantity < 0 or price < 0:
        return error("Price or Quantity can not be less than 0.")
    
    return success(quantity * price)

tools = {
    "get_customer": get_customer,
    "calculate_total": calculate_total,
}

def execute_tool(tool_name, arguments:dict):
    if tool_name not in tools:
        return error("No such tool available!")
    try:
        print("\n---- Tool Execution -> 1 ----\n")
        result = tools[tool_name](**arguments)
        return result
    except Exception as e:
        return error(f"Tool execution failed: {e}")

# print(execute_tool("get_customer", {"customer_id": 1}))
# print(execute_tool("calculate_total", {"quantity": 15, "price": 10}))

tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Get customer information by customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The ID of the customer."
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_total",
            "description": "Calculate the total price from quantity and unit price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "number",
                        "description": "Number of items."
                    },
                    "price": {
                        "type": "number",
                        "description": "Price of one item."
                    }
                },
                "required": ["quantity", "price"],
                "additionalProperties": False
            }
        }
    }
]

client = OpenAI(base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

messages = [
    {
        "role": "user",
        "content": PROMPT_CONTENT
    },
]

response = client.chat.completions.create(
    model="openrouter/free",
    messages=messages,
    tools=tool_schemas,
)

message = response.choices[0].message

# Append LLM response to the messages dictionary
messages.append(message)

if message.tool_calls:
    tool_usage = []
    for tool_call in message.tool_calls:      
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        tool_call_response = execute_tool(
            tool_name=tool_name,
            arguments=arguments,
        )
        
        tool_usage.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_call_response),
        })
        
    messages.extend(tool_usage)  

response_2 = client.chat.completions.create(
    model="openrouter/free",
    messages=messages,
    tools=tool_schemas,
)

print(response_2.choices[0].message.content)

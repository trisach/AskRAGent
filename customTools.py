import requests
from langchain_core.tools import tool
@tool
def add(a:float , b :int) -> int:
    """Adds a and b
    Args :
        a: first number
        b: first number"""
    return a + b

@tool
def subtract(a:float , b :float) -> int:
    """Subtracts b from a
    Args :
        a: first number
        b: first number"""
    return a - b

@tool
def subtract(a:float , b :float) -> int:
    """Subtracts b from a
    Args :
        a: first number
        b: first number"""
    return a - b

@tool
def multiply(a:float , b :float) -> int:
    """Multiply a with b
    Args :
        a: first number
        b: first number"""
    return a * b

@tool
def divide(a:float , b :float) -> int:
    """Divides a by b
    Args :
        a: first number
        b: first number"""
    if b!=0:
        return a / b
    else:
        raise ValueError("Cannot divide by 0")

@tool
def modulus(a:int , b :int) -> int:
    """Gets modulus of 2 numbers
    Args :
        a: first number
        b: first number"""
    return a % b

@tool
def power(a:int , b=1) -> int:
    """Raises a to the power of b, can be used to find square root
    of a by raising it to the power of inverse of 2
    Args :
        a: first number
        b: first number"""
    return a**b

# @tool
# def square_root(a:int) -> int:
#     """Returns square root of 
#     Args :
#         a: first number"""
#     return a % b

@tool
def dictionary(word : str) -> str:
    """Fetches a JSON containing definition
      and synonym of the given word
      Args : 
        word : a word whose definition is to be found"""
    try:
        url = f'https://api.dictionaryapi.dev/api/v2/entries/en/{word}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            definition = data[0]["meanings"][0]["definitions"][0]["definition"]
            return definition
        else:
            return f"Error unable to find the word '{word}.Please check the word or try another."
    except Exception as e:
        return f'An Error Occurred {str(e)}'

# Export each tool individually for easy import in other scripts
__all__ = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "modulus",
    "power",
    "dictionary",
]
all_tools = [add, subtract, divide, multiply, power, modulus,dictionary]
math_tools = [add, subtract, divide, multiply, power, modulus]
dict_tools = [dictionary]

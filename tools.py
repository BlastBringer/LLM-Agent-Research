from langchain.tools import tool 

@tool 
def calculator(expression: str) -> str:
    """Evaluate a math expression safely and return the result."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error : {e}"
    

print(calculator("2+3*4"))
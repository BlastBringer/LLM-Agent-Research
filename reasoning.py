import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from Math import agent  # import the math agent we built

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm_reason = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

def reasoning_agent(problem: str):
    # Ask Gemini to break down the problem into reasoning steps
    reasoning_prompt = f"""
    You are a reasoning assistant. Solve this problem step by step.
    If a calculation is needed, clearly state the expression and send it to the math agent.
    
    Problem: {problem}
    """
    reasoning = llm_reason.invoke(reasoning_prompt)
    print("Reasoning:", reasoning.content)

    # extract a math expression (for simplicity, assume it's between backticks)
    import re
    exprs = re.findall(r"`([^`]+)`", reasoning.content)
    
    results = []
    for expr in exprs:
        answer = agent.run(expr)
        results.append((expr, answer))
        print(f"Math Agent: {expr} = {answer}")

    return reasoning.content, results


if __name__ == "__main__":
    problem = input("Enter your Math Problem: ")
    reasoning, results = reasoning_agent(problem)
    print("\nFinal Answer:", results[-1][1] if results else "No calculation found")
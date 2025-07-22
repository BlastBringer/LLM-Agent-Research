# --- LangChain ReAct Agent for Math Reasoning ---
# Required: pip install langchain-openai langchain-community sympy tiktoken

from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI  # Or: from langchain_openai import ChatOpenAI
from langchain.utilities import SerpAPIWrapper  # Or: from langchain_community.utilities import SerpAPIWrapper
from sympy import symbols, Eq, solve
import math

# --- Tool Definitions ---

def calculator_tool(query: str) -> str:
    try:
        return str(eval(query))
    except Exception as e:
        return f"Calculator Error: {e}"

def symbolic_solver_tool(query: str) -> str:
    try:
        eqn_part, var_part = query.split(";")
        eqn = eqn_part.strip()
        var = var_part.replace("solve for", "").strip()
        var = symbols(var)
        lhs, rhs = eqn.split("=")
        equation = Eq(eval(lhs.strip()), eval(rhs.strip()))
        sol = solve(equation, var)
        return str(sol)
    except Exception as e:
        return f"Solver Error: {e}"

# --- Optional Web Search Tool (requires SerpAPI key) ---
try:
    search = SerpAPIWrapper()
    search_tool = Tool(
        name="Search",
        func=search.run,
        description="Useful for looking up formulas or concepts online"
    )
except Exception:
    search_tool = Tool(
        name="Search",
        func=lambda q: "🔍 (Pretend search result for: '" + q + "')",
        description="Stub for search tool"
    )

# --- Tool Wrappers ---
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Evaluates basic math expressions. Input: a valid Python expression like '3 * (4 + 2)'"
    ),
    Tool(
        name="SolveEquation",
        func=symbolic_solver_tool,
        description="Solves symbolic math equations. Input format: '2*x + 3 = 7; solve for x'"
    ),
    search_tool,
]

# --- LLM Setup ---
llm = ChatOpenAI(
    model_name="google/gemini-2.5-flash-lite-preview-06-17",
    temperature=0.2,
    openai_api_key="",
    openai_api_base="https://openrouter.ai/api/v1"
)

# --- Agent Initialization ---
agent = initialize_agent(
    tools,
    llm,
    agent_type="react-description",  
    verbose=True
)

# --- Main ---
if __name__ == "__main__":
    problem = input("Enter your math problem: ").strip()
    print("Solving...\n")
    response = agent.run(problem)
    print("\n✅ Final Answer:")
    print(response)

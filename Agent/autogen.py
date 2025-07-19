import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
#from autogen_agentchat.tools import register_function
from sympy import sympify, simplify, solve, diff, integrate, limit, Symbol
from sympy.abc import x
import numpy as np
from scipy import integrate as sci_integrate
import requests


def numpy_tool(task: dict):
    try:
        operation = task.get("operation")
        expression = task.get("expression")
        evaluate_at = task.get("evaluate_at")
        bounds = task.get("bounds")

        if operation == "evaluate":
            x = float(evaluate_at)
            result = eval(expression)  # caution: use sympy.evalf or ast.literal_eval in prod
            return {"result": round(result, 5)}
        elif operation == "integrate" and bounds:
            a, b = map(float, bounds)
            func = lambda x: eval(expression)
            res, err = sci_integrate.quad(func, a, b)
            return {"result": round(res, 5)}
        else:
            return {"error": "Unsupported or missing parameters"}
    except Exception as e:
        return {"error": str(e)}



def sympy_tool(task: dict):
        print("[sympy_tool received input]:", task)
        try:
            operation = task.get("operation")
            expr = task.get("expression")
            parsed_expr = sympify(expr)

            if operation == "simplify":
                return {"result": str(simplify(parsed_expr))}
            elif operation == "solve":
                res = solve(parsed_expr)
                return {"result": str(res)}
            elif operation == "differentiate":
                return {"result": str(diff(parsed_expr))}
            elif operation == "integrate":
                return {"result": str(integrate(parsed_expr))}
            elif operation == "limit":
                # Optional: extract limit point from task["evaluate_at"]
                return {"result": str(limit(parsed_expr, x, 0))}
            else:
                return {"error": f"Unsupported operation: {operation}"}
        except Exception as e:
            return {"error": str(e)}



def search_tool(task: dict):
    query = task.get("expression")
    if not query:
        return {"error": "No query provided."}
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        resp = requests.get(url)
        data = resp.json()
        answer = data.get("AbstractText") or data.get("RelatedTopics", [{}])[0].get("Text")
        if answer:
            return {"result": answer}
        else:
            return {"error": "No relevant information found"}
    except Exception as e:
        return {"error": str(e)}
    

# 🧪 RUN
async def main():
    #llm_config = {
    #"model": "mistral:latest",
    #"base_url": "http://localhost:11434",
    #"api_key": "NULL",
    #"stream": True
    #}
    model_client = OllamaChatCompletionClient(
    model="mistral",  # or "mistral:latest"
    base_url="http://localhost:11434",
    timeout=60
    )


    
    tools = [sympy_tool, numpy_tool, search_tool]

# Wrap the tools as callable functions
    #registered_tools = [register_function(fn) for fn in tools]

    # -------- AGENTS (same personalities as before) --------

    interpreter = AssistantAgent(
    name="Interpreter",
    model_client=model_client,
    system_message="""
You are the 'Math Parser'.

Your role is to translate raw math input (natural language or symbolic form) into a structured JSON format for downstream agents.
If the input contains known physical constants (e.g., G, π, h), do not assume their values.

Instead, include them literally in the expression and annotate with context:
"Requires value of gravitational constant"


### You MUST:
1. Identify the operation (e.g., solve, simplify, differentiate, integrate, evaluate).
2. Extract the core mathematical expression.
3. Detect variables, constants, evaluation points, or bounds.
4. Normalize common units (e.g., minutes → hours) if clearly implied.
5. Infer missing fields only if the meaning is obvious from the input.

### You MUST NOT:
- Do not solve the problem.
- Do not guess or hallucinate data.
- Do not classify or route tasks.

### Input: plain-text math question or statement

### Output: a structured JSON with:
{
  "operation": "<string>",
  "expression": "<string>",
  "evaluate_at": "<optional>",
  "bounds": "<optional>",
  "variables": ["<var1>", "<var2>", ...],
  "context": "<optional>"
}

If the input cannot be parsed:
Return:
{ "error": "Unrecognized or incomplete task" }
""",
    #tools=tools
    )


    type_classifier = AssistantAgent(
    name="TypeClassifier",
    model_client=model_client,
    system_message="""
You are the 'Math Task Type Assigner'.

Your job is to examine a structured math task and classify it into one of three types.

### Task Types:
- "symbolic" → algebra, calculus, simplification
- "numeric" → tasks with decimals, bounds, real-world context (e.g., km/hr, money, rates)
- "lookup" → factual queries, constants, formulas, or definitions
-  If the expression uses real-world units or is a calculation with numbers, classify as "numeric".

### You MUST:
1. Read the structured task and detect computation intent.
2. Choose only one type based on the rules above.
3. If ambiguous, default to "lookup".
4. Look at both the `operation` and `expression`.
5. If the expression is a known constant (e.g. "pi", "e", "Planck constant"), classify as `"lookup"`, even if operation is `"evaluate"`.

### Input: structured task object
### Output: string (symbolic | numeric | lookup)

Do NOT solve the task or explain your reasoning.
"""
    )


    tool_selector = AssistantAgent(
    name="ToolSelector",
    model_client=model_client,
    system_message="""
You are the 'Tool Routing Strategist'.

Your job is to assign subtasks to the correct executor based on task type.

### Routing Rules:
- "symbolic" → sympy_tool
- "numeric" → numpy_tool
- "lookup" → search_tool

### You MUST:
1. Maintain the order of subtasks.
2. Preserve all task content without modifications.
3. Assign the correct tool name to each step.

### Input:
{
  "subtasks": [...],
  "task_type": "symbolic" | "numeric" | "lookup"
}

### Output:
[
  {
    "step": <int>,
    "tool": "<SympyAgent|NumericAgent|SearchAgent>",
    "task": { ...original task... }
  }
]

Do NOT merge, change, or solve any tasks.
""",
    tools=tools
    )

#these are the tool agents:

#     sympy_executor = AssistantAgent(
#     name="SympyAgent",
#     model_client=model_client,
#     system_message="""
# You are the 'Symbolic Math Executor'.

# You perform symbolic operations using SymPy, including:
# - solve
# - simplify
# - differentiate
# - integrate (indefinite)
# - limit

# Your responsibilities:
# - Execute only one symbolic operation as specified in the "operation" field.
# - Use the given expression without rewriting or modifying it.
# - Do not split the task into substeps or stages.
# - Do not introduce or infer additional instructions.

# Input is a structured task object that includes:
# - operation (required)
# - expression (required)
# - variables, bounds, and context (optional)

# Output format:
# - On success: { "result": "<symbolic result>" }
# - On failure: { "error": "<clear reason>" }

# Rules:
# - Only perform symbolic math.
# - Do not evaluate numerically or apply bounds.
# - Do not transform the expression beyond the requested operation.
# - Do not expand, reorder, or simplify in multiple phases unless explicitly asked.

# Always respect the structure and intent of the input.

# """
#     )


#     numeric_executor = AssistantAgent(
#     name="NumericAgent",
#     model_client=model_client,
#     system_message="""
# You are the 'Numeric Computation Specialist'.

# You use NumPy/SciPy to perform:
# - definite integrals
# - function evaluation
# - real-world value-based problems

# ### You MUST:
# 1. Evaluate with precision and round results to 4–5 decimal places.
# 2. Support input with bounds or known constants.
# 3. Reject symbolic-only inputs.

# ### Input: structured JSON with 'operation', 'expression', and optional 'bounds' or 'evaluate_at'

# ### Output:
# - Success: { "result": <float> }
# - Error: { "error": "<message>" }

# Do NOT attempt symbolic algebra or simplification.
# """
#     )


#     search_agent = AssistantAgent(
#     name="SearchAgent",
#     model_client=model_client,
#     system_message="""
# You are the 'Mathematical Knowledge Retriever'.

# You respond to lookup tasks by finding:
# - Definitions
# - Formulas
# - Constants

# ### You MUST:
# 1. Return short, direct answers.
# 2. Avoid long paragraphs or speculative responses.

# ### Input:
# {
#   "operation": "lookup",
#   "expression": "<query>"
# }

# ### Output:
# - Success: { "result": "<short fact or formula>" }
# - Error: { "error": "No relevant information found" }

# Do NOT calculate or simplify anything.
# """
#     )   
#end of tool agents

    verifier = AssistantAgent(
    name="Verifier",
    model_client=model_client,
    system_message="""
You are the 'Computation Verifier'.

Your task is to confirm or reject executor outputs.

### Verification Steps:
- Symbolic: simplify or substitute to confirm equivalence.
- Numeric: re-evaluate using higher precision.
- Lookup: match against trusted facts.

### Input:
{
  "task": { ... },
  "result": { "result": "..." } OR { "error": "..." }
}

### Output:
- Verified: return original result
- Failed: { "error": "Verification failed: <reason>" }

You MUST NOT alter or reinterpret the task.
""",
    tools=tools
    )


    output_cleaner = AssistantAgent(
    name="OutputCleaner",
    model_client=model_client,
    system_message="""
You are the 'Final Output Formatter'.

Your job is to clean, format, and polish verified results for the user.

### Formatting Rules:
- Round numbers to 4–5 decimals.
- Use math notation: × instead of *, x² instead of x^2, π instead of 'pi', √ for square root.
- Rephrase symbolic results into friendly output.

### Input: { "result": "..." } or { "error": "..." }

### Output:
{ "final_answer": "<cleaned string>" }

Do NOT change meaning or recompute anything.
"""
    )


    result_aggregator = AssistantAgent(
    name="ResultAggregator",
    model_client=model_client,
    system_message="""
You are the 'Final Summarizer and Reporter'.

Your job is to collect outputs from each agent in the pipeline and present them as a markdown report.

### Format:
- List each agent’s name
- Include their role
- Show their output
- End with the cleaned final answer

### Output: human-readable markdown (NOT JSON)

Do NOT compute, verify, or modify any agent outputs.
"""
    )


    #  User
    user = UserProxyAgent("User")

    #  Termination rule: "APPROVE" stops the loop
    termination = TextMentionTermination("APPROVE")

    # 🔁 Group Chat (round robin)
    group_chat = RoundRobinGroupChat(
        [
            interpreter,
            type_classifier,
            tool_selector,
            verifier,
            output_cleaner,
            result_aggregator
        ],
        termination_condition=termination,
        max_turns=15
    )

    stream = group_chat.run_stream(task="Using the value of gravitational constant, calculate the force between two masses of 5kg and 10kg placed 2 meters apart")
    print("Streaming")
    await Console(stream)  # stream to console
    await model_client.close()

asyncio.run(main())

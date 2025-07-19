from sympy import sympify, simplify, solve, diff, integrate, limit, Symbol
from sympy.abc import x

def sympy_tool(task: dict):
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

def main():
    # Example test case (you can modify this)
    task = {"name":"sympy_tool","arguments":{"task":{"operation":"simplify", "expression":"(3*x + 2*x) - (x - 4)"}}}

    result = sympy_tool(task)
    print("Output:", result)

if __name__ == "__main__":
    main()

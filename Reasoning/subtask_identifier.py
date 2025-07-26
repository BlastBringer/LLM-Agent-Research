import json

class SubtaskIdentifier:
    """
    Analyzes parsed problem data to create a specific subtask for the agent.
    """
    def __init__(self):
        print("💡 Subtask Identifier initialized.")
    
    def identify_subtask(self, parsed_data: dict) -> dict:
        """
        Creates a structured subtask based on the parsed data.

        Args:
            parsed_data: The JSON output from your Problem Parser.

        Returns:
            A dictionary representing the command for the agent.
        """
        print("💡 Identifying subtask from parsed data...")

        # Handle error cases first
        if "error" in parsed_data:
            print("⚠️ Error in parsed data, creating general query subtask.")
            return {
                "tool_name": "GeneralQuery",
                "tool_input": parsed_data,
                "human_readable_goal": "Handle problematic input with general reasoning."
            }

        # Rule-based identification based on the keys in the parsed data
        if "equations" in parsed_data and ("target_variable" in parsed_data or "variables" in parsed_data):
            # This is a classic system of equations problem
            variables = []
            
            # Extract variables from different formats
            if "variables" in parsed_data:
                if isinstance(parsed_data["variables"], dict):
                    variables = list(parsed_data["variables"].keys())
                elif isinstance(parsed_data["variables"], list):
                    # Handle list format like ["x: variable description", "y: another variable"]
                    variables = [v.split(":")[0].strip() for v in parsed_data["variables"] if ":" in v]
            
            # Fallback to target_variable if no variables found
            if not variables and "target_variable" in parsed_data:
                variables = [parsed_data["target_variable"]]
            
            if not variables:
                print("⚠️ Warning: Equations found but no variables defined.")
                variables = ["x"]  # Default variable

            subtask = {
                "tool_name": "SymbolicSolver",
                "tool_input": {
                    "equations": parsed_data["equations"],
                    "variables_to_solve": variables
                },
                "human_readable_goal": f"Solve the system of equations for variables: {', '.join(variables)}."
            }
            print(f"✅ Created SymbolicSolver subtask for variables: {variables}")
            return subtask

        elif "expression_to_evaluate" in parsed_data:
            # This is a simple calculation problem
            subtask = {
                "tool_name": "Calculator",
                "tool_input": {
                    "expression": parsed_data["expression_to_evaluate"]
                },
                "human_readable_goal": f"Evaluate the mathematical expression: {parsed_data['expression_to_evaluate']}"
            }
            print("✅ Created Calculator subtask.")
            return subtask

        else:
            # Default or unknown task type
            print("⚠️ No specific subtask could be identified, using GeneralQuery.")
            return {
                "tool_name": "GeneralQuery",
                "tool_input": parsed_data,
                "human_readable_goal": "Perform a general reasoning task on the provided data."
            }

# --- Example Usage ---
if __name__ == "__main__":
    identifier = SubtaskIdentifier()

    # Example 1: A system of equations problem
    parsed_problem_1 = {
        "problem_type": "system_of_linear_equations",
        "variables": { 
            "n": "number of notebooks", 
            "p": "number of pens" 
        },
        "equations": ["n + p = 120", "50 * n + 20 * p = 3800"],
        "target_variable": "n"
    }
    
    subtask_1 = identifier.identify_subtask(parsed_problem_1)
    print("\n--- Example 1: System of Equations ---")
    print("Parsed Problem Input:")
    print(json.dumps(parsed_problem_1, indent=2))
    print("\nGenerated Subtask Output:")
    print(json.dumps(subtask_1, indent=2))

    # Example 2: A simple calculation
    parsed_problem_2 = {
        "problem_type": "simple_arithmetic",
        "expression_to_evaluate": "25 * 4 + 18 / 3"
    }
    
    subtask_2 = identifier.identify_subtask(parsed_problem_2)
    print("\n--- Example 2: Simple Calculation ---")
    print("Parsed Problem Input:")
    print(json.dumps(parsed_problem_2, indent=2))
    print("\nGenerated Subtask Output:")
    print(json.dumps(subtask_2, indent=2))
    
    # Example 3: Error handling
    parsed_problem_3 = {
        "error": "Failed to parse problem",
        "raw_output": "Some problematic text"
    }
    
    subtask_3 = identifier.identify_subtask(parsed_problem_3)
    print("\n--- Example 3: Error Handling ---")
    print("Parsed Problem Input:")
    print(json.dumps(parsed_problem_3, indent=2))
    print("\nGenerated Subtask Output:")
    print(json.dumps(subtask_3, indent=2))
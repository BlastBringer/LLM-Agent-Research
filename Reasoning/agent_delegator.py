import json

class AgentDelegator:
    """
    Decides whether to delegate a task to the tool-using agent
    or handle it with a simpler, internal method.
    """
    def __init__(self, delegation_rules: dict = None):
        # Simple rule-based logic for delegation
        if delegation_rules is None:
            self.rules = {
                "delegate_by_classification": [
                    "system_of_linear_equations", 
                    "calculus", 
                    "matrix_algebra",
                    "complex_geometry",
                    "differential_equations"
                ],
                "delegate_by_subtask": [
                    "SymbolicSolver", 
                    "WebSearch", 
                    "ComplexCalculation",
                    "ProofAssistant"
                ]
            }
        else:
            self.rules = delegation_rules
        
        print("🚦 Agent Delegator initialized.")

    def should_delegate(self, classification: str, subtask: dict) -> str:
        """
        Makes a decision based on the problem's classification and required subtask.

        Args:
            classification: The output from the Problem Classifier.
            subtask: The output from the Subtask Identifier.

        Returns:
            A string indicating the decision ('delegate_to_agent' or 'solve_internally').
        """
        print(f"\n🚦 Making delegation decision...")
        print(f"   Classification: '{classification}'")
        print(f"   Subtask tool: '{subtask.get('tool_name', 'Unknown')}'")
        
        # Handle None or missing subtask
        if not subtask or not isinstance(subtask, dict):
            print("   Decision: solve_internally (Reason: Invalid subtask)")
            return "solve_internally"
        
        tool_name = subtask.get("tool_name", "")
        
        # Rule 1: Delegate if the classification is known to be complex
        if classification in self.rules["delegate_by_classification"]:
            decision = "delegate_to_agent"
            print(f"   Decision: {decision} (Reason: Complex classification type)")
            return decision

        # Rule 2: Delegate if the required tool is complex
        if tool_name in self.rules["delegate_by_subtask"]:
            decision = "delegate_to_agent"
            print(f"   Decision: {decision} (Reason: Requires specialized tool '{tool_name}')")
            return decision

        # Default Decision: Handle internally if no delegation rule was met
        decision = "solve_internally"
        print(f"   Decision: {decision} (Reason: Simple problem can be handled internally)")
        return decision
    
    def update_rules(self, new_rules: dict):
        """
        Update delegation rules.
        
        Args:
            new_rules: New delegation rules dictionary.
        """
        self.rules.update(new_rules)
        print("🚦 Delegation rules updated.")
    
    def get_rules(self) -> dict:
        """
        Get current delegation rules.
        
        Returns:
            Current delegation rules dictionary.
        """
        return self.rules.copy()

# --- Example Usage ---
if __name__ == "__main__":
    delegator = AgentDelegator()

    # Case 1: Complex problem requiring the solver tool
    classification_1 = "system_of_linear_equations"
    subtask_1 = {
        "tool_name": "SymbolicSolver",
        "tool_input": {
            "equations": ["x+y=10", "x-y=2"], 
            "variables_to_solve": ["x", "y"]
        },
        "human_readable_goal": "Solve system of equations for x and y"
    }
    decision_1 = delegator.should_delegate(classification_1, subtask_1)
    print(f"Final decision: {decision_1}")
    
    # Case 2: A simple arithmetic problem
    classification_2 = "simple_arithmetic"
    subtask_2 = {
        "tool_name": "Calculator",
        "tool_input": {
            "expression": "100 / 5"
        },
        "human_readable_goal": "Calculate simple arithmetic"
    }
    decision_2 = delegator.should_delegate(classification_2, subtask_2)
    print(f"Final decision: {decision_2}")
    
    # Case 3: A general query
    classification_3 = "logic_puzzle"
    subtask_3 = {
        "tool_name": "GeneralQuery",
        "tool_input": {
            "text": "If all cats are mammals, and Felix is a cat, is Felix a mammal?"
        },
        "human_readable_goal": "Solve logic puzzle"
    }
    decision_3 = delegator.should_delegate(classification_3, subtask_3)
    print(f"Final decision: {decision_3}")
    
    # Case 4: Error handling - invalid subtask
    classification_4 = "other"
    subtask_4 = None
    decision_4 = delegator.should_delegate(classification_4, subtask_4)
    print(f"Final decision: {decision_4}")
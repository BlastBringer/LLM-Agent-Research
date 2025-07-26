import json
from datetime import datetime
from typing import List, Dict, Any

class ContextualMemoryTracker:
    """
    Records a sequential log of actions, thoughts, and observations
    during the reasoning process.
    """
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.step_counter = 0
        print("📝 Contextual Memory Tracker initialized.")

    def add_step(self, thought: str, action_taken: str, observation: str, metadata: Dict[str, Any] = None):
        """
        Adds a single step to the reasoning history.

        Args:
            thought: The reasoning or plan of the engine at this step.
            action_taken: The specific action performed (e.g., calling a tool).
            observation: The result or output from the action.
            metadata: Optional additional information about this step.
        """
        self.step_counter += 1
        
        step = {
            "step_number": self.step_counter,
            "thought": thought,
            "action": action_taken,
            "observation": observation,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.history.append(step)
        print(f"📝 Recorded step {self.step_counter}: {action_taken[:50]}...")

    def get_full_history(self) -> List[Dict[str, Any]]:
        """
        Returns the entire history log.
        
        Returns:
            List of all recorded steps.
        """
        return self.history.copy()

    def get_latest_step(self) -> Dict[str, Any]:
        """
        Returns the most recent step.
        
        Returns:
            The latest step dictionary, or None if no steps recorded.
        """
        return self.history[-1] if self.history else None

    def get_steps_by_action(self, action_pattern: str) -> List[Dict[str, Any]]:
        """
        Returns all steps that match a specific action pattern.
        
        Args:
            action_pattern: Pattern to match in action names.
            
        Returns:
            List of matching steps.
        """
        return [step for step in self.history if action_pattern.lower() in step["action"].lower()]

    def clear_memory(self):
        """Resets the history for a new problem."""
        self.history = []
        self.step_counter = 0
        print("📝 Memory cleared for new problem.")

    def export_history(self, filename: str = None) -> str:
        """
        Export history to JSON file.
        
        Args:
            filename: Optional filename for export.
            
        Returns:
            The filename used for export.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reasoning_history_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            print(f"📁 History exported to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to export history: {e}")
            return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the reasoning process.
        
        Returns:
            Summary statistics about the reasoning process.
        """
        if not self.history:
            return {"total_steps": 0, "actions": [], "duration": 0}
        
        actions = [step["action"] for step in self.history]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate duration if timestamps are available
        duration = 0
        if len(self.history) > 1:
            try:
                start_time = datetime.fromisoformat(self.history[0]["timestamp"])
                end_time = datetime.fromisoformat(self.history[-1]["timestamp"])
                duration = (end_time - start_time).total_seconds()
            except:
                duration = 0
        
        return {
            "total_steps": len(self.history),
            "unique_actions": len(action_counts),
            "action_counts": action_counts,
            "duration_seconds": duration,
            "start_time": self.history[0]["timestamp"] if self.history else None,
            "end_time": self.history[-1]["timestamp"] if self.history else None
        }

    def print_history(self):
        """Print the full history in a readable format."""
        if not self.history:
            print("📝 No history recorded yet.")
            return
        
        print(f"\n📝 Reasoning History ({len(self.history)} steps)")
        print("=" * 60)
        
        for step in self.history:
            print(f"Step {step['step_number']}:")
            print(f"  🧠 Thought: {step['thought']}")
            print(f"  ⚡ Action: {step['action']}")
            print(f"  👀 Observation: {step['observation']}")
            if step.get('metadata'):
                print(f"  📊 Metadata: {step['metadata']}")
            print()

# --- Example Usage ---
if __name__ == "__main__":
    memory = ContextualMemoryTracker()

    # Simulate a reasoning process
    memory.add_step(
        thought="I need to understand the problem type.",
        action_taken="ClassifyProblem(problem_text)",
        observation="Classification: 'system_of_linear_equations'",
        metadata={"confidence": 0.95, "method": "llm"}
    )
    
    memory.add_step(
        thought="Now I must extract the equations and variables.",
        action_taken="ParseProblem(problem_text)",
        observation="Parsed data: {'equations': ['x+y=12', 'x-y=4'], 'target_variable': 'x'}",
        metadata={"parsing_method": "structured_prompt"}
    )

    memory.add_step(
        thought="I have the equations, I will use the solver tool.",
        action_taken="SymbolicSolver(equations=['x+y=12', 'x-y=4'])",
        observation="Solver result: {x: 8, y: 4}",
        metadata={"solver_type": "symbolic", "variables_solved": ["x", "y"]}
    )
    
    # Demonstrate different methods
    print("\n--- Full History as JSON ---")
    full_log = memory.get_full_history()
    print(json.dumps(full_log, indent=2))
    
    print("\n--- Readable History ---")
    memory.print_history()
    
    print("\n--- Summary ---")
    summary = memory.get_summary()
    print(json.dumps(summary, indent=2))
    
    # Test export functionality
    print("\n--- Export Test ---")
    exported_file = memory.export_history("test_history.json")
    if exported_file:
        print(f"History exported to: {exported_file}")
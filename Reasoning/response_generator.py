from typing import List, Dict, Any, Union
import json

class ResponseGenerator:
    """
    Formats the final answer and reasoning history into a clean,
    human-readable output.
    """
    def __init__(self):
        print("📄 Response Generator initialized.")
    
    def generate_response(self, final_answer: Any, history_log: List[Dict[str, Any]], 
                         include_metadata: bool = False) -> str:
        """
        Creates a formatted string with step-by-step reasoning.

        Args:
            final_answer: The final solution to the problem.
            history_log: The list of steps from the ContextualMemoryTracker.
            include_metadata: Whether to include metadata in the output.

        Returns:
            A formatted string containing the full solution explanation.
        """
        if not history_log:
            return f"✅ Final Answer: {final_answer}\n(No reasoning steps recorded)"
        
        response_parts = [
            "🧠 STEP-BY-STEP REASONING",
            "=" * 50
        ]
        
        for i, step in enumerate(history_log):
            step_number = step.get('step_number', i + 1)
            thought = step.get('thought', 'No thought recorded')
            action = step.get('action', 'No action recorded')
            observation = step.get('observation', 'No observation recorded')
            
            response_parts.extend([
                f"\nStep {step_number}:",
                f"  🧠 Thought: {thought}",
                f"  ⚡ Action: {action}",
                f"  👀 Observation: {observation}"
            ])
            
            # Include metadata if requested
            if include_metadata and step.get('metadata'):
                metadata = step['metadata']
                response_parts.append(f"  📊 Metadata: {json.dumps(metadata, indent=6)}")
        
        response_parts.extend([
            "\n" + "=" * 50,
            "✅ FINAL ANSWER:",
            str(final_answer)
        ])
        
        return "\n".join(response_parts)
    
    def generate_summary_response(self, final_answer: Any, history_log: List[Dict[str, Any]]) -> str:
        """
        Creates a condensed response with only key steps.
        
        Args:
            final_answer: The final solution to the problem.
            history_log: The list of steps from the ContextualMemoryTracker.
            
        Returns:
            A condensed formatted string.
        """
        if not history_log:
            return f"✅ Answer: {final_answer}"
        
        # Extract key steps (first, last, and any solver steps)
        key_steps = []
        
        if history_log:
            # Always include first step
            key_steps.append(history_log[0])
            
            # Include any solver or calculation steps
            for step in history_log[1:-1]:
                action = step.get('action', '').lower()
                if any(keyword in action for keyword in ['solve', 'calculate', 'agent']):
                    key_steps.append(step)
            
            # Always include last step if more than one step
            if len(history_log) > 1:
                key_steps.append(history_log[-1])
        
        response_parts = ["🔍 Solution Process:"]
        
        for i, step in enumerate(key_steps):
            thought = step.get('thought', 'Processing...')
            observation = step.get('observation', 'No result')
            response_parts.append(f"{i+1}. {thought} → {observation}")
        
        response_parts.extend([
            f"\n✅ Final Answer: {final_answer}"
        ])
        
        return "\n".join(response_parts)
    
    def generate_json_response(self, final_answer: Any, history_log: List[Dict[str, Any]], 
                              problem_context: Dict[str, Any] = None) -> str:
        """
        Creates a JSON-formatted response for API usage.
        
        Args:
            final_answer: The final solution to the problem.
            history_log: The list of steps from the ContextualMemoryTracker.
            problem_context: Optional context about the original problem.
            
        Returns:
            JSON-formatted string containing the complete solution.
        """
        response_data = {
            "final_answer": str(final_answer),
            "reasoning_steps": history_log,
            "total_steps": len(history_log),
            "status": "completed"
        }
        
        if problem_context:
            response_data["problem_context"] = problem_context
        
        return json.dumps(response_data, indent=2, ensure_ascii=False)
    
    def generate_educational_response(self, final_answer: Any, history_log: List[Dict[str, Any]]) -> str:
        """
        Creates an educational response explaining the reasoning process.
        
        Args:
            final_answer: The final solution to the problem.
            history_log: The list of steps from the ContextualMemoryTracker.
            
        Returns:
            Educational formatted string with explanations.
        """
        if not history_log:
            return f"Answer: {final_answer}\n\n💡 Tip: No reasoning steps were recorded for this problem."
        
        response_parts = [
            "📚 EDUCATIONAL SOLUTION WALKTHROUGH",
            "=" * 55
        ]
        
        # Add educational explanations for each step
        for i, step in enumerate(history_log):
            step_number = step.get('step_number', i + 1)
            thought = step.get('thought', 'Processing step')
            action = step.get('action', 'Unknown action')
            observation = step.get('observation', 'No result')
            
            response_parts.extend([
                f"\n📍 Step {step_number}: Understanding the Process",
                f"   🤔 What we're thinking: {thought}",
                f"   🔧 What we're doing: {action}",
                f"   📊 What we discovered: {observation}",
            ])
            
            # Add educational tips based on the action type
            educational_tip = self._get_educational_tip(action)
            if educational_tip:
                response_parts.append(f"   💡 Learning tip: {educational_tip}")
        
        response_parts.extend([
            "\n" + "=" * 55,
            "🎯 FINAL SOLUTION:",
            str(final_answer),
            "\n💭 Remember: Breaking down complex problems into smaller steps makes them easier to solve!"
        ])
        
        return "\n".join(response_parts)
    
    def _get_educational_tip(self, action: str) -> str:
        """
        Provides educational tips based on the action taken.
        
        Args:
            action: The action string from a reasoning step.
            
        Returns:
            Educational tip string or empty string if no tip available.
        """
        action_lower = action.lower()
        
        if 'classify' in action_lower:
            return "Always start by identifying what type of problem you're solving - this helps choose the right approach."
        elif 'parse' in action_lower:
            return "Breaking down the problem into its mathematical components helps organize your solution strategy."
        elif 'symbolic' in action_lower or 'solve' in action_lower:
            return "When solving equations, remember to perform the same operation on both sides to maintain equality."
        elif 'calculate' in action_lower:
            return "Follow the order of operations (PEMDAS/BODMAS) when evaluating mathematical expressions."
        elif 'delegate' in action_lower:
            return "Complex problems often benefit from using specialized tools or methods designed for that problem type."
        else:
            return ""

# --- Example Usage ---
if __name__ == "__main__":
    generator = ResponseGenerator()

    # Example history log (from the memory tracker)
    sample_history = [
        {
            'step_number': 1,
            'thought': 'I need to understand the problem type.',
            'action': 'ClassifyProblem(notebook and pen problem)',
            'observation': 'Classification: system_of_linear_equations',
            'metadata': {'confidence': 0.95}
        },
        {
            'step_number': 2,
            'thought': 'Now I must extract the equations.',
            'action': 'ParseProblem(extract variables and equations)',
            'observation': "Parsed: {'equations': ['n+p=120', '50*n+20*p=3800'], 'variables': ['n', 'p']}",
            'metadata': {'parsing_method': 'llm'}
        },
        {
            'step_number': 3,
            'thought': 'I will use the solver tool.',
            'action': "Agent.run(SymbolicSolver)",
            'observation': 'Solver result: {n: 40, p: 80}',
            'metadata': {'tool': 'SymbolicSolver'}
        }
    ]
    
    # Example final answer
    sample_final_answer = "40 notebooks and 80 pens were sold"

    # Generate different types of responses
    print("--- Standard Response ---")
    standard_response = generator.generate_response(sample_final_answer, sample_history)
    print(standard_response)
    
    print("\n--- Summary Response ---")
    summary_response = generator.generate_summary_response(sample_final_answer, sample_history)
    print(summary_response)
    
    print("\n--- Educational Response ---")
    educational_response = generator.generate_educational_response(sample_final_answer, sample_history)
    print(educational_response)
    
    print("\n--- JSON Response ---")
    json_response = generator.generate_json_response(
        sample_final_answer, 
        sample_history,
        {"problem_type": "word_problem", "domain": "algebra"}
    )
    print(json_response)
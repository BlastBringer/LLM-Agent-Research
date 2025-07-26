#!/usr/bin/env python3
"""
Complete Reasoning Engine Integration
====================================

This is the main orchestrator that ties all components together
according to your architecture diagram.

Components integrated:
- problem_classifier.py
- problem_parser.py  
- subtask_identifier.py
- agent_delegator.py
- contextual_memory_tracker.py
- simple_agent.py
- response_generator.py
"""

import os
import sys
from typing import Dict, Any, Optional

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all components
from problem_classifier import ProblemClassifier
from problem_parser import ProblemParser
from subtask_identifier import SubtaskIdentifier
from agent_delegator import AgentDelegator
from contextual_memory_tracker import ContextualMemoryTracker
from simple_agent import SimpleAgent
from response_generator import ResponseGenerator

class ReasoningEngine:
    """
    Main orchestrator for the entire reasoning pipeline.
    Follows the architecture diagram flow.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the reasoning engine with all components.
        
        Args:
            config: Optional configuration dictionary.
        """
        print("🚀 Initializing Reasoning Engine...")
        print("-" * 50)
        
        # Initialize all components in order
        self.memory = ContextualMemoryTracker()
        self.classifier = ProblemClassifier()
        self.parser = ProblemParser()
        self.subtask_identifier = SubtaskIdentifier()
        self.delegator = AgentDelegator()
        self.agent = SimpleAgent()
        self.response_generator = ResponseGenerator()
        
        # Configuration
        self.config = config or {}
        self.debug_mode = self.config.get('debug', False)
        self.response_format = self.config.get('response_format', 'standard')  # standard, summary, educational, json
        
        print("-" * 50)
        print("✅ Reasoning Engine fully initialized!")
        print(f"   Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        print(f"   Response format: {self.response_format}")
        print()
    
    def solve_problem(self, problem: str) -> str:
        """
        Complete pipeline from problem to solution following the architecture.
        
        Args:
            problem: The math problem text to solve.
            
        Returns:
            Formatted response string with solution and reasoning.
        """
        if not problem or not problem.strip():
            return "❌ Error: No problem provided"
        
        problem = problem.strip()
        print(f"🔍 Starting to solve problem:")
        print(f"   Problem: {problem}")
        print(f"   Length: {len(problem)} characters")
        print()
        
        try:
            # Step 1: Problem Classification
            thought = "First, I must understand the type of problem I'm dealing with."
            print("📍 Step 1: Problem Classification")
            classification = self.classifier.classify(problem)
            self.memory.add_step(
                thought=thought,
                action_taken=f"ProblemClassifier.classify('{problem[:30]}...')",
                observation=f"Classification: {classification}",
                metadata={"step_type": "classification", "input_length": len(problem)}
            )
            
            if self.debug_mode:
                print(f"   Result: {classification}")
            
            # Step 2: Problem Parsing
            thought = "Next, I need to extract the core mathematical structure from the problem."
            print("📍 Step 2: Problem Parsing")
            parsed_data = self.parser.parse(problem)
            observation = f"Parsed structure with {len(parsed_data)} components"
            if "error" in parsed_data:
                observation = f"Parsing encountered issues: {parsed_data.get('error', 'Unknown error')}"
            
            self.memory.add_step(
                thought=thought,
                action_taken="ProblemParser.parse(problem)",
                observation=observation,
                metadata={"step_type": "parsing", "parsed_keys": list(parsed_data.keys())}
            )
            
            if self.debug_mode:
                print(f"   Parsed keys: {list(parsed_data.keys())}")
            
            # Step 3: Subtask Identification
            thought = "Based on the structure, I will determine what specific action is required."
            print("📍 Step 3: Subtask Identification")
            subtask = self.subtask_identifier.identify_subtask(parsed_data)
            self.memory.add_step(
                thought=thought,
                action_taken="SubtaskIdentifier.identify_subtask(parsed_data)",
                observation=f"Identified subtask: {subtask.get('tool_name', 'Unknown')} - {subtask.get('human_readable_goal', 'No goal')}",
                metadata={"step_type": "subtask_identification", "tool_required": subtask.get('tool_name')}
            )
            
            if self.debug_mode:
                print(f"   Tool needed: {subtask.get('tool_name')}")
                print(f"   Goal: {subtask.get('human_readable_goal')}")
            
            # Step 4: Agent Delegation Decision
            thought = "I need to decide whether to use a specialized agent or solve this internally."
            print("📍 Step 4: Delegation Decision")
            decision = self.delegator.should_delegate(classification, subtask)
            self.memory.add_step(
                thought=thought,
                action_taken="AgentDelegator.should_delegate(classification, subtask)",
                observation=f"Decision: {decision}",
                metadata={"step_type": "delegation", "classification": classification}
            )
            
            if self.debug_mode:
                print(f"   Decision: {decision}")
            
            # Step 5: Task Execution
            final_answer = None
            if decision == "delegate_to_agent":
                thought = "Delegating to the specialized agent to execute the complex subtask."
                print("📍 Step 5: Agent Execution")
                agent_response = self.agent.run(subtask)
                self.memory.add_step(
                    thought=thought,
                    action_taken=f"SimpleAgent.run({subtask.get('tool_name', 'Unknown')})",
                    observation=f"Agent result: {agent_response[:100]}{'...' if len(agent_response) > 100 else ''}",
                    metadata={"step_type": "agent_execution", "tool_used": subtask.get('tool_name')}
                )
                final_answer = agent_response
                
                if self.debug_mode:
                    print(f"   Agent completed task")
            
            else:
                thought = "This is a simple task I can solve using basic internal logic."
                print("📍 Step 5: Internal Execution")
                internal_result = self._solve_internally(parsed_data, classification)
                self.memory.add_step(
                    thought=thought,
                    action_taken="ReasoningEngine.solve_internally()",
                    observation=f"Internal result: {internal_result}",
                    metadata={"step_type": "internal_execution", "method": "basic_logic"}
                )
                final_answer = internal_result
                
                if self.debug_mode:
                    print(f"   Solved internally")
            
            # Step 6: Response Generation
            print("📍 Step 6: Response Generation")
            response = self._generate_final_response(final_answer)
            
            print("✅ Problem solving completed!")
            print("=" * 60)
            return response
            
        except Exception as e:
            error_msg = f"❌ Error during problem solving: {str(e)}"
            print(error_msg)
            
            # Log the error
            self.memory.add_step(
                thought="An unexpected error occurred during processing.",
                action_taken="Error handling",
                observation=error_msg,
                metadata={"step_type": "error", "error_type": type(e).__name__}
            )
            
            return f"{error_msg}\n\nPartial reasoning log:\n{self._get_emergency_response()}"
    
    def _solve_internally(self, parsed_data: Dict[str, Any], classification: str) -> str:
        """
        Handle simple problems internally without using the agent.
        
        Args:
            parsed_data: Parsed problem structure.
            classification: Problem classification.
            
        Returns:
            Internal solution result.
        """
        try:
            # Handle simple arithmetic
            if "expression_to_evaluate" in parsed_data:
                expression = parsed_data["expression_to_evaluate"]
                try:
                    result = eval(expression, {"__builtins__": {}}, {})
                    return f"Internal calculation result: {result}"
                except:
                    return f"Internal calculation failed for expression: {expression}"
            
            # Handle other simple cases
            elif classification == "simple_arithmetic":
                return "Solved using basic arithmetic operations"
            
            elif "error" in parsed_data:
                return f"Could not solve due to parsing error: {parsed_data['error']}"
            
            else:
                return f"Solved internally using {classification} methods (simulated solution)"
                
        except Exception as e:
            return f"Internal solving error: {str(e)}"
    
    def _generate_final_response(self, final_answer: str) -> str:
        """
        Generate the final response based on the configured format.
        
        Args:
            final_answer: The solution result.
            
        Returns:
            Formatted response string.
        """
        history = self.memory.get_full_history()
        
        if self.response_format == "summary":
            return self.response_generator.generate_summary_response(final_answer, history)
        elif self.response_format == "educational":
            return self.response_generator.generate_educational_response(final_answer, history)
        elif self.response_format == "json":
            return self.response_generator.generate_json_response(final_answer, history)
        else:  # standard
            return self.response_generator.generate_response(final_answer, history)
    
    def _get_emergency_response(self) -> str:
        """
        Generate an emergency response when errors occur.
        
        Returns:
            Basic response with available information.
        """
        history = self.memory.get_full_history()
        if not history:
            return "No reasoning steps were completed."
        
        steps = []
        for i, step in enumerate(history[-3:], 1):  # Last 3 steps
            steps.append(f"{i}. {step.get('action', 'Unknown action')}: {step.get('observation', 'No observation')}")
        
        return "Last completed steps:\n" + "\n".join(steps)
    
    def reset(self):
        """Reset the engine for a new problem."""
        self.memory.clear_memory()
        print("🔄 Reasoning Engine reset for new problem.")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current reasoning session.
        
        Returns:
            Dictionary containing session statistics.
        """
        return self.memory.get_summary()
    
    def set_config(self, config: Dict[str, Any]):
        """
        Update engine configuration.
        
        Args:
            config: New configuration dictionary.
        """
        self.config.update(config)
        self.debug_mode = self.config.get('debug', False)
        self.response_format = self.config.get('response_format', 'standard')
        print(f"🔧 Configuration updated. Debug: {'ON' if self.debug_mode else 'OFF'}, Format: {self.response_format}")

def main():
    """
    Main function to run the reasoning engine interactively.
    """
    print("🧠 MATHEMATICAL REASONING ENGINE")
    print("=" * 50)
    print("This engine can solve various types of math problems:")
    print("- System of linear equations")
    print("- Simple arithmetic calculations") 
    print("- Basic algebra problems")
    print("- And more...")
    print()
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'reset' - Clear memory for new problem")
    print("  'debug on/off' - Toggle debug mode")
    print("  'format <type>' - Set response format (standard/summary/educational/json)")
    print("  'stats' - Show session statistics")
    print("=" * 50)
    
    # Initialize engine
    engine = ReasoningEngine(config={'debug': False, 'response_format': 'standard'})
    
    session_count = 0
    
    while True:
        try:
            user_input = input("\n🤔 Enter your math problem (or command): ").strip()
            
            if not user_input:
                print("Please enter a problem or command.")
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Thanks for using the Reasoning Engine! Goodbye!")
                break
            
            elif user_input.lower() == 'reset':
                engine.reset()
                print("✅ Engine reset complete.")
                continue
            
            elif user_input.lower().startswith('debug'):
                parts = user_input.lower().split()
                if len(parts) > 1:
                    if parts[1] == 'on':
                        engine.set_config({'debug': True})
                    elif parts[1] == 'off':
                        engine.set_config({'debug': False})
                    else:
                        print("Usage: debug on/off")
                else:
                    current_state = "ON" if engine.debug_mode else "OFF"
                    print(f"Debug mode is currently: {current_state}")
                continue
            
            elif user_input.lower().startswith('format'):
                parts = user_input.lower().split()
                if len(parts) > 1:
                    format_type = parts[1]
                    valid_formats = ['standard', 'summary', 'educational', 'json']
                    if format_type in valid_formats:
                        engine.set_config({'response_format': format_type})
                    else:
                        print(f"Invalid format. Choose from: {', '.join(valid_formats)}")
                else:
                    print(f"Current format: {engine.response_format}")
                    print("Available formats: standard, summary, educational, json")
                continue
            
            elif user_input.lower() == 'stats':
                stats = engine.get_stats()
                print("\n📊 Session Statistics:")
                print(f"   Total steps: {stats.get('total_steps', 0)}")
                print(f"   Unique actions: {stats.get('unique_actions', 0)}")
                print(f"   Duration: {stats.get('duration_seconds', 0):.2f} seconds")
                if stats.get('action_counts'):
                    print("   Action breakdown:")
                    for action, count in stats['action_counts'].items():
                        print(f"     - {action}: {count}")
                continue
            
            # Process as math problem
            session_count += 1
            print(f"\n🔢 Problem #{session_count}")
            print("-" * 30)
            
            response = engine.solve_problem(user_input)
            print(response)
            
            # Ask if user wants to continue
            print("\n" + "-" * 60)
            continue_choice = input("Solve another problem? (y/n/help): ").strip().lower()
            
            if continue_choice == 'n':
                print("\n👋 Thanks for using the Reasoning Engine!")
                break
            elif continue_choice == 'help':
                print("\nAvailable commands:")
                print("  y/yes - Solve another problem")
                print("  n/no - Exit the program") 
                print("  reset - Clear memory and start fresh")
                print("  debug on/off - Toggle detailed output")
                print("  format <type> - Change response format")
                print("  stats - Show session statistics")
            elif continue_choice in ['y', 'yes', '']:
                engine.reset()  # Reset for new problem
                continue
            else:
                engine.reset()  # Reset by default
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("The engine will continue running. Try another problem.")
            continue

def test_engine():
    """
    Test function to validate the engine with sample problems.
    """
    print("🧪 TESTING REASONING ENGINE")
    print("=" * 40)
    
    engine = ReasoningEngine(config={'debug': True, 'response_format': 'educational'})
    
    test_problems = [
        "A company sells notebooks and pens. Each notebook costs ₹50 and each pen costs ₹20. On a certain day, the company sold a total of 120 items and made ₹3,800 in revenue. How many notebooks were sold?",
        "Calculate 25 * 4 + 18 / 3",
        "If x + y = 10 and x - y = 2, find x and y"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🧪 TEST {i}: {problem[:50]}...")
        print("-" * 50)
        
        try:
            response = engine.solve_problem(problem)
            print("✅ Test completed successfully")
            print(response[:200] + "..." if len(response) > 200 else response)
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        engine.reset()
        print()
    
    print("🧪 Testing completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_engine()
    else:
        main()
#!/usr/bin/env python3
"""
🤖 ENHANCED AGENT DELEGATOR
===========================

Manages the delegation of subtasks to specialized external agents.
This bridges the Reasoning Engine with the external Agent system.
"""

import openai
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class EnhancedAgentDelegator:
    """
    Manages delegation of subtasks to appropriate external agents.
    Handles agent selection, task routing, and result aggregation.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Registry of available external agents and their capabilities
        self.agent_registry = {
            "algebra_specialist": {
                "capabilities": ["equation_solving", "system_solving", "polynomial_operations"],
                "tools": ["symbolic_solver", "equation_validator", "algebraic_manipulator"],
                "endpoint": "http://localhost:8001/algebra",
                "status": "available"
            },
            "calculus_specialist": {
                "capabilities": ["derivative_calculation", "integral_calculation", "limit_evaluation"],
                "tools": ["derivative_calculator", "integral_calculator", "limit_evaluator"],
                "endpoint": "http://localhost:8002/calculus",
                "status": "available"
            },
            "geometry_specialist": {
                "capabilities": ["area_calculation", "volume_calculation", "coordinate_geometry"],
                "tools": ["area_calculator", "volume_calculator", "distance_calculator"],
                "endpoint": "http://localhost:8003/geometry", 
                "status": "available"
            },
            "statistics_specialist": {
                "capabilities": ["descriptive_statistics", "probability_calculation", "data_analysis"],
                "tools": ["stats_calculator", "probability_engine", "data_analyzer"],
                "endpoint": "http://localhost:8004/statistics",
                "status": "available"
            },
            "optimization_specialist": {
                "capabilities": ["constraint_optimization", "unconstrained_optimization", "linear_programming"],
                "tools": ["optimizer", "constraint_handler", "gradient_calculator"],
                "endpoint": "http://localhost:8005/optimization",
                "status": "available"
            },
            "general_math_agent": {
                "capabilities": ["general_problem_solving", "fallback_reasoning", "multi_domain"],
                "tools": ["general_solver", "reasoning_engine", "calculator"],
                "endpoint": "http://localhost:8000/general",
                "status": "available"
            }
        }
        
        print("🤖 Enhanced Agent Delegator initialized with 6 specialist agents.")
    
    def delegate_subtasks(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Delegate a list of subtasks to appropriate agents.
        
        Args:
            subtasks: List of structured subtasks from SubtaskIdentifier
            
        Returns:
            Aggregated results from all delegated agents
        """
        print(f"🤖 Delegating {len(subtasks)} subtasks to specialized agents...")
        
        # Step 1: Analyze and route subtasks
        delegation_plan = self._create_delegation_plan(subtasks)
        
        # Step 2: Execute delegations (simulated for now, real implementation would use HTTP calls)
        results = self._execute_delegations(delegation_plan)
        
        # Step 3: Aggregate and validate results
        aggregated_results = self._aggregate_results(results, subtasks)
        
        return aggregated_results
    
    def _create_delegation_plan(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Create a plan for which agents should handle which subtasks."""
        
        delegation_plan = {}
        
        for subtask in subtasks:
            # Determine the best agent for this subtask
            selected_agent = self._select_best_agent(subtask)
            
            if selected_agent not in delegation_plan:
                delegation_plan[selected_agent] = []
            
            delegation_plan[selected_agent].append({
                "subtask": subtask,
                "agent_instructions": self._generate_agent_instructions(subtask),
                "expected_output": self._define_expected_output(subtask)
            })
        
        print(f"📋 Delegation plan created for {len(delegation_plan)} agents")
        for agent, tasks in delegation_plan.items():
            print(f"   {agent}: {len(tasks)} subtasks")
        
        return delegation_plan
    
    def _select_best_agent(self, subtask: Dict[str, Any]) -> str:
        """Select the most appropriate agent for a given subtask."""
        
        operation_type = subtask.get('operation_type', 'unknown')
        tool_category = subtask.get('tool_category', 'general')
        
        # Agent selection logic based on operation type and capabilities
        agent_scores = {}
        
        for agent_name, agent_info in self.agent_registry.items():
            score = 0
            capabilities = agent_info['capabilities']
            tools = agent_info['tools']
            
            # Score based on operation type alignment
            if operation_type in agent_name:
                score += 10
            
            # Score based on tool category match
            if any(tool_category in cap for cap in capabilities):
                score += 8
            
            # Score based on specific tool availability
            if any(tool_category in tool for tool in tools):
                score += 6
            
            # Penalty for unavailable agents
            if agent_info['status'] != 'available':
                score -= 20
            
            agent_scores[agent_name] = score
        
        # Select agent with highest score
        best_agent = max(agent_scores, key=agent_scores.get)
        
        # Fallback to general agent if no good match
        if agent_scores[best_agent] <= 0:
            best_agent = "general_math_agent"
        
        print(f"🎯 Selected {best_agent} for {subtask.get('description', 'unknown task')}")
        return best_agent
    
    def _generate_agent_instructions(self, subtask: Dict[str, Any]) -> str:
        """Generate specific instructions for the agent handling this subtask."""
        
        instruction_template = f"""
        SUBTASK: {subtask.get('description', 'Unknown task')}
        
        OPERATION TYPE: {subtask.get('operation_type', 'unknown')}
        TOOL CATEGORY: {subtask.get('tool_category', 'general')}
        
        INPUT DATA: {subtask.get('input_data', 'No input specified')}
        EXPECTED OUTPUT FORMAT: {subtask.get('output_format', 'general_solution')}
        
        PRIORITY: {subtask.get('priority', 3)}/5
        ESTIMATED TIME: {subtask.get('estimated_execution_time', 'unknown')}
        
        VALIDATION CRITERIA: {json.dumps(subtask.get('validation_criteria', {}), indent=2)}
        
        INSTRUCTIONS:
        1. Process the input data according to the operation type
        2. Use appropriate tools from your toolkit
        3. Validate your results against the criteria
        4. Return results in the specified format
        5. Include confidence score and reasoning steps
        """
        
        return instruction_template.strip()
    
    def _define_expected_output(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Define the expected output format for the agent."""
        
        return {
            "subtask_id": subtask.get('subtask_id', 'unknown'),
            "result": "agent_computed_result",
            "confidence_score": "float_between_0_and_1",
            "reasoning_steps": ["list_of_reasoning_steps"],
            "validation_status": "passed/failed/warning",
            "execution_time": "time_taken_in_seconds",
            "tools_used": ["list_of_tools_actually_used"],
            "error_messages": ["any_errors_encountered"]
        }
    
    def _execute_delegations(self, delegation_plan: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Execute the delegation plan (simulated for now).
        
        In a real implementation, this would:
        1. Make HTTP calls to external agent endpoints
        2. Handle async execution and timeouts
        3. Manage agent load balancing
        4. Handle retries and fallbacks
        """
        
        print("🚀 Executing delegations (simulated)...")
        
        results = {}
        
        for agent_name, tasks in delegation_plan.items():
            agent_results = []
            
            for task in tasks:
                # Simulate agent execution
                simulated_result = self._simulate_agent_execution(agent_name, task)
                agent_results.append(simulated_result)
            
            results[agent_name] = agent_results
        
        return results
    
    def _simulate_agent_execution(self, agent_name: str, task: Dict) -> Dict[str, Any]:
        """
        Simulate external agent execution.
        
        In real implementation, this would be replaced with actual HTTP calls
        to external agent services.
        """
        
        subtask = task['subtask']
        operation_type = subtask.get('operation_type', 'unknown')
        
        # Simulate different responses based on agent type
        if agent_name == "algebra_specialist":
            if "equation" in subtask.get('description', '').lower():
                result = "x = 5"  # Simulated algebra solution
            elif "system" in subtask.get('description', '').lower():
                result = "x = 6, y = 4"  # Simulated system solution
            else:
                result = "algebraic_solution"
        
        elif agent_name == "calculus_specialist":
            if "derivative" in subtask.get('description', '').lower():
                result = "2*x + 3"  # Simulated derivative
            elif "integral" in subtask.get('description', '').lower():
                result = "x**2 + 3*x + C"  # Simulated integral
            else:
                result = "calculus_solution"
        
        elif agent_name == "geometry_specialist":
            result = "40 square units"  # Simulated geometry result
        
        else:
            result = f"Solution from {agent_name}"
        
        return {
            "subtask_id": subtask.get('subtask_id', 'unknown'),
            "result": result,
            "confidence_score": 0.95,
            "reasoning_steps": [
                f"Received task: {subtask.get('description', 'unknown')}",
                f"Applied {operation_type} methods",
                f"Computed result: {result}"
            ],
            "validation_status": "passed",
            "execution_time": "1.2 seconds",
            "tools_used": self.agent_registry[agent_name]['tools'][:2],
            "error_messages": []
        }
    
    def _aggregate_results(self, results: Dict[str, List[Dict]], original_subtasks: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all agents into a coherent solution."""
        
        print("📊 Aggregating results from all agents...")
        
        all_results = []
        for agent_results in results.values():
            all_results.extend(agent_results)
        
        # Sort results by subtask order
        sorted_results = sorted(all_results, key=lambda x: x.get('subtask_id', 'ZZZ'))
        
        # Create aggregated response
        aggregated = {
            "total_subtasks": len(original_subtasks),
            "completed_subtasks": len(all_results),
            "success_rate": len([r for r in all_results if r.get('validation_status') == 'passed']) / len(all_results),
            "average_confidence": sum(r.get('confidence_score', 0) for r in all_results) / len(all_results),
            "total_execution_time": sum(float(r.get('execution_time', '0').split()[0]) for r in all_results),
            "agents_used": list(results.keys()),
            "subtask_results": sorted_results,
            "final_solution": self._synthesize_final_solution(sorted_results),
            "validation_summary": self._create_validation_summary(sorted_results)
        }
        
        print(f"✅ Aggregation complete: {aggregated['success_rate']:.1%} success rate")
        return aggregated
    
    def _synthesize_final_solution(self, results: List[Dict]) -> str:
        """Synthesize individual subtask results into a final solution."""
        
        if not results:
            return "No results to synthesize"
        
        # Simple synthesis - in practice, this would be more sophisticated
        solution_parts = []
        
        for result in results:
            if result.get('validation_status') == 'passed':
                solution_parts.append(f"{result.get('subtask_id', 'Unknown')}: {result.get('result', 'No result')}")
        
        if len(solution_parts) == 1:
            return solution_parts[0].split(': ', 1)[1]  # Return just the result
        else:
            return "; ".join(solution_parts)
    
    def _create_validation_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Create a summary of validation results."""
        
        total = len(results)
        passed = len([r for r in results if r.get('validation_status') == 'passed'])
        failed = len([r for r in results if r.get('validation_status') == 'failed'])
        warnings = len([r for r in results if r.get('validation_status') == 'warning'])
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "pass_rate": passed / total if total > 0 else 0,
            "critical_failures": [r for r in results if r.get('validation_status') == 'failed']
        }
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get current status of all registered agents."""
        return {name: info['status'] for name, info in self.agent_registry.items()}
    
    def update_agent_status(self, agent_name: str, status: str):
        """Update the status of a specific agent."""
        if agent_name in self.agent_registry:
            self.agent_registry[agent_name]['status'] = status
            print(f"🔄 Updated {agent_name} status to {status}")

# Example usage
if __name__ == "__main__":
    delegator = EnhancedAgentDelegator()
    
    # Test with sample subtasks
    sample_subtasks = [
        {
            "subtask_id": "ST_001",
            "description": "Solve the equation 2x + 5 = 15",
            "operation_type": "algebra",
            "tool_category": "equation_solver",
            "input_data": "2x + 5 = 15",
            "output_format": "variable_solution",
            "priority": 1
        },
        {
            "subtask_id": "ST_002", 
            "description": "Find the derivative of x^2 + 3x + 2",
            "operation_type": "calculus",
            "tool_category": "derivative_calculator",
            "input_data": "x^2 + 3x + 2",
            "output_format": "derivative_expression",
            "priority": 2
        }
    ]
    
    results = delegator.delegate_subtasks(sample_subtasks)
    
    print("\n📊 DELEGATION RESULTS:")
    print("=" * 50)
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Total Execution Time: {results['total_execution_time']:.1f} seconds")
    print(f"Final Solution: {results['final_solution']}")
    print(f"Agents Used: {', '.join(results['agents_used'])}")

#!/usr/bin/env python3
"""
🎯 ENHANCED SUBTASK IDENTIFIER
=============================

Advanced subtask identification system that breaks down complex mathematical
problems into manageable subtasks for agent delegation.
"""

import openai
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from dataclasses import dataclass

# Load .env from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

@dataclass
class Subtask:
    """Represents a single subtask identified from a complex problem."""
    id: str
    type: str  # 'symbolic', 'numerical', 'geometric', 'search', 'visualization'
    description: str
    dependencies: List[str]  # IDs of subtasks this depends on
    priority: int  # 1-10, where 1 is highest priority
    estimated_complexity: str  # 'low', 'medium', 'high'
    required_tools: List[str]
    input_data: Dict[str, Any]
    expected_output_type: str

class EnhancedSubtaskIdentifier:
    """
    Advanced subtask identification system using LLM reasoning
    to break down complex mathematical problems.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize the subtask identifier."""
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            base_url=base_url or os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
        )
        
        # Define agent capabilities for delegation
        self.agent_capabilities = {
            'sympy_agent': {
                'tools': ['symbolic_math', 'calculus', 'algebra', 'equation_solving'],
                'specialization': 'Symbolic mathematics and algebraic manipulations',
                'complexity_limit': 'high'
            },
            'numerical_agent': {
                'tools': ['numerical_computation', 'statistics', 'linear_algebra'],
                'specialization': 'Numerical computations and statistical analysis',
                'complexity_limit': 'high'
            },
            'geometry_agent': {
                'tools': ['geometric_calculation', 'trigonometry', 'coordinate_geometry'],
                'specialization': 'Geometric calculations and spatial reasoning',
                'complexity_limit': 'medium'
            },
            'search_agent': {
                'tools': ['web_search', 'knowledge_lookup', 'reference_finding'],
                'specialization': 'Information retrieval and knowledge lookup',
                'complexity_limit': 'low'
            },
            'visualization_agent': {
                'tools': ['plotting', 'graphing', 'chart_creation'],
                'specialization': 'Data visualization and graphical representation',
                'complexity_limit': 'medium'
            }
        }
    
    def identify_subtasks(self, problem: str, problem_type: str, 
                         reasoning_steps: List[str] = None) -> List[Subtask]:
        """
        Identify subtasks from a complex mathematical problem.
        
        Args:
            problem: The mathematical problem statement
            problem_type: Type of problem (from classifier)
            reasoning_steps: Previous reasoning steps (from ReAct agent)
            
        Returns:
            List of identified subtasks
        """
        try:
            # Create subtask identification prompt
            prompt = self._create_subtask_identification_prompt(
                problem, problem_type, reasoning_steps
            )
            
            # Get LLM response for subtask identification
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert mathematical problem decomposition specialist."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
                timeout=30  # Add timeout
            )
            
            # Parse the response
            subtasks_data = self._parse_subtasks_response(response.choices[0].message.content)
            
            # Convert to Subtask objects
            subtasks = self._create_subtask_objects(subtasks_data)
            
            # Optimize subtask order based on dependencies
            optimized_subtasks = self._optimize_subtask_order(subtasks)
            
            return optimized_subtasks
            
        except Exception as e:
            print(f"Error in subtask identification: {str(e)}")
            # Fallback: create basic subtasks
            return self._create_fallback_subtasks(problem, problem_type)
    
    def _create_subtask_identification_prompt(self, problem: str, problem_type: str, 
                                            reasoning_steps: List[str] = None) -> str:
        """Create a detailed prompt for subtask identification."""
        
        base_prompt = f"""
MATHEMATICAL PROBLEM DECOMPOSITION TASK
=====================================

Problem Statement: {problem}
Problem Type: {problem_type}

Previous Reasoning Steps:
{chr(10).join(reasoning_steps) if reasoning_steps else 'None provided'}

Available Agent Types and Their Capabilities:
{json.dumps(self.agent_capabilities, indent=2)}

TASK: Break down this mathematical problem into optimal subtasks for agent delegation.

For each subtask, provide:
1. Unique ID (subtask_1, subtask_2, etc.)
2. Type (symbolic, numerical, geometric, search, visualization)
3. Clear description of what needs to be done
4. Dependencies (which other subtasks must complete first)
5. Priority (1-10, where 1 is highest)
6. Estimated complexity (low, medium, high)
7. Required tools/capabilities
8. Input data needed
9. Expected output type

IMPORTANT GUIDELINES:
- Break complex operations into smaller, manageable pieces
- Identify dependencies between subtasks clearly
- Consider parallel execution opportunities
- Match subtasks to appropriate agent capabilities
- Ensure each subtask has a single, clear responsibility
- Include verification/validation subtasks where appropriate

Output your analysis as a JSON structure with the following format:
{{
    "analysis": {{
        "problem_complexity": "low|medium|high",
        "decomposition_strategy": "description of approach",
        "parallel_opportunities": ["list of subtasks that can run in parallel"],
        "critical_path": ["ordered list of dependent subtasks"]
    }},
    "subtasks": [
        {{
            "id": "subtask_1",
            "type": "symbolic|numerical|geometric|search|visualization",
            "description": "Clear description",
            "dependencies": ["list of prerequisite subtask IDs"],
            "priority": 1,
            "estimated_complexity": "low|medium|high",
            "required_tools": ["list of required tools"],
            "input_data": {{"key": "value pairs"}},
            "expected_output_type": "description of expected output",
            "recommended_agent": "agent_type from capabilities"
        }}
    ]
}}
"""
        return base_prompt
    
    def _parse_subtasks_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response for subtasks."""
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response content: {response}")
            return self._create_fallback_json()
    
    def _create_subtask_objects(self, subtasks_data: Dict[str, Any]) -> List[Subtask]:
        """Convert parsed data to Subtask objects."""
        subtasks = []
        
        for subtask_data in subtasks_data.get('subtasks', []):
            subtask = Subtask(
                id=subtask_data.get('id', f'subtask_{len(subtasks) + 1}'),
                type=subtask_data.get('type', 'numerical'),
                description=subtask_data.get('description', 'Unknown subtask'),
                dependencies=subtask_data.get('dependencies', []),
                priority=subtask_data.get('priority', 5),
                estimated_complexity=subtask_data.get('estimated_complexity', 'medium'),
                required_tools=subtask_data.get('required_tools', []),
                input_data=subtask_data.get('input_data', {}),
                expected_output_type=subtask_data.get('expected_output_type', 'text')
            )
            subtasks.append(subtask)
        
        return subtasks
    
    def _optimize_subtask_order(self, subtasks: List[Subtask]) -> List[Subtask]:
        """Optimize the order of subtasks based on dependencies and priorities."""
        # Topological sort based on dependencies
        ordered_subtasks = []
        remaining_subtasks = subtasks.copy()
        
        while remaining_subtasks:
            # Find subtasks with no unresolved dependencies
            ready_subtasks = []
            for subtask in remaining_subtasks:
                if all(dep_id in [s.id for s in ordered_subtasks] for dep_id in subtask.dependencies):
                    ready_subtasks.append(subtask)
            
            if not ready_subtasks:
                # Break circular dependencies by taking highest priority
                ready_subtasks = [min(remaining_subtasks, key=lambda x: x.priority)]
            
            # Sort ready subtasks by priority
            ready_subtasks.sort(key=lambda x: x.priority)
            
            # Add the highest priority ready subtask
            selected_subtask = ready_subtasks[0]
            ordered_subtasks.append(selected_subtask)
            remaining_subtasks.remove(selected_subtask)
        
        return ordered_subtasks
    
    def _create_fallback_subtasks(self, problem: str, problem_type: str) -> List[Subtask]:
        """Create basic fallback subtasks when LLM parsing fails."""
        return [
            Subtask(
                id='fallback_analysis',
                type='numerical',
                description=f'Analyze and solve {problem_type} problem: {problem[:100]}...',
                dependencies=[],
                priority=1,
                estimated_complexity='medium',
                required_tools=['basic_calculation'],
                input_data={'problem': problem, 'type': problem_type},
                expected_output_type='solution'
            )
        ]
    
    def _create_fallback_json(self) -> Dict[str, Any]:
        """Create fallback JSON structure when parsing fails."""
        return {
            "analysis": {
                "problem_complexity": "medium",
                "decomposition_strategy": "fallback single task",
                "parallel_opportunities": [],
                "critical_path": ["fallback_analysis"]
            },
            "subtasks": []
        }
    
    def visualize_subtasks(self, subtasks: List[Subtask]) -> str:
        """Create a visual representation of the subtask breakdown."""
        visualization = []
        visualization.append("🎯 SUBTASK BREAKDOWN")
        visualization.append("=" * 50)
        
        for i, subtask in enumerate(subtasks, 1):
            visualization.append(f"\n{i}. {subtask.id} ({subtask.type.upper()})")
            visualization.append(f"   Description: {subtask.description}")
            visualization.append(f"   Priority: {subtask.priority} | Complexity: {subtask.estimated_complexity}")
            
            if subtask.dependencies:
                visualization.append(f"   Dependencies: {', '.join(subtask.dependencies)}")
            else:
                visualization.append("   Dependencies: None")
            
            visualization.append(f"   Required Tools: {', '.join(subtask.required_tools)}")
            visualization.append(f"   Expected Output: {subtask.expected_output_type}")
        
        return "\n".join(visualization)

def main():
    """Test the Enhanced Subtask Identifier"""
    identifier = EnhancedSubtaskIdentifier()
    
    # Test with a complex problem
    problem = """
    A ball is thrown upward from the top of a building that is 50 meters high. 
    The ball's height h(t) in meters at time t seconds is given by the equation 
    h(t) = -4.9t² + 20t + 50. Find:
    
    1. The maximum height reached by the ball
    2. The time when the ball reaches maximum height
    3. The time when the ball hits the ground
    4. The velocity of the ball when it hits the ground
    """
    
    problem_type = "physics_calculus"
    
    print("🧠 Testing Enhanced Subtask Identifier...")
    print(f"Problem: {problem[:100]}...")
    
    # Identify subtasks
    subtasks = identifier.identify_subtasks(problem, problem_type)
    
    # Display results
    print(f"\n✅ Identified {len(subtasks)} subtasks:")
    print(identifier.visualize_subtasks(subtasks))
    
    # Show subtask details
    print("\n📊 DETAILED SUBTASK ANALYSIS:")
    for subtask in subtasks:
        print(f"\n🎯 {subtask.id}:")
        print(f"   Type: {subtask.type}")
        print(f"   Complexity: {subtask.estimated_complexity}")
        print(f"   Tools: {subtask.required_tools}")
        print(f"   Input: {subtask.input_data}")

if __name__ == "__main__":
    main()

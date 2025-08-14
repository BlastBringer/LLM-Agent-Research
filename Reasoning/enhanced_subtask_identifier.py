#!/usr/bin/env python3
"""
🧩 ENHANCED SUBTASK IDENTIFIER
=============================

Breaks down complex mathematical problems into manageable subtasks
for delegation to specialized agents.
"""

import openai
import os
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class EnhancedSubtaskIdentifier:
    """
    Analyzes parsed problems and creates structured subtasks for agent delegation.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Define subtask templates for different problem types
        self.subtask_templates = {
            "equation_solving": {
                "tool_category": "algebra_solver",
                "required_tools": ["symbolic_solver", "equation_validator"],
                "complexity": "medium"
            },
            "derivative_calculation": {
                "tool_category": "calculus_engine",
                "required_tools": ["derivative_calculator", "expression_simplifier"],
                "complexity": "medium"
            },
            "integral_calculation": {
                "tool_category": "calculus_engine", 
                "required_tools": ["integral_calculator", "limit_evaluator"],
                "complexity": "high"
            },
            "geometry_calculation": {
                "tool_category": "geometry_engine",
                "required_tools": ["area_calculator", "volume_calculator", "distance_calculator"],
                "complexity": "low"
            },
            "system_solving": {
                "tool_category": "advanced_algebra",
                "required_tools": ["system_solver", "matrix_calculator", "linear_algebra_tools"],
                "complexity": "high"
            },
            "optimization": {
                "tool_category": "optimization_engine",
                "required_tools": ["derivative_calculator", "critical_point_finder", "constraint_handler"],
                "complexity": "very_high"
            }
        }
        
        print("🧩 Enhanced Subtask Identifier initialized.")
    
    def identify_subtasks(self, parsed_data: Dict[str, Any], classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break down a problem into manageable subtasks based on parsing and classification.
        
        Args:
            parsed_data: Output from enhanced problem parser
            classification: Output from enhanced problem classifier
            
        Returns:
            List of structured subtasks for agent delegation
        """
        print(f"🧩 Identifying subtasks for {classification.get('primary_category', 'unknown')} problem...")
        
        # Step 1: Determine main problem type and complexity
        problem_type = classification.get('primary_category', 'unknown')
        subcategory = classification.get('subcategory', 'general')
        difficulty = classification.get('difficulty_level', 'medium')
        
        # Step 2: Use LLM to intelligently break down the problem
        subtasks = self._generate_intelligent_subtasks(parsed_data, classification)
        
        # Step 3: Add metadata and prioritization
        enriched_subtasks = []
        for i, subtask in enumerate(subtasks):
            enriched_subtask = {
                **subtask,
                "subtask_id": f"ST_{i+1:03d}",
                "priority": subtask.get("priority", self._calculate_priority(subtask, i)),
                "dependencies": subtask.get("dependencies", []),
                "estimated_execution_time": self._estimate_execution_time(subtask),
                "agent_requirements": self._determine_agent_requirements(subtask),
                "validation_criteria": self._define_validation_criteria(subtask)
            }
            enriched_subtasks.append(enriched_subtask)
        
        print(f"✅ Generated {len(enriched_subtasks)} subtasks")
        return enriched_subtasks
    
    def _generate_intelligent_subtasks(self, parsed_data: Dict, classification: Dict) -> List[Dict[str, Any]]:
        """Use LLM to intelligently break down complex problems."""
        
        prompt = f"""
        You are an expert mathematical problem decomposer. Break down this problem into logical subtasks.
        
        PARSED PROBLEM DATA:
        {json.dumps(parsed_data, indent=2)}
        
        CLASSIFICATION:
        Category: {classification.get('primary_category', 'unknown')}
        Subcategory: {classification.get('subcategory', 'general')}
        Difficulty: {classification.get('difficulty_level', 'medium')}
        
        TASK: Create a list of subtasks that need to be completed to solve this problem.
        
        Each subtask should include:
        1. A clear description of what needs to be done
        2. The type of mathematical operation required
        3. Required input data
        4. Expected output format
        5. Any dependencies on other subtasks
        
        Return ONLY a JSON array of subtasks:
        [
          {{
            "description": "Clear description of the subtask",
            "operation_type": "algebra|calculus|geometry|statistics|logic",
            "tool_category": "specific tool category needed",
            "input_data": "what data this subtask needs",
            "output_format": "what format the result should be in",
            "priority": 1-5,
            "dependencies": ["list of subtask IDs this depends on"]
          }}
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return self._fallback_subtask_generation(parsed_data, classification)
                
        except Exception as e:
            print(f"❌ Intelligent subtask generation failed: {e}")
            return self._fallback_subtask_generation(parsed_data, classification)
    
    def _fallback_subtask_generation(self, parsed_data: Dict, classification: Dict) -> List[Dict[str, Any]]:
        """Fallback rule-based subtask generation."""
        problem_type = classification.get('primary_category', 'unknown')
        
        if problem_type == 'algebra':
            if 'system' in classification.get('subcategory', ''):
                return [
                    {
                        "description": "Parse and validate the system of equations",
                        "operation_type": "algebra",
                        "tool_category": "system_solver",
                        "input_data": parsed_data.get('equations', []),
                        "output_format": "validated_equation_system",
                        "priority": 1
                    },
                    {
                        "description": "Solve the system using appropriate method",
                        "operation_type": "algebra", 
                        "tool_category": "system_solver",
                        "input_data": "validated_equation_system",
                        "output_format": "variable_solutions",
                        "priority": 2
                    }
                ]
            else:
                return [
                    {
                        "description": "Solve the algebraic equation",
                        "operation_type": "algebra",
                        "tool_category": "equation_solver",
                        "input_data": parsed_data.get('equation', ''),
                        "output_format": "solution_set",
                        "priority": 1
                    }
                ]
        
        elif problem_type == 'calculus':
            if 'derivative' in classification.get('subcategory', ''):
                return [
                    {
                        "description": "Calculate the derivative of the function",
                        "operation_type": "calculus",
                        "tool_category": "derivative_calculator",
                        "input_data": parsed_data.get('function', ''),
                        "output_format": "derivative_expression",
                        "priority": 1
                    },
                    {
                        "description": "Simplify the derivative expression",
                        "operation_type": "algebra",
                        "tool_category": "expression_simplifier", 
                        "input_data": "derivative_expression",
                        "output_format": "simplified_expression",
                        "priority": 2
                    }
                ]
        
        # Default single subtask
        return [
            {
                "description": f"Solve the {problem_type} problem",
                "operation_type": problem_type,
                "tool_category": "general_solver",
                "input_data": str(parsed_data),
                "output_format": "solution",
                "priority": 1
            }
        ]
    
    def _calculate_priority(self, subtask: Dict, index: int) -> int:
        """Calculate priority based on subtask characteristics."""
        base_priority = subtask.get('priority', 3)
        
        # Higher priority for foundational operations
        if 'parse' in subtask.get('description', '').lower():
            return 1
        elif 'validate' in subtask.get('description', '').lower():
            return 2
        elif 'solve' in subtask.get('description', '').lower():
            return max(3, base_priority)
        else:
            return max(4, base_priority)
    
    def _estimate_execution_time(self, subtask: Dict) -> str:
        """Estimate execution time for a subtask."""
        operation = subtask.get('operation_type', 'unknown')
        
        time_estimates = {
            'arithmetic': 'fast',  # < 1 second
            'algebra': 'medium',   # 1-5 seconds
            'calculus': 'slow',    # 5-15 seconds
            'geometry': 'medium',  # 1-5 seconds
            'statistics': 'medium', # 1-5 seconds
            'optimization': 'very_slow'  # > 15 seconds
        }
        
        return time_estimates.get(operation, 'medium')
    
    def _determine_agent_requirements(self, subtask: Dict) -> Dict[str, Any]:
        """Determine what type of agent is needed for this subtask."""
        operation = subtask.get('operation_type', 'unknown')
        tool_category = subtask.get('tool_category', 'general')
        
        return {
            "agent_type": f"{operation}_specialist",
            "required_capabilities": [tool_category, "mathematical_reasoning"],
            "memory_requirements": "standard",
            "parallel_execution": operation in ['arithmetic', 'algebra'],
            "error_handling": "robust"
        }
    
    def _define_validation_criteria(self, subtask: Dict) -> Dict[str, Any]:
        """Define how to validate the subtask results."""
        operation = subtask.get('operation_type', 'unknown')
        
        criteria = {
            "output_format_check": True,
            "mathematical_validity": True,
            "result_reasonableness": True
        }
        
        if operation == 'algebra':
            criteria["solution_verification"] = True
        elif operation == 'calculus':
            criteria["derivative_check"] = True
        elif operation == 'geometry':
            criteria["dimensional_analysis"] = True
        
        return criteria
    
    def analyze_subtask_dependencies(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze and map dependencies between subtasks."""
        dependency_graph = {}
        
        for subtask in subtasks:
            subtask_id = subtask['subtask_id']
            dependencies = subtask.get('dependencies', [])
            dependency_graph[subtask_id] = dependencies
        
        return dependency_graph
    
    def get_execution_order(self, subtasks: List[Dict[str, Any]]) -> List[str]:
        """Determine optimal execution order based on dependencies and priorities."""
        # Simple priority-based ordering (can be enhanced with topological sort)
        sorted_subtasks = sorted(subtasks, key=lambda x: (x.get('priority', 5), x['subtask_id']))
        return [st['subtask_id'] for st in sorted_subtasks]

# Example usage
if __name__ == "__main__":
    identifier = EnhancedSubtaskIdentifier()
    
    # Test with sample data
    sample_parsed = {
        "problem_type": "system_of_linear_equations",
        "equations": ["x + y = 10", "x - y = 2"],
        "variables": {"x": "first variable", "y": "second variable"}
    }
    
    sample_classification = {
        "primary_category": "algebra",
        "subcategory": "system_of_equations", 
        "difficulty_level": "intermediate"
    }
    
    subtasks = identifier.identify_subtasks(sample_parsed, sample_classification)
    
    print("\n📋 GENERATED SUBTASKS:")
    print("=" * 50)
    for subtask in subtasks:
        print(f"ID: {subtask['subtask_id']}")
        print(f"Description: {subtask['description']}")
        print(f"Priority: {subtask['priority']}")
        print(f"Tool Category: {subtask['tool_category']}")
        print("-" * 30)

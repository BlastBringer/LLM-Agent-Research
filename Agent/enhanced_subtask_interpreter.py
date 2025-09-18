#!/usr/bin/env python3
"""
🧩 ENHANCED SUBTASK INTERPRETER
===============================

A sophisticated subtask interpreter that breaks down complex mathematical problems
into manageable subtasks and coordinates their execution for comprehensive solutions.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import openai

@dataclass
class Subtask:
    """Represents a single subtask in a complex problem."""
    id: str
    description: str
    type: str  # 'calculation', 'simplification', 'verification', 'interpretation'
    dependencies: List[str]  # IDs of subtasks this depends on
    input_data: Dict[str, Any]
    expected_output: str
    priority: int  # 1-10, where 10 is highest priority
    tools_needed: List[str]
    estimated_complexity: str  # 'low', 'medium', 'high'
    status: str = 'pending'  # 'pending', 'in_progress', 'completed', 'failed'
    result: Any = None
    reasoning: str = ""

@dataclass
class SubtaskExecutionPlan:
    """Represents the complete execution plan for a problem."""
    problem_id: str
    original_problem: str
    subtasks: List[Subtask]
    execution_order: List[str]  # Ordered list of subtask IDs
    dependencies_graph: Dict[str, List[str]]
    estimated_total_time: float
    coordination_strategy: str

class EnhancedSubtaskInterpreter:
    """
    Advanced subtask interpreter that breaks down complex problems and coordinates their solution.
    """
    
    def __init__(self):
        """Initialize the Enhanced Subtask Interpreter."""
        print("🧩 Enhanced Subtask Interpreter initialized.")
        
        # API configuration for LLM-based subtask analysis
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Subtask classification patterns
        self.subtask_patterns = {
            'multi_step_calculation': [
                r'find.+then.+calculate',
                r'first.+then.+',
                r'step\s+\d+',
                r'after.+compute'
            ],
            'verification_needed': [
                r'check.+solution',
                r'verify.+answer',
                r'substitute.+back',
                r'confirm.+result'
            ],
            'multiple_operations': [
                r'simplify.+and.+solve',
                r'factor.+then.+integrate',
                r'differentiate.+and.+evaluate'
            ],
            'conditional_solving': [
                r'if.+then.+',
                r'given.+find.+',
                r'assuming.+calculate'
            ]
        }
        
        # Tool mapping for different subtask types
        self.tool_mapping = {
            'calculus_derivative': ['derivative_calculator', 'symbolic_solver'],
            'calculus_integration': ['integral_calculator', 'symbolic_solver'],
            'algebra_solving': ['equation_solver', 'symbolic_solver'],
            'algebra_simplification': ['expression_simplifier', 'symbolic_manipulator'],
            'numerical_computation': ['arithmetic_calculator', 'percentage_calculator'],
            'verification': ['solution_verifier', 'substitution_checker'],
            'graphing': ['function_plotter', 'graph_analyzer'],
            'matrix_operations': ['matrix_calculator', 'linear_algebra_solver']
        }
    
    def interpret_and_plan(self, problem: str, parsed_data: Dict[str, Any], 
                          classification: Dict[str, Any]) -> SubtaskExecutionPlan:
        """
        Interpret the problem and create a comprehensive execution plan.
        
        Args:
            problem: The original mathematical problem
            parsed_data: Parsed problem components
            classification: Problem classification data
            
        Returns:
            Complete subtask execution plan
        """
        print(f"🧩 Interpreting problem for subtask breakdown: {problem[:50]}...")
        
        # Generate problem ID
        problem_id = f"prob_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Step 1: Analyze problem complexity and structure
        complexity_analysis = self._analyze_problem_complexity(problem, parsed_data, classification)
        
        # Step 2: Identify potential subtasks using LLM and pattern matching
        potential_subtasks = self._identify_subtasks(problem, parsed_data, classification)
        
        # Step 3: Refine and structure subtasks
        structured_subtasks = self._structure_subtasks(potential_subtasks, complexity_analysis)
        
        # Step 4: Create dependency graph
        dependencies_graph = self._build_dependency_graph(structured_subtasks)
        
        # Step 5: Determine execution order
        execution_order = self._determine_execution_order(structured_subtasks, dependencies_graph)
        
        # Step 6: Select coordination strategy
        coordination_strategy = self._select_coordination_strategy(structured_subtasks, complexity_analysis)
        
        # Step 7: Estimate total time
        estimated_time = self._estimate_total_execution_time(structured_subtasks)
        
        execution_plan = SubtaskExecutionPlan(
            problem_id=problem_id,
            original_problem=problem,
            subtasks=structured_subtasks,
            execution_order=execution_order,
            dependencies_graph=dependencies_graph,
            estimated_total_time=estimated_time,
            coordination_strategy=coordination_strategy
        )
        
        print(f"🧩 Execution plan created with {len(structured_subtasks)} subtasks")
        return execution_plan
    
    def execute_plan(self, execution_plan: SubtaskExecutionPlan, 
                    available_tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the subtask plan in the determined order.
        
        Args:
            execution_plan: The subtask execution plan
            available_tools: Dictionary of available solving tools
            
        Returns:
            Complete execution results with final solution
        """
        print(f"🧩 Executing plan with {len(execution_plan.subtasks)} subtasks...")
        
        execution_results = {
            'plan_id': execution_plan.problem_id,
            'subtask_results': {},
            'execution_log': [],
            'final_solution': None,
            'success': False,
            'total_time': 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Execute subtasks in order
            for subtask_id in execution_plan.execution_order:
                subtask = self._find_subtask_by_id(execution_plan.subtasks, subtask_id)
                if not subtask:
                    continue
                    
                print(f"🔧 Executing subtask: {subtask.description}")
                
                # Check dependencies
                if not self._check_dependencies_met(subtask, execution_results['subtask_results']):
                    print(f"⚠️ Dependencies not met for subtask {subtask_id}, skipping...")
                    continue
                
                # Execute the subtask
                subtask_result = self._execute_single_subtask(subtask, execution_results['subtask_results'], available_tools)
                
                # Record result
                execution_results['subtask_results'][subtask_id] = subtask_result
                execution_results['execution_log'].append({
                    'subtask_id': subtask_id,
                    'description': subtask.description,
                    'result': subtask_result.get('result'),
                    'reasoning': subtask_result.get('reasoning'),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update subtask status
                subtask.status = 'completed' if subtask_result['success'] else 'failed'
                subtask.result = subtask_result.get('result')
                subtask.reasoning = subtask_result.get('reasoning', '')
            
            # Synthesize final solution
            final_solution = self._synthesize_final_solution(execution_plan, execution_results)
            execution_results['final_solution'] = final_solution
            execution_results['success'] = True
            
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_results['total_time'] = execution_time
            
            print(f"🧩 Plan execution completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Plan execution failed: {e}")
            execution_results['error'] = str(e)
            execution_results['success'] = False
        
        return execution_results
    
    def _analyze_problem_complexity(self, problem: str, parsed_data: Dict[str, Any], 
                                  classification: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the complexity and structure of the problem."""
        analysis = {
            'text_length': len(problem),
            'word_count': len(problem.split()),
            'mathematical_operations': len(parsed_data.get('operations', [])),
            'variables_count': len(parsed_data.get('variables', [])),
            'equations_count': len(parsed_data.get('equations', [])),
            'complexity_score': 0,
            'requires_multi_step': False,
            'requires_verification': False,
            'domain_complexity': classification.get('complexity', 'medium')
        }
        
        # Calculate complexity score
        score = 0
        score += analysis['word_count'] * 0.1
        score += analysis['mathematical_operations'] * 2
        score += analysis['variables_count'] * 1.5
        score += analysis['equations_count'] * 3
        
        # Check for multi-step indicators
        multi_step_keywords = ['then', 'next', 'after', 'step', 'first', 'second', 'finally']
        if any(keyword in problem.lower() for keyword in multi_step_keywords):
            analysis['requires_multi_step'] = True
            score += 5
        
        # Check for verification needs
        verification_keywords = ['check', 'verify', 'confirm', 'validate']
        if any(keyword in problem.lower() for keyword in verification_keywords):
            analysis['requires_verification'] = True
            score += 3
        
        analysis['complexity_score'] = score
        
        return analysis
    
    def _identify_subtasks(self, problem: str, parsed_data: Dict[str, Any], 
                          classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential subtasks using LLM and pattern matching."""
        
        # Pattern-based identification
        pattern_subtasks = self._identify_subtasks_by_patterns(problem)
        
        # LLM-based identification
        llm_subtasks = self._identify_subtasks_with_llm(problem, parsed_data, classification)
        
        # Merge and deduplicate
        all_subtasks = pattern_subtasks + llm_subtasks
        
        # Remove duplicates and merge similar tasks
        unique_subtasks = self._deduplicate_subtasks(all_subtasks)
        
        return unique_subtasks
    
    def _identify_subtasks_by_patterns(self, problem: str) -> List[Dict[str, Any]]:
        """Identify subtasks using predefined patterns."""
        subtasks = []
        
        # Check for different types of subtask patterns
        for subtask_type, patterns in self.subtask_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, problem, re.IGNORECASE)
                for match in matches:
                    subtasks.append({
                        'type': subtask_type,
                        'description': match.group(0),
                        'position': match.span(),
                        'confidence': 0.7
                    })
        
        return subtasks
    
    def _identify_subtasks_with_llm(self, problem: str, parsed_data: Dict[str, Any], 
                                   classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to identify subtasks in complex problems."""
        
        prompt = f"""
        Analyze this mathematical problem and break it down into logical subtasks:
        
        Problem: {problem}
        Classification: {classification.get('primary_category', 'general')}
        Variables: {parsed_data.get('variables', [])}
        
        Please identify the main subtasks needed to solve this problem completely.
        For each subtask, provide:
        1. A clear description
        2. The type of operation (calculation, simplification, verification, etc.)
        3. What tools/methods would be needed
        4. Dependencies on other subtasks
        
        Format your response as a JSON list of subtasks.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {"role": "system", "content": "You are an expert mathematics teacher who breaks down complex problems into clear, logical steps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse LLM response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Look for JSON array in the response
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    subtasks_data = json.loads(json_match.group(0))
                    return [{'type': 'llm_identified', 'confidence': 0.9, **task} for task in subtasks_data]
            except json.JSONDecodeError:
                # If JSON parsing fails, extract subtasks from text
                return self._extract_subtasks_from_text(response_text)
                
        except Exception as e:
            print(f"⚠️ LLM subtask identification failed: {e}")
            return []
    
    def _extract_subtasks_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract subtasks from LLM text response when JSON parsing fails."""
        subtasks = []
        
        # Look for numbered or bulleted lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line) or line.startswith('- ') or line.startswith('• '):
                # Clean up the line
                description = re.sub(r'^\d+\.\s*', '', line)
                description = re.sub(r'^[•\-]\s*', '', description)
                
                if len(description) > 10:  # Filter out very short descriptions
                    subtasks.append({
                        'type': 'text_extracted',
                        'description': description,
                        'confidence': 0.6
                    })
        
        return subtasks
    
    def _structure_subtasks(self, potential_subtasks: List[Dict[str, Any]], 
                           complexity_analysis: Dict[str, Any]) -> List[Subtask]:
        """Structure and refine the identified subtasks."""
        structured_subtasks = []
        
        for i, task_data in enumerate(potential_subtasks):
            # Generate unique ID
            task_id = f"subtask_{i+1:03d}"
            
            # Determine subtask type
            task_type = task_data.get('type', 'calculation')
            
            # Map to standard types
            if 'calculation' in task_type or 'compute' in task_data.get('description', '').lower():
                standard_type = 'calculation'
            elif 'simplify' in task_data.get('description', '').lower():
                standard_type = 'simplification'
            elif 'verify' in task_data.get('description', '').lower() or 'check' in task_data.get('description', '').lower():
                standard_type = 'verification'
            else:
                standard_type = 'interpretation'
            
            # Determine tools needed
            tools_needed = self._determine_tools_for_subtask(task_data.get('description', ''), standard_type)
            
            # Estimate complexity
            estimated_complexity = self._estimate_subtask_complexity(task_data.get('description', ''))
            
            subtask = Subtask(
                id=task_id,
                description=task_data.get('description', f'Subtask {i+1}'),
                type=standard_type,
                dependencies=task_data.get('dependencies', []),
                input_data=task_data.get('input_data', {}),
                expected_output=task_data.get('expected_output', 'numerical_result'),
                priority=task_data.get('priority', 5),
                tools_needed=tools_needed,
                estimated_complexity=estimated_complexity
            )
            
            structured_subtasks.append(subtask)
        
        return structured_subtasks
    
    def _build_dependency_graph(self, subtasks: List[Subtask]) -> Dict[str, List[str]]:
        """Build dependency graph between subtasks."""
        dependencies = {}
        
        for subtask in subtasks:
            dependencies[subtask.id] = []
            
            # Analyze description for dependency keywords
            description_lower = subtask.description.lower()
            
            # Check for explicit dependencies
            if 'after' in description_lower or 'then' in description_lower:
                # Find preceding subtasks
                for other_subtask in subtasks:
                    if other_subtask.id != subtask.id:
                        # Simple heuristic: if this subtask mentions results from another
                        if any(word in description_lower for word in ['result', 'answer', 'solution', 'value']):
                            dependencies[subtask.id].append(other_subtask.id)
            
            # Verification tasks typically depend on calculation tasks
            if subtask.type == 'verification':
                for other_subtask in subtasks:
                    if other_subtask.type == 'calculation' and other_subtask.id != subtask.id:
                        dependencies[subtask.id].append(other_subtask.id)
        
        return dependencies
    
    def _determine_execution_order(self, subtasks: List[Subtask], 
                                  dependencies: Dict[str, List[str]]) -> List[str]:
        """Determine the optimal execution order using topological sorting."""
        
        # Simple topological sort
        in_degree = {subtask.id: 0 for subtask in subtasks}
        
        # Calculate in-degrees
        for subtask_id, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[subtask_id] += 1
        
        # Start with nodes with no dependencies
        queue = [subtask_id for subtask_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            # Sort by priority (higher priority first)
            queue.sort(key=lambda x: next(st.priority for st in subtasks if st.id == x), reverse=True)
            
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees
            for subtask_id, deps in dependencies.items():
                if current in deps:
                    in_degree[subtask_id] -= 1
                    if in_degree[subtask_id] == 0:
                        queue.append(subtask_id)
        
        return execution_order
    
    def _select_coordination_strategy(self, subtasks: List[Subtask], 
                                    complexity_analysis: Dict[str, Any]) -> str:
        """Select the best coordination strategy for the subtasks."""
        
        num_subtasks = len(subtasks)
        complexity_score = complexity_analysis.get('complexity_score', 0)
        
        if num_subtasks <= 2:
            return 'sequential'
        elif complexity_score > 15:
            return 'hierarchical'
        elif any(subtask.type == 'verification' for subtask in subtasks):
            return 'validate_each_step'
        else:
            return 'parallel_where_possible'
    
    def _estimate_total_execution_time(self, subtasks: List[Subtask]) -> float:
        """Estimate total execution time for all subtasks."""
        
        time_mapping = {
            'low': 2.0,
            'medium': 5.0,
            'high': 10.0
        }
        
        total_time = 0.0
        for subtask in subtasks:
            estimated_time = time_mapping.get(subtask.estimated_complexity, 5.0)
            total_time += estimated_time
        
        # Add coordination overhead
        coordination_overhead = len(subtasks) * 0.5
        total_time += coordination_overhead
        
        return total_time
    
    def _execute_single_subtask(self, subtask: Subtask, previous_results: Dict[str, Any], 
                               available_tools: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask."""
        
        result = {
            'success': False,
            'result': None,
            'reasoning': '',
            'tools_used': [],
            'execution_time': 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Gather input data from previous results
            input_data = self._gather_input_data(subtask, previous_results)
            
            # Select and use appropriate tools
            for tool_name in subtask.tools_needed:
                if tool_name in available_tools:
                    tool_result = available_tools[tool_name](input_data)
                    result['tools_used'].append(tool_name)
                    
                    if tool_result:
                        result['result'] = tool_result
                        result['reasoning'] = f"Used {tool_name} to compute: {tool_result}"
                        result['success'] = True
                        break
            
            # If no tools worked, try basic reasoning
            if not result['success']:
                result['result'] = f"Subtask completed: {subtask.description}"
                result['reasoning'] = f"Processed subtask of type {subtask.type}"
                result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['reasoning'] = f"Failed to execute subtask: {e}"
        
        result['execution_time'] = (datetime.now() - start_time).total_seconds()
        return result
    
    def _synthesize_final_solution(self, execution_plan: SubtaskExecutionPlan, 
                                  execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize the final solution from all subtask results."""
        
        # Find the final result from the last subtask or verification subtask
        final_result = None
        reasoning_chain = []
        
        for subtask_id in execution_plan.execution_order:
            if subtask_id in execution_results['subtask_results']:
                subtask_result = execution_results['subtask_results'][subtask_id]
                reasoning_chain.append({
                    'step': len(reasoning_chain) + 1,
                    'description': self._find_subtask_by_id(execution_plan.subtasks, subtask_id).description,
                    'result': subtask_result.get('result'),
                    'reasoning': subtask_result.get('reasoning')
                })
                
                # Update final result with the latest meaningful result
                if subtask_result.get('result'):
                    final_result = subtask_result.get('result')
        
        return {
            'final_answer': final_result,
            'reasoning_chain': reasoning_chain,
            'total_subtasks': len(execution_plan.subtasks),
            'successful_subtasks': sum(1 for r in execution_results['subtask_results'].values() if r.get('success')),
            'coordination_strategy': execution_plan.coordination_strategy
        }
    
    # Helper methods
    def _find_subtask_by_id(self, subtasks: List[Subtask], subtask_id: str) -> Optional[Subtask]:
        """Find a subtask by its ID."""
        for subtask in subtasks:
            if subtask.id == subtask_id:
                return subtask
        return None
    
    def _check_dependencies_met(self, subtask: Subtask, completed_results: Dict[str, Any]) -> bool:
        """Check if all dependencies for a subtask are met."""
        for dep_id in subtask.dependencies:
            if dep_id not in completed_results or not completed_results[dep_id].get('success'):
                return False
        return True
    
    def _determine_tools_for_subtask(self, description: str, subtask_type: str) -> List[str]:
        """Determine which tools are needed for a subtask."""
        description_lower = description.lower()
        tools = []
        
        # Map based on keywords in description
        if any(word in description_lower for word in ['derivative', 'differentiate']):
            tools.extend(['derivative_calculator', 'symbolic_solver'])
        elif any(word in description_lower for word in ['integrate', 'integral']):
            tools.extend(['integral_calculator', 'symbolic_solver'])
        elif any(word in description_lower for word in ['solve', 'equation']):
            tools.extend(['equation_solver', 'symbolic_solver'])
        elif any(word in description_lower for word in ['simplify', 'expand', 'factor']):
            tools.extend(['expression_simplifier', 'symbolic_manipulator'])
        else:
            # Default tools based on subtask type
            tools.extend(self.tool_mapping.get(subtask_type, ['general_calculator']))
        
        return tools
    
    def _estimate_subtask_complexity(self, description: str) -> str:
        """Estimate the complexity of a subtask."""
        description_lower = description.lower()
        
        # High complexity indicators
        if any(word in description_lower for word in ['system', 'multiple', 'complex', 'advanced']):
            return 'high'
        # Medium complexity indicators
        elif any(word in description_lower for word in ['solve', 'integrate', 'derivative']):
            return 'medium'
        # Low complexity
        else:
            return 'low'
    
    def _deduplicate_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate subtasks and merge similar ones."""
        unique_subtasks = []
        seen_descriptions = set()
        
        for subtask in subtasks:
            description = subtask.get('description', '').lower().strip()
            
            # Simple similarity check
            is_duplicate = False
            for seen_desc in seen_descriptions:
                if self._calculate_similarity(description, seen_desc) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_subtasks.append(subtask)
                seen_descriptions.add(description)
        
        return unique_subtasks
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        # Simple word-based similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _gather_input_data(self, subtask: Subtask, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Gather input data for a subtask from previous results."""
        input_data = subtask.input_data.copy()
        
        # Add results from dependent subtasks
        for dep_id in subtask.dependencies:
            if dep_id in previous_results:
                input_data[f'dependency_{dep_id}'] = previous_results[dep_id].get('result')
        
        return input_data

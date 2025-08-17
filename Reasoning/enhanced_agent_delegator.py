#!/usr/bin/env python3
"""
🚦 ENHANCED AGENT DELEGATION SYSTEM
==================================

Advanced agent delegation system that determines the optimal delegation strategy
for mathematical problem solving, bridging the Reasoning Engine with specialized agents.
"""

import openai
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load .env from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class DelegationStrategy(Enum):
    """Enumeration of available delegation strategies."""
    INTERNAL_REACT = "internal_react"  # Use internal ReAct agent
    EXTERNAL_AGENT = "external_agent"  # Delegate to external agent system
    HYBRID_APPROACH = "hybrid_approach"  # Combine internal and external
    SEQUENTIAL_DELEGATION = "sequential_delegation"  # Chain multiple agents
    PARALLEL_DELEGATION = "parallel_delegation"  # Run multiple agents in parallel

@dataclass
class DelegationDecision:
    """Represents a delegation decision with rationale."""
    strategy: DelegationStrategy
    confidence: float  # 0.0 to 1.0
    reasoning: str
    estimated_time: float  # in seconds
    estimated_accuracy: float  # 0.0 to 1.0
    recommended_agents: List[str]
    fallback_strategy: Optional[DelegationStrategy]
    resource_requirements: Dict[str, Any]

class EnhancedAgentDelegator:
    """
    Advanced agent delegation system that makes intelligent decisions about
    how to handle different types of mathematical problems.
    """
    
    def __init__(self):
        """Initialize the Enhanced Agent Delegator."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        
        # Agent performance profiles
        self.agent_profiles = {
            'internal_react': {
                'strengths': ['integration', 'step_by_step_reasoning', 'tool_coordination'],
                'weaknesses': ['complex_symbolic_math', 'large_computations'],
                'avg_accuracy': 0.85,
                'avg_response_time': 15.0,
                'complexity_limit': 'medium',
                'cost_factor': 0.1
            },
            'sympy_agent': {
                'strengths': ['symbolic_math', 'calculus', 'algebra', 'equation_solving'],
                'weaknesses': ['word_problems', 'numerical_approximations'],
                'avg_accuracy': 0.95,
                'avg_response_time': 5.0,
                'complexity_limit': 'very_high',
                'cost_factor': 0.05
            },
            'numerical_agent': {
                'strengths': ['numerical_computation', 'statistics', 'optimization'],
                'weaknesses': ['symbolic_manipulation', 'proof_generation'],
                'avg_accuracy': 0.90,
                'avg_response_time': 8.0,
                'complexity_limit': 'high',
                'cost_factor': 0.03
            },
            'geometry_agent': {
                'strengths': ['geometric_calculation', 'trigonometry', 'visualization'],
                'weaknesses': ['abstract_algebra', 'calculus'],
                'avg_accuracy': 0.88,
                'avg_response_time': 10.0,
                'complexity_limit': 'high',
                'cost_factor': 0.04
            },
            'search_agent': {
                'strengths': ['information_retrieval', 'formula_lookup', 'context_finding'],
                'weaknesses': ['computation', 'problem_solving'],
                'avg_accuracy': 0.75,
                'avg_response_time': 12.0,
                'complexity_limit': 'low',
                'cost_factor': 0.02
            }
        }
        
        # Delegation criteria thresholds
        self.thresholds = {
            'complexity_for_external': 0.7,  # Above this, prefer external agents
            'accuracy_requirement': 0.9,     # Minimum required accuracy
            'time_constraint': 30.0,         # Maximum acceptable time (seconds)
            'confidence_threshold': 0.8      # Minimum confidence for delegation
        }
        
        print("🚦 Enhanced Agent Delegator initialized.")
    
    def recommend_strategy(self, problem: str, parsed_data: Dict[str, Any], classification: Dict[str, Any]) -> str:
        """
        Recommend a solving strategy based on problem analysis.
        
        Args:
            problem: The mathematical problem text
            parsed_data: Parsed problem data
            classification: Problem classification results
            
        Returns:
            Recommended strategy string
        """
        try:
            # Get problem type from classification
            problem_type = classification.get('primary_category', 'general_mathematics')
            complexity = classification.get('complexity', 'medium')
            confidence = classification.get('confidence', 0.5)
            
            # Strategy mapping based on problem characteristics
            strategy_map = {
                'calculus_differentiation': 'symbolic_calculus',
                'calculus_integration': 'symbolic_calculus',
                'calculus_limits': 'symbolic_calculus',
                'algebra_equations': 'algebraic_solving',
                'algebra_manipulation': 'symbolic_manipulation',
                'algebra_systems': 'system_solving',
                'arithmetic_percentage': 'numerical_computation',
                'linear_algebra': 'matrix_operations',
                'trigonometry': 'trigonometric_solving'
            }
            
            # Get base strategy
            base_strategy = strategy_map.get(problem_type, 'general_problem_solving')
            
            # Adjust based on complexity and confidence
            if complexity in ['high', 'very_high'] and confidence > 0.8:
                # Use enhanced strategies for complex problems
                if problem_type.startswith('calculus'):
                    return 'react_reasoning'
                elif len(parsed_data.get('equations', [])) > 1:
                    return 'coordinated_subtask_solving'
            
            # Check for multi-step problems that need delegation
            if any(keyword in problem.lower() for keyword in ['and', 'then', 'also', 'determine', 'find', 'calculate']):
                word_count = len(problem.split())
                if word_count > 15:
                    return 'coordinated_subtask_solving'
            
            return base_strategy
            
        except Exception as e:
            # Fallback to default strategy
            return 'general_problem_solving'
    
    def make_delegation_decision(self, 
                               problem: str,
                               problem_type: str,
                               classification_result: Dict[str, Any],
                               subtasks: List[Any] = None,
                               constraints: Dict[str, Any] = None) -> DelegationDecision:
        """
        Make an intelligent delegation decision based on problem characteristics.
        
        Args:
            problem: The mathematical problem
            problem_type: Type of problem from classifier
            classification_result: Detailed classification from enhanced classifier
            subtasks: List of identified subtasks
            constraints: Time, accuracy, resource constraints
            
        Returns:
            DelegationDecision with recommended strategy
        """
        
        # Analyze problem characteristics
        problem_analysis = self._analyze_problem_for_delegation(
            problem, problem_type, classification_result
        )
        
        # Get LLM recommendation
        llm_recommendation = self._get_llm_delegation_recommendation(
            problem, problem_analysis, subtasks
        )
        
        # Combine analysis with heuristics
        final_decision = self._synthesize_delegation_decision(
            problem_analysis, llm_recommendation, constraints
        )
        
        return final_decision
    
    def _analyze_problem_for_delegation(self, 
                                      problem: str, 
                                      problem_type: str, 
                                      classification: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze problem characteristics relevant to delegation."""
        
        analysis = {
            'problem_length': len(problem),
            'complexity_score': classification.get('difficulty_level', 'unknown'),
            'tools_needed': classification.get('tools_needed', []),
            'estimated_steps': classification.get('estimated_solution_steps', 1),
            'mathematical_concepts': classification.get('mathematical_concepts', []),
            'has_word_problem': 'word' in problem.lower() or 'if' in problem.lower(),
            'requires_symbolic_math': any(tool in classification.get('tools_needed', []) 
                                        for tool in ['symbolic_calculator', 'equation_solver']),
            'requires_numerical_computation': 'numerical' in problem.lower() or 'calculate' in problem.lower(),
            'requires_visualization': 'graph' in problem.lower() or 'plot' in problem.lower()
        }
        
        # Calculate complexity score
        complexity_factors = [
            len(analysis['tools_needed']) / 5.0,  # Normalize by max expected tools
            analysis['estimated_steps'] / 10.0,  # Normalize by max expected steps
            len(analysis['mathematical_concepts']) / 8.0,  # Normalize by max concepts
            0.5 if analysis['has_word_problem'] else 0.0,
            0.3 if analysis['requires_symbolic_math'] else 0.0
        ]
        
        analysis['calculated_complexity'] = min(sum(complexity_factors), 1.0)
        
        return analysis
    
    def _get_llm_delegation_recommendation(self, 
                                         problem: str, 
                                         analysis: Dict[str, Any], 
                                         subtasks: List[Any]) -> Dict[str, Any]:
        """Get LLM recommendation for delegation strategy."""
        
        prompt = f"""
MATHEMATICAL PROBLEM DELEGATION ANALYSIS
======================================

Problem: {problem}

Problem Analysis:
- Complexity Score: {analysis.get('calculated_complexity', 0.5)}
- Tools Needed: {analysis.get('tools_needed', [])}
- Estimated Steps: {analysis.get('estimated_steps', 1)}
- Mathematical Concepts: {analysis.get('mathematical_concepts', [])}
- Word Problem: {analysis.get('has_word_problem', False)}
- Requires Symbolic Math: {analysis.get('requires_symbolic_math', False)}
- Requires Numerical Computation: {analysis.get('requires_numerical_computation', False)}

Available Agent Profiles:
{json.dumps(self.agent_profiles, indent=2)}

Available Delegation Strategies:
1. INTERNAL_REACT: Use the integrated ReAct agent with embedded tools
2. EXTERNAL_AGENT: Delegate to specialized external agent
3. HYBRID_APPROACH: Combine internal reasoning with external tool calls
4. SEQUENTIAL_DELEGATION: Chain multiple specialized agents
5. PARALLEL_DELEGATION: Run multiple agents simultaneously and aggregate

Subtasks Identified: {len(subtasks) if subtasks else 0}

TASK: Recommend the optimal delegation strategy for this problem.

Consider:
- Problem complexity and agent capabilities
- Expected accuracy and response time
- Resource efficiency
- Risk of failure and need for fallbacks

Respond with JSON:
{{
    "recommended_strategy": "strategy_name",
    "confidence": 0.95,
    "reasoning": "detailed explanation",
    "estimated_accuracy": 0.92,
    "estimated_time": 15.0,
    "recommended_agents": ["agent1", "agent2"],
    "fallback_strategy": "alternative_strategy",
    "risk_factors": ["potential issues"],
    "optimization_suggestions": ["ways to improve performance"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert in mathematical problem solving delegation and agent orchestration."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse LLM response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return self._create_fallback_recommendation()
                
        except Exception as e:
            print(f"❌ LLM delegation error: {e}")
            return self._create_fallback_recommendation()
    
    def _synthesize_delegation_decision(self, 
                                      analysis: Dict[str, Any], 
                                      llm_rec: Dict[str, Any], 
                                      constraints: Dict[str, Any] = None) -> DelegationDecision:
        """Synthesize final delegation decision from analysis and LLM recommendation."""
        
        constraints = constraints or {}
        
        # Get strategy from LLM recommendation
        strategy_name = llm_rec.get('recommended_strategy', 'internal_react')
        try:
            strategy = DelegationStrategy(strategy_name.lower())
        except ValueError:
            strategy = DelegationStrategy.INTERNAL_REACT
        
        # Adjust based on constraints
        if constraints.get('max_time', float('inf')) < llm_rec.get('estimated_time', 15.0):
            strategy = DelegationStrategy.INTERNAL_REACT  # Faster fallback
        
        if constraints.get('min_accuracy', 0.0) > llm_rec.get('estimated_accuracy', 0.85):
            if strategy == DelegationStrategy.INTERNAL_REACT:
                strategy = DelegationStrategy.EXTERNAL_AGENT  # More accurate option
        
        # Calculate final confidence
        base_confidence = llm_rec.get('confidence', 0.7)
        complexity_penalty = analysis.get('calculated_complexity', 0.5) * 0.2
        final_confidence = max(0.1, base_confidence - complexity_penalty)
        
        # Determine fallback strategy
        fallback = None
        if strategy == DelegationStrategy.EXTERNAL_AGENT:
            fallback = DelegationStrategy.INTERNAL_REACT
        elif strategy == DelegationStrategy.INTERNAL_REACT:
            fallback = DelegationStrategy.HYBRID_APPROACH
        
        return DelegationDecision(
            strategy=strategy,
            confidence=final_confidence,
            reasoning=llm_rec.get('reasoning', 'Default reasoning based on analysis'),
            estimated_time=llm_rec.get('estimated_time', 15.0),
            estimated_accuracy=llm_rec.get('estimated_accuracy', 0.85),
            recommended_agents=llm_rec.get('recommended_agents', ['internal_react']),
            fallback_strategy=fallback,
            resource_requirements={
                'memory': 'medium',
                'cpu': 'medium',
                'network': 'required' if strategy == DelegationStrategy.EXTERNAL_AGENT else 'optional'
            }
        )
    
    def _create_fallback_recommendation(self) -> Dict[str, Any]:
        """Create a safe fallback recommendation when LLM fails."""
        return {
            'recommended_strategy': 'internal_react',
            'confidence': 0.6,
            'reasoning': 'Fallback to internal ReAct agent due to analysis failure',
            'estimated_accuracy': 0.80,
            'estimated_time': 20.0,
            'recommended_agents': ['internal_react'],
            'fallback_strategy': 'hybrid_approach',
            'risk_factors': ['analysis_failure'],
            'optimization_suggestions': ['retry_with_simpler_approach']
        }
    
    def execute_delegation_strategy(self, 
                                  decision: DelegationDecision, 
                                  problem: str, 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chosen delegation strategy.
        
        Args:
            decision: The delegation decision
            problem: The mathematical problem
            context: Additional context and data
            
        Returns:
            Execution result with solution and metadata
        """
        
        print(f"🚦 Executing delegation strategy: {decision.strategy.value}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Expected accuracy: {decision.estimated_accuracy:.2f}")
        print(f"   Expected time: {decision.estimated_time:.1f}s")
        
        execution_start = time.time()
        
        try:
            if decision.strategy == DelegationStrategy.INTERNAL_REACT:
                result = self._execute_internal_react(problem, context)
            elif decision.strategy == DelegationStrategy.EXTERNAL_AGENT:
                result = self._execute_external_agent(problem, context, decision.recommended_agents)
            elif decision.strategy == DelegationStrategy.HYBRID_APPROACH:
                result = self._execute_hybrid_approach(problem, context)
            elif decision.strategy == DelegationStrategy.SEQUENTIAL_DELEGATION:
                result = self._execute_sequential_delegation(problem, context, decision.recommended_agents)
            elif decision.strategy == DelegationStrategy.PARALLEL_DELEGATION:
                result = self._execute_parallel_delegation(problem, context, decision.recommended_agents)
            else:
                result = self._execute_internal_react(problem, context)  # Safe fallback
            
            execution_time = time.time() - execution_start
            
            # Add metadata
            result['delegation_metadata'] = {
                'strategy_used': decision.strategy.value,
                'actual_execution_time': execution_time,
                'predicted_time': decision.estimated_time,
                'confidence': decision.confidence,
                'agents_used': decision.recommended_agents
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Delegation execution error: {e}")
            
            # Try fallback strategy
            if decision.fallback_strategy:
                print(f"🔄 Trying fallback strategy: {decision.fallback_strategy.value}")
                fallback_decision = DelegationDecision(
                    strategy=decision.fallback_strategy,
                    confidence=decision.confidence * 0.8,
                    reasoning="Fallback due to primary strategy failure",
                    estimated_time=decision.estimated_time * 1.5,
                    estimated_accuracy=decision.estimated_accuracy * 0.9,
                    recommended_agents=['internal_react'],
                    fallback_strategy=None,
                    resource_requirements=decision.resource_requirements
                )
                return self.execute_delegation_strategy(fallback_decision, problem, context)
            else:
                return {
                    'success': False,
                    'error': str(e),
                    'delegation_metadata': {
                        'strategy_used': decision.strategy.value,
                        'execution_failed': True
                    }
                }
    
    def _execute_internal_react(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using internal ReAct agent."""
        try:
            from enhanced_react_math_agent import EnhancedReActMathAgent
            agent = EnhancedReActMathAgent()
            result = agent.solve_problem(problem)
            
            return {
                'success': True,
                'solution': result.get('solution', 'No solution'),
                'reasoning_steps': result.get('reasoning_steps', []),
                'iterations_used': result.get('iterations_used', 0),
                'method': 'internal_react'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Internal ReAct execution failed: {str(e)}",
                'method': 'internal_react'
            }
    
    def _execute_external_agent(self, problem: str, context: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """Execute using external agent system."""
        # Placeholder for external agent integration
        # This would interface with the Agent system from the diagram
        
        return {
            'success': True,
            'solution': f"External agent solution for: {problem[:50]}...",
            'method': 'external_agent',
            'agents_used': agents,
            'note': 'This is a placeholder - integrate with actual external agent system'
        }
    
    def _execute_hybrid_approach(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using hybrid internal + external approach."""
        # Placeholder for hybrid implementation
        return {
            'success': True,
            'solution': f"Hybrid solution for: {problem[:50]}...",
            'method': 'hybrid_approach',
            'note': 'This is a placeholder - implement hybrid reasoning + external tools'
        }
    
    def _execute_sequential_delegation(self, problem: str, context: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """Execute using sequential agent delegation."""
        # Placeholder for sequential delegation
        return {
            'success': True,
            'solution': f"Sequential delegation solution for: {problem[:50]}...",
            'method': 'sequential_delegation',
            'agents_used': agents,
            'note': 'This is a placeholder - implement agent chaining'
        }
    
    def _execute_parallel_delegation(self, problem: str, context: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """Execute using parallel agent delegation."""
        # Placeholder for parallel delegation
        return {
            'success': True,
            'solution': f"Parallel delegation solution for: {problem[:50]}...",
            'method': 'parallel_delegation',
            'agents_used': agents,
            'note': 'This is a placeholder - implement parallel agent execution'
        }

def main():
    """Test the Enhanced Agent Delegator"""
    delegator = EnhancedAgentDelegator()
    
    # Test problem
    problem = "Find the derivative of f(x) = x³ + 2x² - 5x + 3 and determine its critical points"
    problem_type = "calculus"
    
    # Mock classification result
    classification = {
        'difficulty_level': 'intermediate',
        'tools_needed': ['symbolic_calculator', 'equation_solver', 'derivative_calculator'],
        'estimated_solution_steps': 4,
        'mathematical_concepts': ['derivatives', 'critical_points', 'algebra']
    }
    
    print("🚦 Testing Enhanced Agent Delegator...")
    print(f"Problem: {problem}")
    print(f"Type: {problem_type}")
    
    # Make delegation decision
    decision = delegator.make_delegation_decision(problem, problem_type, classification)
    
    print(f"\n📋 DELEGATION DECISION:")
    print(f"   Strategy: {decision.strategy.value}")
    print(f"   Confidence: {decision.confidence:.2f}")
    print(f"   Reasoning: {decision.reasoning}")
    print(f"   Estimated Time: {decision.estimated_time:.1f}s")
    print(f"   Estimated Accuracy: {decision.estimated_accuracy:.2f}")
    print(f"   Recommended Agents: {decision.recommended_agents}")
    print(f"   Fallback Strategy: {decision.fallback_strategy.value if decision.fallback_strategy else 'None'}")
    
    # Execute the strategy
    print(f"\n🚀 EXECUTING STRATEGY...")
    context = {'problem_type': problem_type, 'classification': classification}
    result = delegator.execute_delegation_strategy(decision, problem, context)
    
    print(f"\n✅ EXECUTION RESULT:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Method: {result.get('method', 'unknown')}")
    if result.get('solution'):
        print(f"   Solution: {result['solution']}")
    if result.get('delegation_metadata'):
        metadata = result['delegation_metadata']
        print(f"   Actual Time: {metadata.get('actual_execution_time', 0):.1f}s")
        print(f"   Strategy Used: {metadata.get('strategy_used', 'unknown')}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🤖 ENHANCED CREWAI INTEGRATION FOR REASONING ENGINE
=================================================

Enhanced CrewAI setup that works seamlessly with our Enhanced Reasoning Engine.
Uses OpenRouter configuration for consistency with reasoning components.
"""

import os
import yaml
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import CrewAI components
try:
    from crewai import Agent, LLM, Crew, Process, Task
    from crewai_tools import SerperDevTool
    CREWAI_AVAILABLE = True
except ImportError:
    print("⚠️ CrewAI not installed. Using mock implementations.")
    CREWAI_AVAILABLE = False

# Import our enhanced tools and new components
try:
    from enhanced_tools import MATHEMATICAL_TOOLS, get_available_tools
    TOOLS_AVAILABLE = True
except ImportError:
    print("⚠️ Enhanced tools not available. Using mock tools.")
    TOOLS_AVAILABLE = False

try:
    from rl_policy_updater import get_policy_manager, update_agent_policy
    RL_POLICY_AVAILABLE = True
except ImportError:
    print("⚠️ RL Policy system not available.")
    RL_POLICY_AVAILABLE = False

try:
    from external_tools_integration import get_external_tools_manager, execute_external_tool
    EXTERNAL_TOOLS_AVAILABLE = True
except ImportError:
    print("⚠️ External tools integration not available.")
    EXTERNAL_TOOLS_AVAILABLE = False

class EnhancedCrewManager:
    """Enhanced CrewAI manager compatible with Reasoning Engine."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the enhanced crew manager."""
        self.config_dir = Path(config_dir)
        self.agents_config = None
        self.tasks_config = None
        self.llm = None
        self.agents = {}
        self.tasks = {}
        self.crew = None
        
        # Load configurations
        self._load_configurations()
        self._setup_llm()
        
        if CREWAI_AVAILABLE:
            self._create_agents()
            self._create_tasks()
            self._create_crew()
        else:
            self._create_mock_crew()
    
    def _load_configurations(self):
        """Load YAML configurations."""
        try:
            agents_file = self.config_dir / "agents.yaml"
            tasks_file = self.config_dir / "tasks.yaml"
            
            if agents_file.exists():
                with open(agents_file, "r") as f:
                    self.agents_config = yaml.safe_load(f)
                print(f"✅ Loaded agents configuration from {agents_file}")
            else:
                print(f"⚠️ Agents config not found at {agents_file}")
                self.agents_config = self._get_default_agents_config()
            
            if tasks_file.exists():
                with open(tasks_file, "r") as f:
                    self.tasks_config = yaml.safe_load(f)
                print(f"✅ Loaded tasks configuration from {tasks_file}")
            else:
                print(f"⚠️ Tasks config not found at {tasks_file}")
                self.tasks_config = self._get_default_tasks_config()
                
        except Exception as e:
            print(f"❌ Error loading configurations: {e}")
            self.agents_config = self._get_default_agents_config()
            self.tasks_config = self._get_default_tasks_config()
    
    def _setup_llm(self):
        """Setup LLM configuration compatible with Reasoning Engine."""
        try:
            # Use OpenRouter configuration for consistency
            openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
            
            if CREWAI_AVAILABLE and openrouter_api_key:
                self.llm = LLM(
                    model="google/gemini-2.0-flash-exp:free",
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key,
                    temperature=0.1,
                    max_tokens=4000
                )
                print("✅ Using OpenRouter LLM configuration")
            else:
                # Fallback configuration
                self.llm = self._create_mock_llm()
                print("⚠️ Using mock LLM configuration")
                
        except Exception as e:
            print(f"❌ Error setting up LLM: {e}")
            self.llm = self._create_mock_llm()
    
    def _create_agents(self):
        """Create specialized agents."""
        if not CREWAI_AVAILABLE or not self.agents_config:
            return
        
        try:
            # Core mathematical agents
            self.agents['interpreter'] = Agent(
                config=self.agents_config.get('interpreter', {}),
                llm=self.llm,
                verbose=True
            )
            
            self.agents['classifier'] = Agent(
                config=self.agents_config.get('type_classifier', {}),
                llm=self.llm,
                verbose=True
            )
            
            self.agents['tool_selector'] = Agent(
                config=self.agents_config.get('tool_selector', {}),
                llm=self.llm,
                verbose=True
            )
            
            self.agents['sympy_executor'] = Agent(
                config=self.agents_config.get('sympy_executor', {}),
                llm=self.llm,
                verbose=True
            )
            
            self.agents['result_aggregator'] = Agent(
                config=self.agents_config.get('result_aggregator', {}),
                llm=self.llm,
                verbose=True
            )
            
            # Additional specialized agents
            if 'numeric_executor' in self.agents_config:
                self.agents['numeric_executor'] = Agent(
                    config=self.agents_config['numeric_executor'],
                    llm=self.llm,
                    verbose=True
                )
            
            if 'verifier' in self.agents_config:
                self.agents['verifier'] = Agent(
                    config=self.agents_config['verifier'],
                    llm=self.llm,
                    verbose=True
                )
            
            print(f"✅ Created {len(self.agents)} specialized agents")
            
        except Exception as e:
            print(f"❌ Error creating agents: {e}")
    
    def _create_tasks(self):
        """Create task workflow."""
        if not CREWAI_AVAILABLE or not self.tasks_config:
            return
        
        try:
            # Core mathematical tasks
            self.tasks['parse_and_classify'] = Task(
                config=self.tasks_config.get('parse_and_classify_task', {}),
                agent=self.agents.get('interpreter'),
                verbose=True
            )
            
            self.tasks['classify_type'] = Task(
                config=self.tasks_config.get('classify_task_type', {}),
                agent=self.agents.get('classifier'),
                verbose=True,
                context=[self.tasks.get('parse_and_classify')] if 'parse_and_classify' in self.tasks else []
            )
            
            self.tasks['select_tools'] = Task(
                config=self.tasks_config.get('select_tool_task', {}),
                agent=self.agents.get('tool_selector'),
                verbose=True,
                context=[self.tasks.get('parse_and_classify')] if 'parse_and_classify' in self.tasks else []
            )
            
            self.tasks['execute_symbolic'] = Task(
                config=self.tasks_config.get('execute_symbolic_task', {}),
                agent=self.agents.get('sympy_executor'),
                verbose=True,
                context=[t for t in [self.tasks.get('parse_and_classify'), 
                                   self.tasks.get('classify_type'),
                                   self.tasks.get('select_tools')] if t]
            )
            
            self.tasks['aggregate_results'] = Task(
                config=self.tasks_config.get('aggregate_results_task', {}),
                agent=self.agents.get('result_aggregator'),
                verbose=True,
                context=[t for t in self.tasks.values() if t]
            )
            
            print(f"✅ Created {len(self.tasks)} specialized tasks")
            
        except Exception as e:
            print(f"❌ Error creating tasks: {e}")
    
    def _create_crew(self):
        """Create the crew with agents and tasks."""
        if not CREWAI_AVAILABLE:
            return
        
        try:
            self.crew = Crew(
                agents=list(self.agents.values()),
                tasks=list(self.tasks.values()),
                process=Process.sequential,
                verbose=True
            )
            print("✅ Enhanced crew created successfully")
            
        except Exception as e:
            print(f"❌ Error creating crew: {e}")
    
    def solve_mathematical_problem(self, problem: str, subtask_type: str = "general") -> Dict[str, Any]:
        """
        Solve a mathematical problem using the crew with RL policy updates.
        
        Args:
            problem: Mathematical problem description
            subtask_type: Type of mathematical subtask
        
        Returns:
            Solution result from the crew
        """
        start_time = time.time()
        
        if self.crew and CREWAI_AVAILABLE:
            try:
                # Execute with crew
                result = self.crew.kickoff(inputs={
                    "input": problem,
                    "subtask_type": subtask_type,
                    "context": f"Solving {subtask_type} mathematical problem"
                })
                
                execution_time = time.time() - start_time
                
                # Prepare result
                crew_result = {
                    "success": True,
                    "result": str(result),
                    "method": "crewai_agents",
                    "subtask_type": subtask_type,
                    "execution_time": execution_time
                }
                
                # Update RL policy if available
                if RL_POLICY_AVAILABLE:
                    try:
                        policy_update = update_agent_policy(
                            "crew_system", subtask_type, crew_result, execution_time
                        )
                        crew_result["policy_update"] = policy_update
                    except Exception as e:
                        logger.warning(f"Policy update failed: {e}")
                
                return crew_result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_result = {
                    "success": False,
                    "result": f"CrewAI error: {str(e)}",
                    "method": "crewai_agents",
                    "subtask_type": subtask_type,
                    "execution_time": execution_time
                }
                
                # Update policy for failed execution
                if RL_POLICY_AVAILABLE:
                    try:
                        update_agent_policy("crew_system", subtask_type, error_result, execution_time)
                    except Exception as pe:
                        logger.warning(f"Policy update for error failed: {pe}")
                
                print(f"❌ CrewAI execution error: {e}")
                return error_result
        else:
            # Mock implementation with external tools integration
            return self._mock_solve_with_external_tools(problem, subtask_type)
    
    def _mock_solve_problem(self, problem: str, subtask_type: str) -> Dict[str, Any]:
        """Mock problem solving when CrewAI is not available."""
        return {
            "success": True,
            "result": f"Mock solution for {subtask_type} problem: {problem[:50]}...",
            "method": "mock_agents",
            "subtask_type": subtask_type,
            "note": "CrewAI not available - using mock implementation"
        }
    
    def _mock_solve_with_external_tools(self, problem: str, subtask_type: str) -> Dict[str, Any]:
        """Mock problem solving with external tools integration."""
        try:
            # Try to use external tools if available
            if EXTERNAL_TOOLS_AVAILABLE:
                # Determine which external tool to use based on subtask type
                if subtask_type in ["algebraic_simplification", "equation_solving", "calculus"]:
                    tool_result = execute_external_tool("sympy", "auto", {
                        'expression': problem
                    })
                    
                    if tool_result.get('success'):
                        return {
                            "success": True,
                            "result": f"External SymPy solution: {tool_result['result']}",
                            "method": "external_sympy",
                            "subtask_type": subtask_type,
                            "external_tool_used": True
                        }
                
                elif subtask_type in ["numerical_computation", "statistics"]:
                    # Use Python sandbox for numerical computations
                    code = f"""
import math
import numpy as np

# Attempt to solve: {problem}
try:
    # Basic numerical evaluation
    result = eval("{problem.replace('^', '**')}")
    print(f"Numerical result: {{result}}")
except:
    print("Could not evaluate numerically")
"""
                    tool_result = execute_external_tool("sandbox", "execute", {
                        'code': code
                    })
                    
                    if tool_result.get('success'):
                        return {
                            "success": True,
                            "result": f"Sandbox computation: {tool_result['result']}",
                            "method": "python_sandbox",
                            "subtask_type": subtask_type,
                            "external_tool_used": True
                        }
                
                elif subtask_type in ["knowledge_lookup", "definitions"]:
                    # Use search engine for knowledge lookup
                    tool_result = execute_external_tool("search", "search", {
                        'query': problem,
                        'source': 'wikipedia'
                    })
                    
                    if tool_result.get('success'):
                        return {
                            "success": True,
                            "result": f"Knowledge search: {tool_result['result'][:200]}...",
                            "method": "search_engine",
                            "subtask_type": subtask_type,
                            "external_tool_used": True
                        }
            
            # Fallback to basic mock
            return self._mock_solve_problem(problem, subtask_type)
            
        except Exception as e:
            logger.error(f"External tools mock error: {e}")
            return {
                "success": False,
                "result": f"External tools error: {str(e)}",
                "method": "external_tools_mock",
                "subtask_type": subtask_type,
                "error": str(e)
            }
    
    def _create_mock_crew(self):
        """Create mock crew when CrewAI is not available."""
        print("🤖 Creating mock crew implementation")
        
    def _create_mock_llm(self):
        """Create mock LLM when actual LLM is not available."""
        class MockLLM:
            def __init__(self):
                self.model = "mock-llm"
            
            def generate(self, prompt: str) -> str:
                return f"Mock response to: {prompt[:50]}..."
        
        return MockLLM()
    
    def _get_default_agents_config(self) -> Dict[str, Any]:
        """Get default agents configuration."""
        return {
            "interpreter": {
                "role": "Mathematical Problem Interpreter",
                "goal": "Parse and understand mathematical problems",
                "backstory": "Expert at interpreting mathematical language and expressions"
            },
            "type_classifier": {
                "role": "Problem Type Classifier", 
                "goal": "Classify mathematical problems by type",
                "backstory": "Specialist in categorizing mathematical problem types"
            },
            "tool_selector": {
                "role": "Mathematical Tool Selector",
                "goal": "Select appropriate tools for mathematical operations",
                "backstory": "Expert at choosing the right mathematical tools"
            },
            "sympy_executor": {
                "role": "Symbolic Mathematics Executor",
                "goal": "Execute symbolic mathematical operations",
                "backstory": "Specialist in symbolic mathematics using SymPy"
            },
            "result_aggregator": {
                "role": "Results Aggregator",
                "goal": "Combine and present final results",
                "backstory": "Expert at synthesizing mathematical results"
            }
        }
    
    def _get_default_tasks_config(self) -> Dict[str, Any]:
        """Get default tasks configuration."""
        return {
            "parse_and_classify_task": {
                "description": "Parse and interpret the mathematical problem",
                "expected_output": "Structured representation of the problem"
            },
            "classify_task_type": {
                "description": "Classify the type of mathematical problem",
                "expected_output": "Problem classification result"
            },
            "select_tool_task": {
                "description": "Select appropriate mathematical tools",
                "expected_output": "List of recommended tools"
            },
            "execute_symbolic_task": {
                "description": "Execute symbolic mathematical operations",
                "expected_output": "Symbolic computation results"
            },
            "aggregate_results_task": {
                "description": "Aggregate and present final results",
                "expected_output": "Complete solution with explanation"
            }
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about available agents and components."""
        return {
            "total_agents": len(self.agents),
            "agent_names": list(self.agents.keys()),
            "crewai_available": CREWAI_AVAILABLE,
            "tools_available": TOOLS_AVAILABLE,
            "rl_policy_available": RL_POLICY_AVAILABLE,
            "external_tools_available": EXTERNAL_TOOLS_AVAILABLE,
            "llm_model": getattr(self.llm, 'model', 'mock-llm'),
            "capabilities": {
                "mathematical_tools": TOOLS_AVAILABLE,
                "policy_learning": RL_POLICY_AVAILABLE,
                "external_integration": EXTERNAL_TOOLS_AVAILABLE,
                "crew_coordination": CREWAI_AVAILABLE
            }
        }

# Global crew manager instance
crew_manager = EnhancedCrewManager()

def get_crew_manager() -> EnhancedCrewManager:
    """Get the global crew manager instance."""
    return crew_manager

def solve_with_agents(problem: str, subtask_type: str = "general") -> Dict[str, Any]:
    """
    Convenience function to solve problems with agents.
    
    Args:
        problem: Mathematical problem description
        subtask_type: Type of mathematical subtask
    
    Returns:
        Solution result from the crew
    """
    return crew_manager.solve_mathematical_problem(problem, subtask_type)

if __name__ == "__main__":
    print("🤖 Enhanced CrewAI Integration for Reasoning Engine")
    print("=" * 55)
    
    # Test the crew manager
    manager = get_crew_manager()
    info = manager.get_agent_info()
    
    print(f"Agent Info: {info}")
    
    # Test problem solving
    print(f"\n🧮 Testing problem solving...")
    test_problem = "Simplify the expression: (3x + 2x) - (x - 4)"
    result = solve_with_agents(test_problem, "algebraic_simplification")
    
    print(f"Test Result: {result}")
    print(f"\n✅ Enhanced CrewAI integration ready!")

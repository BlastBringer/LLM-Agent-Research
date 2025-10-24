#!/usr/bin/env python3
"""
👨‍🏫 ORACLE MODEL - The Expert Teacher
======================================

The Oracle is the "teacher" that provides gold-standard solutions when the
Apprentice fails. It uses:
- State-of-the-art LLM (Claude 3.5 Sonnet or GPT-4o)
- ReAct (Reason + Act) framework for step-by-step reasoning
- Tool calling (Python interpreter) for accurate calculations
- Comprehensive solution traces for training data

The Oracle's solutions are saved and used to fine-tune the Apprentice.

Key Features:
- LangChain Agent with PythonREPL tool
- Detailed reasoning at each step
- Verification of calculations
- Formatted output for training data

Usage:
    oracle = OracleModel()
    solution = oracle.solve(problem_data)
    print(solution.reasoning_steps)
    print(solution.final_answer)
"""

import os
import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain_experimental.utilities import PythonREPL
    from langchain import hub
    
    # Try importing agent components (may fail with version conflicts)
    try:
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.tools import Tool
        AGENT_AVAILABLE = True
    except ImportError as agent_err:
        AGENT_AVAILABLE = False
        print(f"⚠️  LangChain agents not available (version conflict): {agent_err}")
        print(f"   Oracle will be disabled. This is OK for simple problems.")
    
    LANGCHAIN_AVAILABLE = AGENT_AVAILABLE  # Oracle needs agents
    
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    AGENT_AVAILABLE = False
    print(f"⚠️  LangChain import error: {e}")
    print("   Install with: pip install langchain langchain-openai langchain-experimental")
except Exception as e:
    LANGCHAIN_AVAILABLE = False
    AGENT_AVAILABLE = False
    print(f"❌ Unexpected error importing LangChain: {e}")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OracleSolution:
    """Result from the oracle model."""
    final_answer: Optional[float]
    reasoning_steps: List[str]
    tool_calls: List[Dict[str, Any]]  # Record of all tool usage
    raw_output: str
    confidence: float
    metadata: Dict[str, Any]


class OracleModel:
    """
    The Oracle (Teacher) Model.
    Uses a powerful LLM with ReAct framework and tool calling to provide
    expert-level solutions for training the Apprentice.
    """
    
    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.1,
        max_iterations: int = 10
    ):
        """
        Initialize the oracle model.
        
        Args:
            model_name: Model to use (default: Claude 3.5 Sonnet via OpenRouter)
            api_key: OpenRouter API key
            base_url: API base URL
            temperature: Sampling temperature
            max_iterations: Max reasoning iterations for agent
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.model_name = model_name or os.getenv("ORACLE_MODEL", "anthropic/claude-3.5-sonnet")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        # Initialize components
        self.llm = None
        self.agent = None
        self.agent_executor = None  # Initialize to None
        self.tools = []
        
        self._initialize_llm()
        self._initialize_tools()
        self._initialize_agent()
        
        self.logger.info(f"👨‍🏫 Oracle Model initialized: {self.model_name}")
    
    def _initialize_llm(self):
        """Initialize the powerful LLM."""
        if not LANGCHAIN_AVAILABLE:
            self.logger.error("❌ LangChain not available")
            return
        
        if not self.api_key:
            self.logger.error("❌ No API key found")
            return
        
        try:
            # Check if using DeepSeek R1 - needs special handling
            is_deepseek = "deepseek" in self.model_name.lower()
            
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                max_tokens=2000 if is_deepseek else 4000,  # DeepSeek needs token limit
                timeout=90 if is_deepseek else 60  # DeepSeek can be slower
            )
            self.logger.info(f"✅ Oracle LLM initialized: {self.model_name}")
            if is_deepseek:
                self.logger.info("ℹ️  DeepSeek R1 detected - using optimized settings")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Oracle LLM: {e}")
            self.llm = None
    
    def _initialize_tools(self):
        """Initialize tools for the oracle (Python interpreter, calculator)."""
        if not LANGCHAIN_AVAILABLE:
            return
        
        try:
            # Python REPL for executing code
            python_repl = PythonREPL()
            
            # Wrap in a tool with description
            python_tool = Tool(
                name="Python_Calculator",
                description="""A Python interpreter for mathematical calculations.
                Use this for ANY numerical computation to ensure accuracy.
                Input should be valid Python code that prints the result.
                Example: 'print(60 * 2)' to calculate 60 times 2.
                Always use print() to see the result.""",
                func=python_repl.run
            )
            
            self.tools.append(python_tool)
            self.logger.info(f"✅ Oracle tools initialized: {len(self.tools)} tools")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize tools: {e}")
            self.tools = []
    
    def _initialize_agent(self):
        """Initialize the ReAct agent."""
        if not self.llm or not self.tools:
            self.logger.warning("⚠️  Cannot initialize agent without LLM and tools")
            self.agent_executor = None  # Ensure it's set
            return
        
        try:
            # Custom ReAct prompt for math problem solving
            react_prompt = PromptTemplate(
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                template="""You are an expert mathematics teacher solving word problems step-by-step.

Your goal is to provide a clear, detailed solution that a student can learn from.

AVAILABLE TOOLS:
{tools}

TOOL NAMES: {tool_names}

INSTRUCTIONS:
1. Break down the problem into clear steps
2. Use the Python_Calculator tool for ALL numerical calculations (never do mental math)
3. Explain your reasoning at each step
4. Show your work clearly
5. Verify your final answer

FORMAT:
You must use this EXACT format:

Thought: I need to understand what the problem is asking
Action: Python_Calculator
Action Input: # any needed calculation or just pass
Observation: [tool output will appear here]

... (repeat Thought/Action/Action Input/Observation as needed)

Thought: I now know the final answer
Final Answer: [your complete solution with final numerical answer]

PROBLEM:
{input}

Begin! Remember to:
- Think step-by-step
- Use the Python_Calculator for ALL calculations
- Explain each step clearly
- Provide the final answer at the end

{agent_scratchpad}"""
            )
            
            # Create ReAct agent
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt
            )
            
            # Create executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=self.max_iterations,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            self.logger.info("✅ Oracle agent initialized with ReAct framework")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agent: {e}")
            self.agent = None
            self.agent_executor = None  # Ensure it's set on error
    
    def _clean_deepseek_output(self, text: str) -> str:
        """
        Clean DeepSeek R1 output by removing verbose <think> tags.
        
        DeepSeek R1 often includes very long reasoning in <think>...</think> tags.
        We strip these to get cleaner training data.
        """
        if not text:
            return text
        
        # Remove <think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove multiple consecutive newlines
        cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def solve(self, problem_data: Dict[str, Any]) -> OracleSolution:
        """
        Solve a math problem using the oracle's expert knowledge.
        
        Args:
            problem_data: All processed data from previous pipeline stages
        
        Returns:
            OracleSolution with detailed reasoning and answer
        """
        if not self.agent_executor:
            return self._create_error_solution("Oracle agent not initialized")
        
        self.logger.info("👨‍🏫 Oracle solving problem...")
        
        # Build a comprehensive problem description
        problem_text = self._build_problem_text(problem_data)
        
        try:
            # Run the agent
            result = self.agent_executor.invoke({"input": problem_text})
            
            # Extract information
            output = result.get('output', '')
            intermediate_steps = result.get('intermediate_steps', [])
            
            # Parse the solution
            solution = self._parse_solution(output, intermediate_steps)
            
            self.logger.info(f"✅ Oracle solution complete: {solution.final_answer}")
            return solution
            
        except Exception as e:
            self.logger.error(f"❌ Oracle failed: {e}")
            return self._create_error_solution(str(e))
    
    def _build_problem_text(self, problem_data: Dict[str, Any]) -> str:
        """Build a comprehensive problem description for the oracle."""
        parts = []
        
        # Original problem
        if 'original_problem' in problem_data:
            parts.append(f"PROBLEM:\n{problem_data['original_problem']}\n")
        
        # Known variables
        variables = {}
        if 'unit_standardization' in problem_data:
            std_vars = problem_data['unit_standardization'].get('standardized_variables', {})
            for var_name, var_data in std_vars.items():
                value = var_data.get('standardized_value', 0)
                unit = var_data.get('standardized_unit', '')
                variables[var_name] = f"{value} {unit}".strip()
        
        if variables:
            parts.append("KNOWN VALUES:")
            for var, val in variables.items():
                parts.append(f"  - {var} = {val}")
            parts.append("")
        
        # Equations
        if 'parsing' in problem_data:
            equations = []
            for eq in problem_data['parsing'].get('equations', []):
                if isinstance(eq, dict):
                    equations.append(eq.get('equation_string', ''))
            
            if equations:
                parts.append("RELEVANT EQUATIONS:")
                for eq in equations:
                    parts.append(f"  - {eq}")
                parts.append("")
            
            # Target
            target = problem_data['parsing'].get('target_variable', '')
            if target:
                parts.append(f"FIND: {target}\n")
        
        parts.append("Please solve this step-by-step, using the Python_Calculator for all calculations.")
        
        return "\n".join(parts)
    
    def _parse_solution(
        self,
        output: str,
        intermediate_steps: List[tuple]
    ) -> OracleSolution:
        """Parse the oracle's solution into a structured format."""
        
        # Clean DeepSeek R1 output if needed
        is_deepseek = "deepseek" in self.model_name.lower()
        if is_deepseek:
            output = self._clean_deepseek_output(output)
        
        # Extract reasoning steps from intermediate steps
        reasoning_steps = []
        tool_calls = []
        
        for step in intermediate_steps:
            if len(step) >= 2:
                action, observation = step[0], step[1]
                
                # Extract thought
                if hasattr(action, 'log'):
                    log = action.log
                    # Parse the thought from the log
                    thought_match = re.search(r'Thought:(.*?)(?:Action:|$)', log, re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        # Clean DeepSeek verbose output
                        if is_deepseek:
                            thought = self._clean_deepseek_output(thought)
                        if thought:
                            reasoning_steps.append(f"Thought: {thought}")
                    
                    # Record tool call
                    if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                        tool_calls.append({
                            'tool': action.tool,
                            'input': action.tool_input,
                            'output': str(observation)
                        })
                        reasoning_steps.append(f"Action: {action.tool}({action.tool_input})")
                        reasoning_steps.append(f"Result: {observation}")
        
        # Extract final answer from output
        final_answer = self._extract_final_answer(output)
        
        # Add final conclusion
        if output:
            reasoning_steps.append(f"Final Answer: {output}")
        
        return OracleSolution(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_calls,
            raw_output=output,
            confidence=0.95,  # High confidence in oracle
            metadata={
                'num_steps': len(intermediate_steps),
                'num_tool_calls': len(tool_calls),
                'model': self.model_name
            }
        )
    
    def _extract_final_answer(self, text: str) -> Optional[float]:
        """Extract the numerical final answer from text."""
        # Look for common patterns
        patterns = [
            r'final[_\s]answer[:\s]+(\d+\.?\d*)',
            r'answer[:\s]+(\d+\.?\d*)',
            r'result[:\s]+(\d+\.?\d*)',
            r'=\s*(\d+\.?\d*)\s*(?:\n|$)',
            r'(\d+\.?\d*)\s*(?:is the answer|is the final answer)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        
        # Last resort: find the last number in the text
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass
        
        return None
    
    def _create_error_solution(self, error_msg: str) -> OracleSolution:
        """Create an error solution."""
        return OracleSolution(
            final_answer=None,
            reasoning_steps=[f"Error: {error_msg}"],
            tool_calls=[],
            raw_output="",
            confidence=0.0,
            metadata={'error': error_msg}
        )
    
    def format_for_training(
        self,
        problem: str,
        solution: 'OracleSolution'
    ) -> Dict[str, Any]:
        """
        Format the oracle's solution for training data.
        
        Returns a dictionary ready to be saved to JSONL for fine-tuning.
        """
        return {
            'problem': problem,
            'steps': solution.reasoning_steps,
            'final_answer': solution.final_answer,
            'tool_calls': solution.tool_calls,
            'metadata': {
                'source': 'oracle',
                'model': self.model_name,
                'confidence': solution.confidence
            }
        }


if __name__ == "__main__":
    # Test the oracle
    print("🧪 Testing Oracle Model")
    print("=" * 70)
    
    # Sample problem
    test_problem = {
        'original_problem': 'A train travels at 60 miles per hour for 2.5 hours. How far does it travel?',
        'parsing': {
            'equations': [{'equation_string': 'distance = speed * time'}],
            'target_variable': 'distance'
        },
        'unit_standardization': {
            'standardized_variables': {
                'speed': {'standardized_value': 60, 'standardized_unit': 'mph'},
                'time': {'standardized_value': 2.5, 'standardized_unit': 'hours'}
            }
        }
    }
    
    oracle = OracleModel()
    
    if oracle.agent_executor:
        solution = oracle.solve(test_problem)
        
        print("\n📝 Oracle's Reasoning:")
        for i, step in enumerate(solution.reasoning_steps, 1):
            print(f"  {i}. {step}")
        
        print(f"\n🎯 Final Answer: {solution.final_answer}")
        print(f"📊 Confidence: {solution.confidence}")
        print(f"🔧 Tool Calls: {len(solution.tool_calls)}")
        
        # Show training format
        print("\n💾 Training Data Format:")
        training_data = oracle.format_for_training(
            test_problem['original_problem'],
            solution
        )
        print(json.dumps(training_data, indent=2))
    else:
        print("❌ Oracle not initialized. Check your .env file.")

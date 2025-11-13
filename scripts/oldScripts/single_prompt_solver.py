"""
Single-Prompt Solver: All-in-one approach combining templating, parsing, variable extraction, and solving.
Tests whether structured guidance in a single prompt performs better than multi-stage pipeline.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculator_tool(expression: str) -> str:
    """
    A safe calculator that evaluates mathematical expressions.
    Supports basic arithmetic: +, -, *, /, **, (), and common math functions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "63 / 13" or "4.846 * 39")
    
    Returns:
        The result of the calculation as a string
    """
    import math
    import re
    
    try:
        # Remove any whitespace
        expression = expression.strip()
        
        # Create a safe namespace with only math functions
        safe_dict = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
        }
        
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        # Format the result nicely
        if isinstance(result, float):
            # Round to reasonable precision
            if abs(result - round(result)) < 0.0001:
                return str(int(round(result)))
            else:
                return f"{result:.4f}".rstrip('0').rstrip('.')
        else:
            return str(result)
            
    except Exception as e:
        return f"Error: Could not evaluate '{expression}'. {str(e)}"


class SinglePromptSolver:
    """Solver that uses one comprehensive prompt for the entire problem-solving process."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.3, use_calculator: bool = False):
        """
        Initialize the single-prompt solver.
        
        Args:
            model_name: Model to use (defaults to APPRENTICE_MODEL from .env)
            temperature: Temperature for generation (0.3 for more focused reasoning)
            use_calculator: Whether to process calculator calls in the model's response
        """
        self.model_name = model_name or os.getenv("APPRENTICE_MODEL", "meta-llama/llama-3.2-3b-instruct")
        self.temperature = temperature
        self.use_calculator = use_calculator
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            max_tokens=2000
        )
        
        logger.info(f"Initialized SinglePromptSolver with model: {self.model_name}, calculator: {self.use_calculator}")
    
    def _process_calculator_expressions(self, text: str) -> str:
        """
        Find and evaluate CALC[expression] in the text.
        Replaces each CALC[...] with the computed result.
        
        Args:
            text: Text containing CALC[...] expressions
            
        Returns:
            Text with CALC[...] replaced by calculated values
        """
        def replace_calc(match):
            expression = match.group(1)
            result = calculator_tool(expression)
            return f"{result}"
        
        # Find all CALC[...] patterns and replace with calculated values
        processed = re.sub(r'CALC\[(.*?)\]', replace_calc, text)
        return processed
    
    
    def _build_comprehensive_prompt(self, problem: str) -> str:
        """
        Build a comprehensive prompt that guides the model through all stages.
        
        This prompt instructs the model to:
        1. Read and understand the problem
        2. Extract and identify all relevant data
        3. Set up the mathematical structure
        4. Solve step-by-step with actual calculations
        5. Provide final answer
        """
        
        calculator_instructions = ""
        if self.use_calculator:
            calculator_instructions = """
**YOU HAVE ACCESS TO A CALCULATOR!**
Whenever you need to perform ANY calculation, write it like this:
CALC[expression]

For example:
- To calculate 63 ÷ 13, write: CALC[63 / 13]
- To calculate 4.846 × 39, write: CALC[4.846 * 39]  
- The calculator will evaluate it and show you the result

DO NOT do mental math - ALWAYS use CALC[...] for ALL calculations!
"""
        
        prompt = f"""You are a mathematical problem solver. I will guide you through solving this problem step-by-step like a student learning to solve problems systematically.

{calculator_instructions}

**PROBLEM:**
{problem}

**YOUR TASK:**
Solve this problem by following these phases carefully. Show your work for each phase.

**PHASE 1: UNDERSTAND AND TEMPLATE**
Read the problem and identify:
- What is being asked? (the target/question)
- What type of problem is this? (arithmetic, rate, proportion, multi-step, etc.)
- What are the key relationships or operations needed?

**PHASE 2: EXTRACT ALL DATA**
List ALL numerical values and what they represent:
- Write each value with its unit and meaning
- Include rates (e.g., "63 minutes per 13 miles")
- Include totals, quantities, prices, times, distances, etc.
- Don't skip any numbers from the problem!

**PHASE 3: SET UP THE MATHEMATICS**
Define variables and equations:
- Assign variable names to unknown quantities
- Write equations that relate the given data to what you need to find
- For multi-step problems, identify which calculation must come first
- For rate problems, remember: find the rate first, then apply it

**PHASE 4: SOLVE WITH ACTUAL CALCULATIONS**
Solve step-by-step, showing ALL arithmetic:
- {"Use CALC[expression] for EVERY calculation" if self.use_calculator else "Write each calculation explicitly (e.g., '63 ÷ 13 = 4.846')"}
- Show intermediate results with units
- For multi-step problems, solve dependencies first
- Verify your answer makes sense

**PHASE 5: FINAL ANSWER**
State your final answer clearly with the appropriate unit.

---

**IMPORTANT GUIDELINES:**
- Show actual numbers and calculations, not generic statements like "divide the values"
- Keep track of units throughout (minutes, miles, dollars, etc.)
- For rate problems: Rate = Amount/Unit, then apply rate to find total
- {"ALWAYS use CALC[...] - do NOT do mental math!" if self.use_calculator else "Double-check your arithmetic"}
- Make sure your final answer directly addresses what was asked

**EXAMPLE OF GOOD REASONING:**
✓ {"Calculate the rate: CALC[63 / 13] = 4.8462 minutes per mile" if self.use_calculator else "Calculate the rate: 63 minutes ÷ 13 miles = 4.846 minutes per mile"}
✓ {"Apply the rate: CALC[4.8462 * 39] = 189.0 minutes" if self.use_calculator else "Apply the rate: 4.846 minutes/mile × 39 miles = 189.0 minutes"}

**EXAMPLE OF BAD REASONING:**
✗ "Divide the time by distance to get the rate"
✗ "Multiply to get the answer"

---

**CRITICAL: FORMAT YOUR FINAL ANSWER**
After completing all phases, you MUST end your response with the final answer on the last line in this exact format:

Answer: [your computed numeric value]

For example:
- Answer: 189.0
- Answer: 42
- Answer: 3.5

Do NOT include units in this final line, just the numeric value.

---

Now solve the problem following all five phases. Be thorough and show your work!

**YOUR SOLUTION:**
"""
        
        return prompt
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve a single problem using the comprehensive single-prompt approach.
        
        Args:
            problem: The problem text to solve
            
        Returns:
            Dictionary with:
                - answer: The extracted numeric answer
                - reasoning: The full reasoning provided by the model
                - raw_response: The complete model response
                - success: Whether a valid answer was extracted
        """
        
        try:
            # Build the comprehensive prompt
            prompt = self._build_comprehensive_prompt(problem)
            
            # Get model response
            messages = [
                SystemMessage(content="You are a helpful math tutor who solves problems systematically and shows all work."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            raw_response = response.content
            
            # Process calculator expressions if enabled
            if self.use_calculator:
                raw_response = self._process_calculator_expressions(raw_response)
            
            logger.debug(f"Model response:\n{raw_response}")
            
            # Extract answer from response
            answer = self._extract_answer(raw_response)
            
            result = {
                "answer": answer,
                "reasoning": raw_response,
                "raw_response": raw_response,
                "success": answer is not None
            }
            
            if answer is not None:
                logger.info(f"Successfully solved problem. Answer: {answer}")
            else:
                logger.warning("Could not extract answer from model response")
            
            return result
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                "answer": None,
                "reasoning": "",
                "raw_response": "",
                "success": False,
                "error": str(e)
            }
    
    def _extract_answer(self, response: str) -> Optional[float]:
        """
        Extract numeric answer from model response.
        Looks for patterns like:
        - "Answer: 123" (preferred format - should be on last line)
        - "Final answer: 123"
        - "= 123" (at the end)
        - Numbers in the last line
        """
        
        import re
        
        # First, try to find "Answer: number" pattern (our requested format)
        # Look for this pattern specifically on the last few lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-3:]):  # Check last 3 lines
            match = re.search(r'^Answer\s*:\s*([-+]?[\d,]+\.?\d*)', line.strip(), re.IGNORECASE)
            if match:
                try:
                    answer_str = match.group(1).replace(',', '')
                    return float(answer_str)
                except ValueError:
                    continue
        
        # Try to find "Final answer:" or "Answer:" patterns anywhere
        patterns = [
            r"(?:final\s+)?answer\s*:?\s*[=]?\s*([-+]?[\d,]+\.?\d*)",
            r"(?:the\s+)?answer\s*:?\s*[=]?\s*([-+]?[\d,]+\.?\d*)",
            r"(?:equals|is)\s+([-+]?[\d,]+\.?\d*)\s*(?:minutes|miles|dollars|hours|days|pounds|feet|inches|gallons|years|months|weeks|seconds)?",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    # Remove commas and convert to float
                    answer_str = match.group(1).replace(',', '')
                    return float(answer_str)
                except ValueError:
                    continue
        
        # Try to find the last number in the response
        # Split into lines and check last few lines
        for line in reversed(lines[-5:]):  # Check last 5 lines
            # Look for standalone numbers
            match = re.search(r'[=:]\s*([-+]?[\d,]+\.?\d*)\s*(?:minutes|miles|dollars|hours|days|pounds|feet|inches|gallons|years|months|weeks|seconds)?', line)
            if match:
                try:
                    answer_str = match.group(1).replace(',', '')
                    return float(answer_str)
                except ValueError:
                    continue
        
        return None


def extract_ground_truth(problem_data: Dict[str, Any]) -> Optional[float]:
    """Extract ground truth answer from problem data."""
    
    # Try to get from 'output' field (GSM-Symbolic format)
    if 'output' in problem_data:
        output = problem_data['output']
        
        # Handle case where output is already a number
        if isinstance(output, (int, float)):
            return float(output)
        
        # Handle string output
        if isinstance(output, str):
            # Look for "#### number" pattern
            import re
            match = re.search(r'####\s*([-+]?[\d,]+\.?\d*)', output)
            if match:
                try:
                    return float(match.group(1).replace(',', ''))
                except ValueError:
                    pass
    
    # Try 'answer' field
    if 'answer' in problem_data:
        try:
            return float(str(problem_data['answer']).replace(',', ''))
        except ValueError:
            pass
    
    return None


def compare_answers(predicted: Optional[float], ground_truth: Optional[float], tolerance: float = 0.01) -> bool:
    """
    Compare predicted answer with ground truth.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Relative tolerance (default 1%)
    
    Returns:
        True if answers match within tolerance
    """
    
    if predicted is None or ground_truth is None:
        return False
    
    # Handle exact match
    if predicted == ground_truth:
        return True
    
    # Handle zero ground truth
    if ground_truth == 0:
        return abs(predicted) < 0.001
    
    # Relative difference
    rel_diff = abs(predicted - ground_truth) / abs(ground_truth)
    return rel_diff <= tolerance


def test_single_prompt_solver(input_file: str, output_file: str, max_workers: int = 10, use_calculator: bool = False):
    """
    Test the single-prompt solver on a dataset.
    
    Args:
        input_file: Path to input JSONL file (e.g., gsm_symbolic_batch1.jsonl)
        output_file: Path to save results
        max_workers: Number of parallel workers
        use_calculator: Whether to give the model access to a calculator tool
    """
    
    logger.info(f"Testing Single-Prompt Solver")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Workers: {max_workers}")
    logger.info(f"Calculator: {use_calculator}")
    
    # Load problems
    problems = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    
    logger.info(f"Loaded {len(problems)} problems")
    
    # Initialize solver
    solver = SinglePromptSolver(use_calculator=use_calculator)
    
    # Results storage
    results = []
    correct_count = 0
    total_with_gt = 0
    
    def solve_single(idx: int, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a single problem."""
        
        # Try multiple field names for the problem text
        problem_text = problem_data.get('input', problem_data.get('question', problem_data.get('problem', '')))
        
        logger.info(f"Processing problem {idx + 1}/{len(problems)}")
        
        # Solve
        result = solver.solve_problem(problem_text)
        
        # Get ground truth
        ground_truth = extract_ground_truth(problem_data)
        
        # Compare
        is_correct = False
        if ground_truth is not None and result['answer'] is not None:
            is_correct = compare_answers(result['answer'], ground_truth)
        
        return {
            'problem_id': idx,
            'question': problem_text,
            'predicted_answer': result['answer'],
            'ground_truth': ground_truth,
            'correct': is_correct,
            'reasoning': result['reasoning'],
            'success': result['success']
        }
    
    # Solve problems in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(solve_single, idx, problem_data): idx
            for idx, problem_data in enumerate(problems)
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                
                # Update stats
                if result['ground_truth'] is not None:
                    total_with_gt += 1
                    if result['correct']:
                        correct_count += 1
                
            except Exception as e:
                logger.error(f"Error processing problem: {e}")
    
    # Sort results by problem_id
    results.sort(key=lambda x: x['problem_id'])
    
    # Calculate accuracy
    accuracy = (correct_count / total_with_gt * 100) if total_with_gt > 0 else 0
    
    # Save results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total problems: {len(problems)}")
    logger.info(f"Problems with ground truth: {total_with_gt}")
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*60}\n")
    
    return results, accuracy


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test single-prompt solver on math problems")
    parser.add_argument(
        '--input',
        type=str,
        default='gsm_symbolic_batch1.jsonl',
        help='Input JSONL file with problems'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='single_prompt_results.jsonl',
        help='Output JSONL file for results'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--calculator',
        action='store_true',
        help='Give the model access to a calculator tool for accurate calculations'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Test on a single problem (provide problem text)'
    )
    
    args = parser.parse_args()
    
    if args.single:
        # Test single problem
        solver = SinglePromptSolver(use_calculator=args.calculator)
        result = solver.solve_problem(args.single)
        
        print("\n" + "="*60)
        print("PROBLEM:")
        print(args.single)
        print("\n" + "="*60)
        print("SOLUTION:")
        print(result['reasoning'])
        print("\n" + "="*60)
        print(f"EXTRACTED ANSWER: {result['answer']}")
        print("="*60 + "\n")
    else:
        # Test on dataset
        test_single_prompt_solver(args.input, args.output, args.workers, args.calculator)


if __name__ == "__main__":
    main()

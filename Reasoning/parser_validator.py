#!/usr/bin/env python3
"""
Parser Training Data Generator and Validator
===========================================

This module creates training data from the existing datasets and validates
the enhanced parser performance.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import re

class ParserDatasetProcessor:
    """
    Processes the existing datasets to create parser training/validation data
    and evaluates parser performance.
    """
    
    def __init__(self, datasets_dir: str = "/home/karthik-g-s/Desktop/Capstone/Implementation"):
        self.datasets_dir = Path(datasets_dir)
        self.math_data = []
        self.amps_data = []
        
    def load_datasets(self):
        """Load both MATH and AMPS datasets."""
        print("📂 Loading datasets...")
        
        # Load MATH dataset
        math_file = self.datasets_dir / "AllProblemsCleaned.jsonl"
        if math_file.exists():
            with open(math_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        self.math_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Error parsing MATH line {line_num}: {e}")
                        
        print(f"   ✅ Loaded {len(self.math_data)} MATH problems")
        
        # Load AMPS dataset (sample first 1000 for testing)
        amps_file = self.datasets_dir / "amps.jsonl"
        if amps_file.exists():
            with open(amps_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num > 1000:  # Limit for testing
                        break
                    try:
                        data = json.loads(line.strip())
                        self.amps_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Error parsing AMPS line {line_num}: {e}")
                        
        print(f"   ✅ Loaded {len(self.amps_data)} AMPS problems (sample)")
    
    def create_parser_examples(self, num_examples: int = 50) -> List[Dict[str, Any]]:
        """
        Create diverse examples for parser training from the datasets.
        """
        examples = []
        
        # Sample from MATH dataset
        math_sample = random.sample(self.math_data, min(num_examples // 2, len(self.math_data)))
        for item in math_sample:
            problem_text = item.get('input', '') or item.get('latex_format', '')
            if problem_text:
                examples.append({
                    'problem': problem_text,
                    'source': 'MATH',
                    'difficulty': item.get('difficulty', 'Unknown'),
                    'tags': item.get('tags', []),
                    'expected_type': self._classify_math_problem(problem_text, item.get('tags', []))
                })
        
        # Sample from AMPS dataset
        amps_sample = random.sample(self.amps_data, min(num_examples // 2, len(self.amps_data)))
        for item in amps_sample:
            problem_text = item.get('problem', '')
            if problem_text:
                examples.append({
                    'problem': problem_text,
                    'source': 'AMPS',
                    'difficulty': 'Algebraic',
                    'tags': ['algebra', 'manipulation'],
                    'expected_type': 'algebraic_manipulation'
                })
        
        return examples
    
    def _classify_math_problem(self, problem: str, tags: List[str]) -> str:
        """Classify MATH dataset problems based on content and tags."""
        problem_lower = problem.lower()
        tag_str = ' '.join(tags).lower()
        
        # Classification based on tags and content
        if any(tag in tag_str for tag in ['calculus', 'derivative', 'integral']):
            return 'calculus'
        elif any(tag in tag_str for tag in ['algebra', 'polynomial']):
            if 'system' in problem_lower or ('equation' in problem_lower and 'and' in problem_lower):
                return 'system_of_equations'
            else:
                return 'algebra'
        elif any(tag in tag_str for tag in ['geometry', 'coordinate']):
            return 'geometry'
        elif any(tag in tag_str for tag in ['number theory', 'combinatorics']):
            return 'discrete_math'
        elif 'inequality' in problem_lower or '\\ge' in problem or '\\le' in problem:
            return 'inequality'
        elif any(op in problem for op in ['+', '-', '*', '/', '=']):
            return 'algebra'
        else:
            return 'general_math'
    
    def validate_parser_on_dataset(self, parser, sample_size: int = 100) -> Dict[str, Any]:
        """
        Validate the enhanced parser on a sample of the datasets.
        """
        print(f"🔍 Validating parser on {sample_size} problems...")
        
        # Create test sample
        test_examples = self.create_parser_examples(sample_size)
        
        results = {
            'total_tested': len(test_examples),
            'successful_parses': 0,
            'parsing_errors': 0,
            'type_matches': 0,
            'by_source': {'MATH': {'total': 0, 'success': 0}, 'AMPS': {'total': 0, 'success': 0}},
            'by_type': {},
            'error_details': [],
            'sample_results': []
        }
        
        for i, example in enumerate(test_examples):
            source = example['source']
            expected_type = example['expected_type']
            problem = example['problem']
            
            results['by_source'][source]['total'] += 1
            
            if expected_type not in results['by_type']:
                results['by_type'][expected_type] = {'total': 0, 'success': 0}
            results['by_type'][expected_type]['total'] += 1
            
            try:
                parsed_result = parser.parse(problem)
                
                if parsed_result.get('problem_type') != 'parsing_failed':
                    results['successful_parses'] += 1
                    results['by_source'][source]['success'] += 1
                    results['by_type'][expected_type]['success'] += 1
                    
                    # Check if parsed type matches expected (loose matching)
                    parsed_type = parsed_result.get('problem_type', '').lower()
                    if self._types_match(parsed_type, expected_type):
                        results['type_matches'] += 1
                
                # Store sample results (first 10)
                if i < 10:
                    results['sample_results'].append({
                        'problem': problem[:100] + '...' if len(problem) > 100 else problem,
                        'expected_type': expected_type,
                        'parsed_type': parsed_result.get('problem_type'),
                        'success': parsed_result.get('problem_type') != 'parsing_failed',
                        'source': source
                    })
                    
            except Exception as e:
                results['parsing_errors'] += 1
                results['error_details'].append({
                    'problem': problem[:100] + '...',
                    'error': str(e),
                    'source': source
                })
                
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i + 1}/{len(test_examples)}")
        
        # Calculate success rates
        for source_data in results['by_source'].values():
            if source_data['total'] > 0:
                source_data['success_rate'] = source_data['success'] / source_data['total']
        
        for type_data in results['by_type'].values():
            if type_data['total'] > 0:
                type_data['success_rate'] = type_data['success'] / type_data['total']
        
        results['overall_success_rate'] = results['successful_parses'] / results['total_tested']
        results['type_accuracy'] = results['type_matches'] / results['total_tested']
        
        return results
    
    def _types_match(self, parsed_type: str, expected_type: str) -> bool:
        """Check if parsed and expected types are compatible."""
        type_mappings = {
            'calculus': ['derivative', 'integral', 'limit', 'calculus'],
            'algebra': ['algebra', 'polynomial', 'quadratic', 'linear', 'factoring'],
            'system_of_equations': ['system', 'linear_equations', 'simultaneous'],
            'geometry': ['geometry', 'triangle', 'circle', 'area', 'volume'],
            'inequality': ['inequality', 'greater', 'less'],
            'arithmetic': ['arithmetic', 'calculation', 'expression'],
            'algebraic_manipulation': ['algebra', 'manipulation', 'complete_square', 'factoring']
        }
        
        for category, variants in type_mappings.items():
            if expected_type in variants or expected_type == category:
                if any(variant in parsed_type for variant in variants) or category in parsed_type:
                    return True
        
        return parsed_type == expected_type
    
    def generate_fine_tuning_data(self, num_examples: int = 500) -> List[Dict[str, Any]]:
        """
        Generate training data for fine-tuning a parser model.
        """
        print(f"🎯 Generating {num_examples} fine-tuning examples...")
        
        examples = self.create_parser_examples(num_examples)
        fine_tuning_data = []
        
        for example in examples:
            # Create ideal parse structure for this problem
            ideal_parse = self._create_ideal_parse(example)
            
            fine_tuning_data.append({
                'messages': [
                    {
                        'role': 'user',
                        'content': f"Parse this math problem into structured JSON:\n\n{example['problem']}"
                    },
                    {
                        'role': 'assistant',
                        'content': json.dumps(ideal_parse, indent=2)
                    }
                ]
            })
        
        return fine_tuning_data
    
    def _create_ideal_parse(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Create an ideal parse structure for training."""
        problem = example['problem']
        problem_type = example['expected_type']
        
        # Extract basic information
        variables = self._extract_variables(problem)
        numbers = re.findall(r'-?\d+\.?\d*', problem)
        
        ideal_parse = {
            'problem_type': problem_type,
            'source': example['source'],
            'difficulty': example['difficulty']
        }
        
        if variables:
            ideal_parse['variables'] = {var: {'description': f'variable {var}', 'domain': 'real'} for var in variables}
        
        if numbers:
            ideal_parse['numerical_values'] = [float(n) for n in numbers if n]
        
        # Add problem-specific structure
        if 'equation' in problem.lower() and '=' in problem:
            equations = re.findall(r'[^=]+=+[^=]+', problem)
            if equations:
                ideal_parse['equations'] = equations
        
        if any(op in problem for op in ['+', '-', '*', '/', '^']):
            # Try to extract mathematical expressions
            expressions = re.findall(r'[a-zA-Z0-9+\-*/^().\s]+[=<>]', problem)
            if expressions:
                ideal_parse['expressions'] = expressions
        
        return ideal_parse
    
    def _extract_variables(self, problem: str) -> List[str]:
        """Extract mathematical variables from problem text."""
        # Find single letters that are likely variables
        variables = set()
        
        # Look for patterns like "x =", "y +", etc.
        var_patterns = [
            r'\b([a-zA-Z])\s*[=+\-*/^<>]',
            r'[=+\-*/^<>]\s*([a-zA-Z])\b',
            r'\b([a-zA-Z])\s*\^',
            r'solve.*for\s+([a-zA-Z])',
            r'find\s+([a-zA-Z])'
        ]
        
        for pattern in var_patterns:
            matches = re.findall(pattern, problem, re.IGNORECASE)
            variables.update(matches)
        
        # Filter out common words that aren't variables
        non_variables = {'a', 'an', 'is', 'in', 'of', 'to', 'and', 'or', 'the', 'be', 'it', 'as', 'on', 'at', 'by', 'for'}
        variables = {v.lower() for v in variables if v.lower() not in non_variables and len(v) == 1}
        
        return sorted(list(variables))
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save validation results to file."""
        output_path = Path(filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {output_path}")


def main():
    """Main function to test the dataset processor and parser validation."""
    print("🚀 PARSER DATASET PROCESSOR & VALIDATOR")
    print("=" * 60)
    
    # Initialize processor
    processor = ParserDatasetProcessor()
    processor.load_datasets()
    
    # Create sample examples
    print("\n📋 Creating sample examples...")
    examples = processor.create_parser_examples(20)
    
    print(f"Created {len(examples)} examples:")
    for i, example in enumerate(examples[:5], 1):
        print(f"  {i}. [{example['source']}] {example['problem'][:60]}...")
        print(f"     Expected type: {example['expected_type']}")
    
    # Generate fine-tuning data sample
    print("\n🎯 Generating fine-tuning data sample...")
    ft_data = processor.generate_fine_tuning_data(10)
    
    print(f"Generated {len(ft_data)} fine-tuning examples")
    print("Sample fine-tuning format:")
    print(json.dumps(ft_data[0], indent=2)[:500] + "...")
    
    # If enhanced parser is available, validate it
    try:
        from enhanced_problem_parser import EnhancedProblemParser
        
        print("\n🔍 Validating Enhanced Parser...")
        parser = EnhancedProblemParser()
        validation_results = processor.validate_parser_on_dataset(parser, sample_size=50)
        
        print("\n📊 VALIDATION RESULTS")
        print("-" * 40)
        print(f"Overall Success Rate: {validation_results['overall_success_rate']:.1%}")
        print(f"Type Accuracy: {validation_results['type_accuracy']:.1%}")
        print(f"Parsing Errors: {validation_results['parsing_errors']}")
        
        print("\nBy Source:")
        for source, data in validation_results['by_source'].items():
            if data['total'] > 0:
                print(f"  {source}: {data['success_rate']:.1%} ({data['success']}/{data['total']})")
        
        print("\nBy Type:")
        for ptype, data in validation_results['by_type'].items():
            if data['total'] > 0:
                print(f"  {ptype}: {data['success_rate']:.1%} ({data['success']}/{data['total']})")
        
        # Save results
        processor.save_results(validation_results, "parser_validation_results.json")
        
    except ImportError:
        print("⚠️ Enhanced parser not available for validation")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🔄 WORD PROBLEM TEMPLATIZATION ENGINE
====================================

This module converts word problems into generic templates by:
1. Identifying and replacing proper nouns with generic placeholders
2. Creating a legend/mapping for all replacements
3. Preserving mathematical relationships and structure
4. Using Chain-of-Thought reasoning for systematic processing

"""

import os
import sys
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Try to import spaCy
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        print("✅ spaCy NLP model loaded")
    except OSError:
        print("⚠️ spaCy model not found, using pattern-based fallback")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    print("⚠️ spaCy not available, using pattern-based fallback")
    SPACY_AVAILABLE = False
    nlp = None

# LangChain imports for robust processing
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import BaseOutputParser
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain with Gemini available for templatization")
except ImportError:
    print("⚠️ LangChain not available, using fallback methods")
    LANGCHAIN_AVAILABLE = False

@dataclass
class TemplatizationResult:
    """Result of the templatization process."""
    original_problem: str
    templated_problem: str
    legend: Dict[str, str]
    entities_found: Dict[str, List[str]]
    confidence_score: float
    processing_steps: List[str]
    metadata: Dict[str, Any]

class WordProblemTemplatizer:
    """
    Advanced word problem templatization engine using NLP and LangChain.
    
    This class converts word problems with proper nouns into generic templates
    while maintaining a mapping legend for reconstruction.
    """
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-exp"):
        """Initialize the templatizer with optional LangChain support."""
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("MODEL_NAME")
        self.llm = None
        
        # Initialize counters for generic names
        self.counters = {
            'PERSON': 0,
            'LOCATION': 0,
            'ORGANIZATION': 0,
            'ITEM': 0,
            'ANIMAL': 0,
            'VEHICLE': 0,
            'FOOD': 0,
            'CURRENCY': 0,
            'EVENT': 0,
            'MISC': 0
        }
        
        # Common proper noun patterns
        self.name_patterns = {
            'PERSON': [
                # Names with titles (Mr., Mrs., Dr., etc.)
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b',
            ],
            'LOCATION': [
                # Cities, streets with geographic indicators
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|Road|Avenue|Boulevard|St|Rd|Ave|Blvd|City|Town|Village|County|State)\.?\b',
            ],
            'ORGANIZATION': [
                # Companies, schools with corporate/institutional indicators
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Company|Corp|Corporation|Inc|LLC|Ltd|School|University|College|Institute)\.?\b',
            ]
        }
        
        # Common words to exclude (not proper nouns even if capitalized)
        self.exclude_words = {
            # Sentence starters and common words
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            # Question words
            'when', 'where', 'what', 'which', 'who', 'whom', 'whose', 'why', 'how',
            # Common sentence starters
            'if', 'then', 'that', 'this', 'these', 'those', 'there', 'here',
            # Time/measurement
            'after', 'before', 'since', 'until', 'while',
            # Modal verbs and auxiliaries  
            'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
            # Pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
        }
        
        # Initialize LangChain if available
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LangChain LLM if available."""
        if not LANGCHAIN_AVAILABLE or not self.api_key:
            self.logger.warning("LangChain or API key not available, using rule-based approach")
            return
            
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0.1,  # Low temperature for consistent results
                google_api_key=self.api_key
            )
            self.logger.info(f"✅ LangChain LLM initialized: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    def _extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy NLP."""
        if not SPACY_AVAILABLE:
            return {}
            
        doc = nlp(text)
        entities = {
            'PERSON': [],
            'LOCATION': [],
            'ORGANIZATION': [],
            'MISC': []
        }
        
        for ent in doc.ents:
            entity_text = ent.text.strip()
            
            # Filter out single characters and excluded words
            if len(entity_text) <= 1:
                continue
            if entity_text.lower() in self.exclude_words:
                continue
            
            # Also check if the entity starts with an excluded word (e.g., "If John")
            first_word = entity_text.split()[0].lower()
            if first_word in self.exclude_words:
                # Try to extract the actual name after the excluded word
                remaining = ' '.join(entity_text.split()[1:])
                if remaining and len(remaining) > 1:
                    entity_text = remaining
                else:
                    continue
                    
            if ent.label_ == "PERSON":
                entities['PERSON'].append(entity_text)
            elif ent.label_ in ["GPE", "LOC"]:  # Geo-political entity, Location
                entities['LOCATION'].append(entity_text)
            elif ent.label_ == "ORG":
                entities['ORGANIZATION'].append(entity_text)
            elif ent.label_ in ["PRODUCT", "WORK_OF_ART", "EVENT"]:
                entities['MISC'].append(entity_text)
        
        # Remove duplicates while preserving order
        for category in entities:
            entities[category] = list(dict.fromkeys(entities[category]))
            
        return entities
    
    def _extract_entities_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns as fallback."""
        entities = {category: [] for category in self.name_patterns}
        
        for category, patterns in self.name_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # Filter out excluded words
                    if match.lower() not in self.exclude_words:
                        # Check if first word is excluded
                        first_word = match.split()[0].lower()
                        if first_word not in self.exclude_words:
                            entities[category].append(match)
        
        # Remove duplicates
        for category in entities:
            entities[category] = list(dict.fromkeys(entities[category]))
            
        return entities
    
    def _extract_entities_llm(self, text: str) -> Tuple[Dict[str, List[str]], float]:
        """Extract entities using LangChain LLM with Chain-of-Thought reasoning."""
        if not self.llm:
            return {}, 0.0
            
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
You are an expert at analyzing word problems and identifying proper nouns that should be templatized.

Text to analyze: {text}

Think step by step:

1. **Identify all proper nouns** in the text (names of people, places, organizations, specific items, etc.)
2. **Categorize them** into appropriate types
3. **Consider context** - are they actually being used as proper nouns in this math problem?
4. **Filter out** common words that might be capitalized but aren't proper nouns

Provide your analysis in this JSON format:
{{
    "thinking_process": "Your step-by-step reasoning here",
    "entities": {{
        "PERSON": ["list of person names"],
        "LOCATION": ["list of place names"], 
        "ORGANIZATION": ["list of organization names"],
        "ITEM": ["list of specific item/product names"],
        "MISC": ["list of other proper nouns"]
    }},
    "confidence": 0.95
}}

Be careful to only include actual proper nouns that should be templatized for math problems.
"""
        )
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({"text": text})
            
            # Parse the JSON response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('entities', {}), result.get('confidence', 0.5)
            else:
                self.logger.warning("Could not parse LLM response for entity extraction")
                return {}, 0.0
                
        except Exception as e:
            self.logger.error(f"LLM entity extraction failed: {e}")
            return {}, 0.0
    
    def _generate_placeholder(self, entity: str, category: str) -> str:
        """Generate a placeholder for an entity."""
        self.counters[category] += 1
        return f"[{category.title()}{self.counters[category]}]"
    
    def _create_legend(self, replacements: Dict[str, str]) -> Dict[str, str]:
        """Create a legend mapping placeholders back to original entities."""
        return {placeholder: original for original, placeholder in replacements.items()}
    
    def _validate_mathematical_structure(self, original: str, templated: str) -> bool:
        """Ensure mathematical structure is preserved after templatization."""
        # Extract numbers and mathematical operators
        number_pattern = r'\d+(?:\.\d+)?'
        operator_pattern = r'[+\-*/=<>]'
        
        original_numbers = re.findall(number_pattern, original)
        templated_numbers = re.findall(number_pattern, templated)
        
        original_operators = re.findall(operator_pattern, original)
        templated_operators = re.findall(operator_pattern, templated)
        
        # Check if mathematical content is preserved
        return (len(original_numbers) == len(templated_numbers) and 
                len(original_operators) == len(templated_operators))
    
    def templatize_problem(self, problem: str) -> TemplatizationResult:
        """
        Main method to templatize a word problem.
        
        Args:
            problem (str): The original word problem text
            
        Returns:
            TemplatizationResult: Complete templatization result with legend
        """
        start_time = datetime.now()
        processing_steps = []
        
        processing_steps.append("🔄 Starting templatization process")
        
        # Step 1: Extract entities using multiple methods
        processing_steps.append("🔍 Extracting named entities")
        
        entities_spacy = self._extract_entities_spacy(problem) if SPACY_AVAILABLE else {}
        entities_patterns = self._extract_entities_patterns(problem)
        
        # Combine results, prioritizing spaCy if available
        entities_found = {}
        for category in ['PERSON', 'LOCATION', 'ORGANIZATION', 'ITEM', 'MISC']:
            entities_found[category] = []
            
            # Add spaCy results first
            if category in entities_spacy:
                entities_found[category].extend(entities_spacy[category])
            
            # Add pattern results if not already found
            if category in entities_patterns:
                for entity in entities_patterns[category]:
                    if entity not in entities_found[category]:
                        entities_found[category].append(entity)
        
        # Try LLM enhancement if available
        llm_confidence = 0.0
        if self.llm:
            processing_steps.append("🤖 Enhancing entity extraction with LLM")
            llm_entities, llm_confidence = self._extract_entities_llm(problem)
            
            # Merge LLM results
            for category, entity_list in llm_entities.items():
                if category in entities_found:
                    for entity in entity_list:
                        if entity not in entities_found[category]:
                            entities_found[category].append(entity)
        
        processing_steps.append(f"✅ Found entities: {sum(len(v) for v in entities_found.values())} total")
        
        # Step 2: Create replacements
        processing_steps.append("🔄 Creating entity replacements")
        
        replacements = {}
        templated_problem = problem
        
        # Process entities by length (longest first to avoid partial replacements)
        all_entities = []
        for category, entity_list in entities_found.items():
            for entity in entity_list:
                all_entities.append((entity, category))
        
        # Sort by length (descending) to replace longer entities first
        all_entities.sort(key=lambda x: len(x[0]), reverse=True)
        
        for entity, category in all_entities:
            if entity in templated_problem:
                placeholder = self._generate_placeholder(entity, category)
                replacements[entity] = placeholder
                
                # Use word boundaries to avoid partial replacements
                pattern = r'\b' + re.escape(entity) + r'\b'
                templated_problem = re.sub(pattern, placeholder, templated_problem)
        
        processing_steps.append(f"🔄 Applied {len(replacements)} replacements")
        
        # Step 3: Create legend
        legend = self._create_legend(replacements)
        
        # Step 4: Validate mathematical structure
        processing_steps.append("✅ Validating mathematical structure")
        structure_preserved = self._validate_mathematical_structure(problem, templated_problem)
        
        if not structure_preserved:
            processing_steps.append("⚠️ Mathematical structure may have been affected")
        
        # Step 5: Calculate confidence score
        base_confidence = 0.8 if entities_found else 0.5
        spacy_bonus = 0.1 if SPACY_AVAILABLE else 0.0
        llm_bonus = llm_confidence * 0.1 if self.llm else 0.0
        structure_bonus = 0.1 if structure_preserved else -0.2
        
        confidence_score = min(1.0, base_confidence + spacy_bonus + llm_bonus + structure_bonus)
        
        # Step 6: Finalize result
        processing_time = (datetime.now() - start_time).total_seconds()
        processing_steps.append(f"🎉 Templatization completed in {processing_time:.3f}s")
        
        return TemplatizationResult(
            original_problem=problem,
            templated_problem=templated_problem,
            legend=legend,
            entities_found=entities_found,
            confidence_score=confidence_score,
            processing_steps=processing_steps,
            metadata={
                'processing_time': processing_time,
                'methods_used': {
                    'spacy': SPACY_AVAILABLE,
                    'patterns': True,
                    'llm': self.llm is not None
                },
                'replacements_count': len(replacements),
                'structure_preserved': structure_preserved
            }
        )
    
    def restore_from_template(self, templated_solution: str, legend: Dict[str, str]) -> str:
        """
        Restore original entity names in a solution using the legend.
        
        Args:
            templated_solution (str): Solution with placeholders
            legend (Dict[str, str]): Mapping from placeholders to original names
            
        Returns:
            str: Solution with original entity names restored
        """
        restored_solution = templated_solution
        
        # Sort by placeholder length (descending) to avoid partial replacements
        sorted_legend = sorted(legend.items(), key=lambda x: len(x[0]), reverse=True)
        
        for placeholder, original_entity in sorted_legend:
            # Use word boundaries to ensure complete replacement
            pattern = r'\b' + re.escape(placeholder) + r'\b'
            restored_solution = re.sub(pattern, original_entity, restored_solution)
        
        return restored_solution
    
    def batch_templatize(self, problems: List[str]) -> List[TemplatizationResult]:
        """
        Templatize multiple problems efficiently.
        
        Args:
            problems (List[str]): List of word problems to templatize
            
        Returns:
            List[TemplatizationResult]: Results for each problem
        """
        results = []
        
        for i, problem in enumerate(problems):
            print(f"🔄 Processing problem {i+1}/{len(problems)}")
            result = self.templatize_problem(problem)
            results.append(result)
            
        return results
    
    def get_statistics(self, results: List[TemplatizationResult]) -> Dict[str, Any]:
        """Generate statistics from multiple templatization results."""
        if not results:
            return {}
        
        total_entities = sum(sum(len(v) for v in r.entities_found.values()) for r in results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        avg_processing_time = sum(r.metadata['processing_time'] for r in results) / len(results)
        
        entity_breakdown = {}
        for result in results:
            for category, entities in result.entities_found.items():
                if category not in entity_breakdown:
                    entity_breakdown[category] = 0
                entity_breakdown[category] += len(entities)
        
        return {
            'total_problems': len(results),
            'total_entities_found': total_entities,
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time,
            'entity_breakdown': entity_breakdown,
            'success_rate': len([r for r in results if r.confidence_score > 0.7]) / len(results)
        }

# Convenience functions for easy usage
def templatize_word_problem(problem: str, api_key: str = None) -> TemplatizationResult:
    """Quick function to templatize a single word problem."""
    templatizer = WordProblemTemplatizer(api_key=api_key)
    return templatizer.templatize_problem(problem)

def create_templatizer(api_key: str = None, model: str = "gpt-3.5-turbo") -> WordProblemTemplatizer:
    """Create a new templatizer instance."""
    return WordProblemTemplatizer(api_key=api_key, model=model)

# Example usage and testing
if __name__ == "__main__":
    print("🔄 Word Problem Templatization Engine")
    print("=" * 50)
    
    # Test problems with proper nouns
    test_problems = [
        "When the water is cold Ray swims a mile in 16 minutes. When the water is warm Ray swims a mile in 2 minutes more than twice as long. How much longer does Ray take to swim 3 miles on a hot day than a cold day?",
        
        "Sarah bought 3 books from Amazon for $15 each. She also bought 2 notebooks from Target for $4 each. How much did Sarah spend in total?",
        
        "At McDonald's, Mike ordered 2 Big Macs for $6.50 each and 1 large fries for $3.25. If he pays with a $20 bill, how much change will Mike receive?",
        
        "The distance from New York to Boston is 215 miles. If Tom drives at an average speed of 65 mph, how long will it take Tom to drive from New York to Boston?",
        
        "Lisa works at Google and earns $85,000 per year. If she gets a 12% raise, what will be Lisa's new annual salary?"
    ]
    
    try:
        # Create templatizer
        templatizer = create_templatizer()
        
        print("\n🧪 Testing templatization on sample problems...")
        print("-" * 60)
        
        results = []
        for i, problem in enumerate(test_problems, 1):
            print(f"\n📝 Test Problem {i}:")
            print(f"Original: {problem}")
            
            result = templatizer.templatize_problem(problem)
            results.append(result)
            
            print(f"Template: {result.templated_problem}")
            print(f"Legend: {result.legend}")
            print(f"Confidence: {result.confidence_score:.2f}")
            print(f"Entities: {result.entities_found}")
            
            # Test restoration
            if result.legend:
                sample_solution = f"The answer involves {list(result.legend.keys())[0]} and is 42."
                restored = templatizer.restore_from_template(sample_solution, result.legend)
                print(f"Restoration test: '{sample_solution}' → '{restored}'")
            
            print("-" * 40)
        
        # Generate statistics
        print("\n📊 TEMPLATIZATION STATISTICS")
        print("=" * 40)
        stats = templatizer.get_statistics(results)
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 To enable full functionality:")
        print("   - Install spaCy: pip install spacy")
        print("   - Download model: python -m spacy download en_core_web_sm")
        print("   - Install LangChain: pip install langchain langchain-openai")
        print("   - Set OpenAI API key: export OPENAI_API_KEY=your_key")

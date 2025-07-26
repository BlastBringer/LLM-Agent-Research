import openai
import os
from dotenv import load_dotenv

load_dotenv()

class ProblemClassifier:
    """
    Classifies math problems into specific categories.
    """
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_base = "https://openrouter.ai/api/v1"
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        print("🔍 Problem Classifier initialized.")
    
    def classify(self, problem: str) -> str:
        """
        Classifies the type of math problem.
        
        Args:
            problem: The math problem text to classify.
            
        Returns:
            A string representing the problem category.
        """
        print(f"🔍 Classifying problem: {problem[:50]}...")
        
        prompt = f"""
        Classify this math problem into exactly one of these categories:
        - system_of_linear_equations
        - simple_arithmetic
        - calculus
        - matrix_algebra
        - logic_puzzle
        - percentage_word_problem
        - geometry
        - other
        
        Problem: "{problem}"
        
        Return only the classification category, nothing else.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            classification = response.choices[0].message.content.strip()
            print(f"✅ Classification: {classification}")
            return classification
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return "other"

# Example usage
if __name__ == "__main__":
    classifier = ProblemClassifier()
    
    # Test problems
    test_problems = [
        "A company sells notebooks and pens. Each notebook costs ₹50 and each pen costs ₹20. On a certain day, the company sold a total of 120 items and made ₹3,800 in revenue. How many notebooks were sold?",
        "Calculate 25 * 4 + 18 / 3",
        "Find the area of a circle with radius 5 cm"
    ]
    
    for problem in test_problems:
        result = classifier.classify(problem)
        print(f"Problem: {problem[:50]}...")
        print(f"Classification: {result}\n")
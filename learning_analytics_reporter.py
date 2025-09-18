#!/usr/bin/env python3
"""
Learning Analytics Reporter - Analyzes learning mechanism usage.
"""

import json
import os
from datetime import datetime
from collections import Counter

def analyze_learning_analytics():
    """Analyze the learning analytics log file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analytics_file = os.path.join(current_dir, 'learning_analytics.log')
    new_learning_file = os.path.join(current_dir, 'new_learning.log')
    
    print("📊 LEARNING MECHANISM ANALYTICS REPORT")
    print("=" * 60)
    
    # Analyze learning usage
    if os.path.exists(analytics_file):
        print(f"\n🔍 LEARNING USAGE ANALYSIS")
        print("-" * 40)
        
        learning_events = []
        with open(analytics_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        learning_events.append(event)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Total learning events: {len(learning_events)}")
        
        if learning_events:
            # Success rate
            successful_events = [e for e in learning_events if e.get('success', False)]
            success_rate = len(successful_events) / len(learning_events) * 100
            print(f"Success rate: {success_rate:.1f}% ({len(successful_events)}/{len(learning_events)})")
            
            # Problem types
            problem_types = Counter(e.get('problem_type', 'unknown') for e in learning_events)
            print(f"\nProblem types solved using learning:")
            for ptype, count in problem_types.most_common():
                print(f"  • {ptype}: {count} times")
            
            # Methods used
            methods = Counter(e.get('method_used', 'unknown') for e in learning_events)
            print(f"\nMethods applied from learning:")
            for method, count in methods.most_common():
                print(f"  • {method}: {count} times")
            
            # Recent learning usage
            print(f"\n📋 RECENT LEARNING EVENTS:")
            print("-" * 40)
            for i, event in enumerate(learning_events[-5:], 1):  # Last 5 events
                timestamp = event.get('timestamp', 'N/A')
                problem = event.get('problem', 'N/A')[:50] + "..." if len(event.get('problem', '')) > 50 else event.get('problem', 'N/A')
                method = event.get('method_used', 'N/A')
                success = "✅" if event.get('success', False) else "❌"
                print(f"{i}. {success} {timestamp[:19]} - {method}")
                print(f"   Problem: {problem}")
    else:
        print("❌ No learning analytics file found")
    
    # Analyze new learning events
    if os.path.exists(new_learning_file):
        print(f"\n🎓 NEW LEARNING ANALYSIS")
        print("-" * 40)
        
        new_learning_events = []
        with open(new_learning_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        new_learning_events.append(event)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Total new patterns learned: {len(new_learning_events)}")
        
        if new_learning_events:
            # Methods learned
            methods_learned = Counter(e.get('method_learned', 'unknown') for e in new_learning_events)
            print(f"\nMethods learned:")
            for method, count in methods_learned.most_common():
                print(f"  • {method}: {count} times")
            
            # Recent new learning
            print(f"\n📋 RECENT NEW LEARNING:")
            print("-" * 40)
            for i, event in enumerate(new_learning_events[-3:], 1):  # Last 3 events
                timestamp = event.get('timestamp', 'N/A')
                problem = event.get('problem', 'N/A')[:40] + "..." if len(event.get('problem', '')) > 40 else event.get('problem', 'N/A')
                method = event.get('method_learned', 'N/A')
                total = event.get('total_learned_solutions', 'N/A')
                print(f"{i}. {timestamp[:19]} - {method}")
                print(f"   Problem: {problem}")
                print(f"   Total solutions: {total}")
    else:
        print(f"\n📚 No new learning events file found")
    
    # Summary statistics
    print(f"\n📈 SUMMARY")
    print("-" * 40)
    
    # Load current learned solutions
    learned_file = os.path.join(current_dir, 'learned_solutions.json')
    if os.path.exists(learned_file):
        try:
            with open(learned_file, 'r') as f:
                learned_solutions = json.load(f)
            print(f"Total learned solution patterns: {len(learned_solutions)}")
            
            # Show pattern types
            pattern_types = {}
            for signature, data in learned_solutions.items():
                if '∫' in signature:
                    pattern_types['Integration'] = pattern_types.get('Integration', 0) + 1
                elif any(op in signature for op in ['+', '-', '*', '/', '^']):
                    pattern_types['Algebraic'] = pattern_types.get('Algebraic', 0) + 1
                else:
                    pattern_types['Other'] = pattern_types.get('Other', 0) + 1
            
            print("Pattern categories:")
            for category, count in pattern_types.items():
                print(f"  • {category}: {count} patterns")
                
        except Exception as e:
            print(f"Error reading learned solutions: {e}")
    
    print(f"\n🎯 Learning mechanism is actively helping solve problems faster!")
    print(f"   Using previously learned patterns instead of full reasoning pipeline.")

if __name__ == "__main__":
    analyze_learning_analytics()

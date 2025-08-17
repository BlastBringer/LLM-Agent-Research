#!/usr/bin/env python3
"""
🎯 REINFORCEMENT LEARNING POLICY UPDATER FOR AGENT SYSTEM
========================================================

Implements RL-based policy updates (PPO, GPO) with reward scoring
to improve agent performance over time based on solution accuracy.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyType(Enum):
    """Types of RL policies supported."""
    PPO = "Proximal Policy Optimization"
    GPO = "Generalized Policy Optimization"
    BASIC = "Basic Reward-based Update"

@dataclass
class AgentPerformanceMetric:
    """Performance metrics for an agent."""
    agent_name: str
    task_type: str
    success_rate: float
    average_confidence: float
    execution_time: float
    accuracy_score: float
    reward_score: float

@dataclass
class PolicyUpdateResult:
    """Result of a policy update operation."""
    policy_type: str
    update_applied: bool
    performance_improvement: float
    new_parameters: Dict[str, Any]
    confidence_threshold: float

class RLRewardScorer:
    """Calculates reward scores for agent actions and results."""
    
    def __init__(self):
        self.reward_weights = {
            'accuracy': 0.4,        # 40% weight for correctness
            'efficiency': 0.2,      # 20% weight for speed
            'confidence': 0.2,      # 20% weight for confidence
            'completeness': 0.2     # 20% weight for completeness
        }
        self.baseline_performance = {}
    
    def calculate_reward(self, 
                        agent_result: Dict[str, Any], 
                        expected_result: Optional[Dict[str, Any]] = None,
                        execution_time: float = 0.0) -> float:
        """
        Calculate reward score for an agent's performance.
        
        Args:
            agent_result: Result from agent execution
            expected_result: Expected/ground truth result (if available)
            execution_time: Time taken for execution
            
        Returns:
            Reward score between 0.0 and 1.0
        """
        try:
            rewards = {}
            
            # Accuracy reward
            if expected_result and 'result' in agent_result:
                accuracy = self._calculate_accuracy(agent_result['result'], expected_result['result'])
                rewards['accuracy'] = accuracy
            else:
                # Use success flag if no ground truth
                rewards['accuracy'] = 1.0 if agent_result.get('success', False) else 0.0
            
            # Efficiency reward (inverse of execution time, normalized)
            max_time = 30.0  # Maximum acceptable time in seconds
            efficiency = max(0.0, 1.0 - (execution_time / max_time))
            rewards['efficiency'] = min(1.0, efficiency)
            
            # Confidence reward
            confidence = agent_result.get('confidence', 0.5)
            rewards['confidence'] = confidence
            
            # Completeness reward (based on presence of required fields)
            required_fields = ['success', 'result', 'method']
            completeness = sum(1 for field in required_fields if field in agent_result) / len(required_fields)
            rewards['completeness'] = completeness
            
            # Calculate weighted total reward
            total_reward = sum(
                self.reward_weights[component] * reward
                for component, reward in rewards.items()
            )
            
            logger.debug(f"Reward calculation: {rewards} -> {total_reward:.3f}")
            return total_reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _calculate_accuracy(self, agent_result: str, expected_result: str) -> float:
        """Calculate accuracy score between agent result and expected result."""
        try:
            # Simple string similarity for now
            # In production, would use mathematical equivalence checking
            agent_clean = str(agent_result).lower().strip()
            expected_clean = str(expected_result).lower().strip()
            
            if agent_clean == expected_clean:
                return 1.0
            
            # Calculate similarity score
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, agent_clean, expected_clean).ratio()
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.0
    
    def update_baseline(self, agent_name: str, performance_metrics: AgentPerformanceMetric):
        """Update baseline performance for an agent."""
        self.baseline_performance[agent_name] = performance_metrics

class PPOPolicyUpdater:
    """Proximal Policy Optimization for agent policies."""
    
    def __init__(self, learning_rate: float = 0.001, clip_epsilon: float = 0.2):
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.policy_parameters = {}
        self.value_function = {}
    
    def update_policy(self, 
                     agent_name: str,
                     state: Dict[str, Any],
                     action: Dict[str, Any],
                     reward: float,
                     next_state: Dict[str, Any]) -> PolicyUpdateResult:
        """
        Update agent policy using PPO algorithm.
        
        Args:
            agent_name: Name of the agent
            state: Current state representation
            action: Action taken by agent
            reward: Reward received
            next_state: Resulting state
            
        Returns:
            Policy update result
        """
        try:
            # Initialize policy parameters if not exist
            if agent_name not in self.policy_parameters:
                self.policy_parameters[agent_name] = {
                    'weights': np.random.normal(0, 0.1, 10),  # Example parameter vector
                    'bias': 0.0,
                    'confidence_threshold': 0.7
                }
            
            current_params = self.policy_parameters[agent_name]
            
            # Calculate advantage (simplified)
            advantage = reward - self._estimate_value(state)
            
            # PPO policy update (simplified implementation)
            if advantage > 0:
                # Positive advantage - encourage this action
                update_factor = min(1 + advantage, 1 + self.clip_epsilon)
            else:
                # Negative advantage - discourage this action
                update_factor = max(1 + advantage, 1 - self.clip_epsilon)
            
            # Update parameters
            new_params = current_params.copy()
            new_params['confidence_threshold'] *= update_factor
            new_params['confidence_threshold'] = np.clip(new_params['confidence_threshold'], 0.1, 0.9)
            
            # Apply learning rate
            parameter_change = (new_params['confidence_threshold'] - current_params['confidence_threshold']) * self.learning_rate
            new_params['confidence_threshold'] = current_params['confidence_threshold'] + parameter_change
            
            # Update stored parameters
            self.policy_parameters[agent_name] = new_params
            
            performance_improvement = reward - 0.5  # Baseline assumption
            
            return PolicyUpdateResult(
                policy_type="PPO",
                update_applied=True,
                performance_improvement=performance_improvement,
                new_parameters=new_params,
                confidence_threshold=new_params['confidence_threshold']
            )
            
        except Exception as e:
            logger.error(f"PPO policy update error: {e}")
            return PolicyUpdateResult(
                policy_type="PPO",
                update_applied=False,
                performance_improvement=0.0,
                new_parameters={},
                confidence_threshold=0.7
            )
    
    def _estimate_value(self, state: Dict[str, Any]) -> float:
        """Estimate value function for current state."""
        # Simplified value estimation
        complexity = len(str(state.get('problem', '')))
        return 0.5 + (complexity / 1000.0)  # Normalize complexity to value

class GPOPolicyUpdater:
    """Generalized Policy Optimization for agent policies."""
    
    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self.policy_history = {}
        self.performance_history = {}
    
    def update_policy(self,
                     agent_name: str,
                     performance_metrics: AgentPerformanceMetric,
                     context: Dict[str, Any]) -> PolicyUpdateResult:
        """
        Update agent policy using GPO algorithm.
        
        Args:
            agent_name: Name of the agent
            performance_metrics: Current performance metrics
            context: Additional context for policy update
            
        Returns:
            Policy update result
        """
        try:
            # Initialize history if not exist
            if agent_name not in self.performance_history:
                self.performance_history[agent_name] = []
            
            # Add current performance to history
            self.performance_history[agent_name].append(performance_metrics)
            
            # Calculate performance trend
            recent_performances = self.performance_history[agent_name][-5:]  # Last 5 performances
            if len(recent_performances) > 1:
                trend = np.mean([p.reward_score for p in recent_performances[-3:]]) - \
                       np.mean([p.reward_score for p in recent_performances[:2]])
            else:
                trend = 0.0
            
            # Update policy based on trend
            new_parameters = {
                'exploration_bonus': self.exploration_rate * (1 + trend),
                'confidence_multiplier': 1.0 + (trend * 0.1),
                'tool_selection_bias': trend * 0.05
            }
            
            performance_improvement = trend
            
            return PolicyUpdateResult(
                policy_type="GPO",
                update_applied=True,
                performance_improvement=performance_improvement,
                new_parameters=new_parameters,
                confidence_threshold=0.7 + (trend * 0.1)
            )
            
        except Exception as e:
            logger.error(f"GPO policy update error: {e}")
            return PolicyUpdateResult(
                policy_type="GPO",
                update_applied=False,
                performance_improvement=0.0,
                new_parameters={},
                confidence_threshold=0.7
            )

class RLPolicyManager:
    """Main manager for RL-based policy updates."""
    
    def __init__(self, policy_type: PolicyType = PolicyType.PPO):
        self.policy_type = policy_type
        self.reward_scorer = RLRewardScorer()
        self.ppo_updater = PPOPolicyUpdater()
        self.gpo_updater = GPOPolicyUpdater()
        self.agent_metrics = {}
        self.update_history = []
    
    def process_agent_result(self,
                           agent_name: str,
                           task_type: str,
                           agent_result: Dict[str, Any],
                           execution_time: float,
                           expected_result: Optional[Dict[str, Any]] = None) -> PolicyUpdateResult:
        """
        Process agent result and update policy accordingly.
        
        Args:
            agent_name: Name of the agent
            task_type: Type of task executed
            agent_result: Result from agent
            execution_time: Time taken for execution
            expected_result: Expected result (if available)
            
        Returns:
            Policy update result
        """
        try:
            # Calculate reward score
            reward_score = self.reward_scorer.calculate_reward(
                agent_result, expected_result, execution_time
            )
            
            # Create performance metrics
            metrics = AgentPerformanceMetric(
                agent_name=agent_name,
                task_type=task_type,
                success_rate=1.0 if agent_result.get('success', False) else 0.0,
                average_confidence=agent_result.get('confidence', 0.5),
                execution_time=execution_time,
                accuracy_score=reward_score,
                reward_score=reward_score
            )
            
            # Store metrics
            if agent_name not in self.agent_metrics:
                self.agent_metrics[agent_name] = []
            self.agent_metrics[agent_name].append(metrics)
            
            # Update policy based on selected algorithm
            if self.policy_type == PolicyType.PPO:
                state = {'task_type': task_type, 'problem': agent_result.get('original_problem', '')}
                action = {'method': agent_result.get('method', 'unknown')}
                next_state = {'result': agent_result.get('result', '')}
                
                update_result = self.ppo_updater.update_policy(
                    agent_name, state, action, reward_score, next_state
                )
                
            elif self.policy_type == PolicyType.GPO:
                context = {'task_type': task_type, 'execution_time': execution_time}
                update_result = self.gpo_updater.update_policy(agent_name, metrics, context)
                
            else:  # Basic update
                update_result = self._basic_policy_update(agent_name, metrics)
            
            # Store update history
            self.update_history.append({
                'timestamp': str(np.datetime64('now')),
                'agent_name': agent_name,
                'reward_score': reward_score,
                'policy_update': update_result
            })
            
            logger.info(f"Policy updated for {agent_name}: reward={reward_score:.3f}, "
                       f"improvement={update_result.performance_improvement:.3f}")
            
            return update_result
            
        except Exception as e:
            logger.error(f"Error processing agent result: {e}")
            return PolicyUpdateResult(
                policy_type="ERROR",
                update_applied=False,
                performance_improvement=0.0,
                new_parameters={},
                confidence_threshold=0.7
            )
    
    def _basic_policy_update(self, agent_name: str, metrics: AgentPerformanceMetric) -> PolicyUpdateResult:
        """Basic policy update based on reward score."""
        # Simple threshold-based update
        if metrics.reward_score > 0.8:
            confidence_boost = 0.1
        elif metrics.reward_score < 0.3:
            confidence_boost = -0.1
        else:
            confidence_boost = 0.0
        
        new_parameters = {
            'confidence_adjustment': confidence_boost,
            'performance_bonus': metrics.reward_score - 0.5
        }
        
        return PolicyUpdateResult(
            policy_type="BASIC",
            update_applied=abs(confidence_boost) > 0,
            performance_improvement=confidence_boost,
            new_parameters=new_parameters,
            confidence_threshold=0.7 + confidence_boost
        )
    
    def get_agent_performance_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get performance summary for an agent."""
        if agent_name not in self.agent_metrics:
            return {"error": f"No metrics found for agent {agent_name}"}
        
        metrics_list = self.agent_metrics[agent_name]
        if not metrics_list:
            return {"error": f"No metrics available for agent {agent_name}"}
        
        return {
            "agent_name": agent_name,
            "total_tasks": len(metrics_list),
            "average_success_rate": np.mean([m.success_rate for m in metrics_list]),
            "average_confidence": np.mean([m.average_confidence for m in metrics_list]),
            "average_execution_time": np.mean([m.execution_time for m in metrics_list]),
            "average_accuracy": np.mean([m.accuracy_score for m in metrics_list]),
            "average_reward": np.mean([m.reward_score for m in metrics_list]),
            "performance_trend": self._calculate_trend([m.reward_score for m in metrics_list])
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate performance trend from score history."""
        if len(scores) < 3:
            return "insufficient_data"
        
        recent_avg = np.mean(scores[-3:])
        earlier_avg = np.mean(scores[:-3])
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"

# Global policy manager instance
policy_manager = RLPolicyManager(PolicyType.PPO)

def update_agent_policy(agent_name: str,
                       task_type: str,
                       agent_result: Dict[str, Any],
                       execution_time: float = 0.0,
                       expected_result: Optional[Dict[str, Any]] = None) -> PolicyUpdateResult:
    """
    Convenience function to update agent policy.
    
    Args:
        agent_name: Name of the agent
        task_type: Type of task executed
        agent_result: Result from agent
        execution_time: Time taken for execution
        expected_result: Expected result (if available)
        
    Returns:
        Policy update result
    """
    return policy_manager.process_agent_result(
        agent_name, task_type, agent_result, execution_time, expected_result
    )

def get_policy_manager() -> RLPolicyManager:
    """Get the global policy manager instance."""
    return policy_manager

if __name__ == "__main__":
    print("🎯 RL Policy Manager for Agent System")
    print("=" * 40)
    
    # Test the policy manager
    manager = get_policy_manager()
    
    # Simulate agent result
    test_result = {
        'success': True,
        'result': 'x^2 + 2*x - 3 = 0 solutions: [-3, 1]',
        'method': 'symbolic_solver',
        'confidence': 0.9
    }
    
    # Test policy update
    update_result = update_agent_policy(
        'test_agent', 'quadratic_equation', test_result, 2.5
    )
    
    print(f"Policy Update Result: {update_result}")
    
    # Get performance summary
    summary = manager.get_agent_performance_summary('test_agent')
    print(f"Performance Summary: {summary}")
    
    print("✅ RL Policy Manager testing completed!")

"""
🧠 SOLVER MODULE
================

This module contains the complete solving architecture:
- Apprentice: The student model (Llama 3 8B) that learns over time
- Verifier: Deterministic equation solver for ground truth
- Oracle: The teacher model (to be implemented next)
- SolverAgent: Main orchestrator

The learning loop:
1. Apprentice attempts to solve
2. Verifier checks the answer
3. If wrong, Oracle provides correct solution
4. Oracle's solution is saved for fine-tuning
"""

from .apprentice import ApprenticeModel, ApprenticeSolution
from .verifier import MathVerifier, VerificationResult
from .oracle import OracleModel, OracleSolution
from .solver_agent import SolverAgent

__all__ = [
    'ApprenticeModel',
    'ApprenticeSolution',
    'MathVerifier',
    'VerificationResult',
    'OracleModel',
    'OracleSolution',
    'SolverAgent'
]

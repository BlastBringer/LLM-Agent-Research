# 🧠 Solver Module

The **Solver** is the brain of the learning system. It coordinates an **Apprentice Model** (student) and a **Verifier** (judge) to solve math problems, with plans for an **Oracle Model** (teacher) for continuous learning.

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SOLVER AGENT                         │
│                                                          │
│  Problem → Apprentice → Verifier                        │
│                           │                              │
│                      Is Correct?                         │
│                      ┌────┴────┐                         │
│                    Yes        No                         │
│                     │          │                         │
│                  Success   Oracle → Save for Training    │
│                              (TODO)                      │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Components

### 1. **Apprentice Model** (`apprentice.py`)
- **Role:** The "student" that learns over time
- **Current:** Llama 3 8B via OpenRouter API
- **Future:** Local Llama 3 8B fine-tuned with QLoRA
- **Output:** Step-by-step reasoning + final answer
- **Format:** Structured JSON for consistency

### 2. **Verifier** (`verifier.py`)
- **Role:** The "judge" that provides 100% reliable ground truth
- **Technology:** SymPy (symbolic math) + NumPy (numerical)
- **Method:** Deterministic equation solving (no AI)
- **Output:** `is_correct`, `correct_answer`, `difference`

### 3. **Oracle Model** (`oracle.py`) - *Coming Next*
- **Role:** The "teacher" that provides gold-standard solutions
- **Model:** Claude 3.5 Sonnet or GPT-4o
- **Framework:** ReAct (Reason + Act) with tool calling
- **Tools:** Python interpreter for precise calculations
- **Output:** Detailed reasoning trace for training data

### 4. **Solver Agent** (`solver_agent.py`)
- **Role:** The orchestrator that ties everything together
- **Flow:**
  1. Apprentice attempts to solve
  2. Verifier checks the answer
  3. If wrong → Oracle provides correct solution (TODO)
  4. Oracle's solution saved for fine-tuning
- **Tracks:** Statistics (accuracy, oracle usage rate, etc.)

## 🚀 Usage

### Quick Test

```bash
# Test individual components
python Solver/verifier.py          # Test ground truth computation
python Solver/apprentice.py        # Test apprentice solving
python Solver/solver_agent.py      # Test complete pipeline

# Comprehensive test suite
python scripts/test/test_solver.py
```

### Integration with Main Pipeline

```python
from Solver import SolverAgent

# After preprocessing (templatization, parsing, extraction, standardization)
solver = SolverAgent()
result = solver.solve(problem_data, verbose=True)

print(f"Answer: {result.final_answer}")
print(f"Correct: {result.is_correct}")
print(f"Solver: {result.solver_used}")
```

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Verifier | ✅ Complete | SymPy + NumPy, tested |
| Apprentice | ✅ Working | Llama 3 8B via API |
| Solver Agent | ✅ Working | Orchestration ready |
| Oracle | 🚧 Next | Will use Claude/GPT-4o with ReAct |
| Local Llama | 📅 Phase 2 | For QLoRA fine-tuning |
| Training Loop | 📅 Phase 2 | After Oracle implementation |

## 🎓 Learning Loop (Future)

Once Oracle is implemented:

1. **Collect Training Data:**
   - Every time Apprentice fails, Oracle solves it
   - Solution saved to `solver_training_data.jsonl`
   - Format: `{"problem": "...", "golden_solution": "..."}`

2. **Fine-Tune Apprentice:**
   - Once we have 500-1000 examples
   - Use QLoRA to fine-tune local Llama 3 8B
   - Batch size: 4, LoRA rank: 16, learning rate: 2e-4

3. **Measure Improvement:**
   - Track apprentice accuracy over time
   - Track oracle usage rate (should decrease)
   - Goal: 80%+ apprentice accuracy

4. **Iterate:**
   - Continuous collection of new examples
   - Periodic fine-tuning (every 1000 examples)
   - Model gets smarter over time

## 🔬 Technical Details

### Verifier Solving Strategy

1. **SymPy (Primary):** Symbolic equation solving
2. **Substitution (Fallback):** Direct variable substitution
3. **Safe Eval (Last Resort):** Sandboxed evaluation

### Apprentice Prompting

The apprentice uses a carefully crafted prompt that:
- Provides problem context
- Lists known variables and values
- Shows relevant equations
- Requests step-by-step reasoning
- Enforces JSON output format

### Error Handling

- Apprentice fails to answer → Log for Oracle
- Verifier fails to solve → Log for human review
- Oracle fails → Log for human intervention
- All failures tracked in `solver_failures.jsonl`

## 📝 Configuration

Set in `.env`:

```bash
# Apprentice (student model)
APPRENTICE_MODEL="meta-llama/llama-3-8b-instruct"

# Oracle (teacher model) - for next phase
ORACLE_MODEL="anthropic/claude-3.5-sonnet"

# OpenRouter API
OPENAI_API_KEY="sk-or-v1-..."
OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

## 🧪 Testing

The test suite (`scripts/test/test_solver.py`) validates:

1. **Verifier Accuracy:**
   - Simple arithmetic
   - Multiplication
   - Wrong answer detection
   - Complex expressions

2. **Apprentice Response:**
   - Can produce structured output
   - Reasoning quality
   - Answer extraction

3. **Solver Integration:**
   - Complete pipeline flow
   - Statistics tracking
   - Error handling

## 📈 Next Steps

1. **Implement Oracle:**
   - Create `oracle.py` with Claude 3.5 Sonnet
   - Add ReAct framework with Python tool
   - Integrate into `solver_agent.py`

2. **Set Up Local Llama:**
   - Download Llama 3 8B weights
   - Set up QLoRA fine-tuning pipeline
   - Create training scripts

3. **Build Training Loop:**
   - Automated data collection
   - Periodic fine-tuning
   - Performance tracking dashboard

4. **Advanced Features:**
   - Multi-step problem decomposition
   - Self-correction mechanisms
   - Confidence-based oracle triggering

## 🤝 Contributing

When adding features:
- Keep components modular and independent
- Add comprehensive logging
- Write tests for new functionality
- Document prompts and design decisions

---

**Built with:** Python, LangChain, SymPy, NumPy, Llama 3  
**Research Focus:** Self-improving mathematical reasoning through apprentice-oracle learning

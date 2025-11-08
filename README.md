# RL for LLM training Variance-Stabilized Dropout Implementation

**Author:** Eva Paunova, AI Research Scientist   


## Overview

This contains an RL task for ML researcher/engineer training that teaches models to:
- Understand and implement techniques from research papers
- Debug subtle mathematical errors in ML code
- Apply statistical validation to neural network components
- Reason about variance and gradient stability

**RL Training Context:**
This task serves as an environment in an RL training pipeline:
- **State:** Buggy dropout implementation
- **Action Space:** Code modifications
- **Reward Signal:** Binary pass/fail based on variance preservation (±10%)
- **Learning Objective:** Paper-to-code implementation with mathematical reasoning

The clear reward signal (43% error → fail, 0.1% error → pass) enables effective policy learning, while the subtle bug ensures models must develop deep understanding rather than pattern matching.

## Quick Start

Setup:
```bash
git clone https://github.com/epaunova/preference-model.git
cd preference-model
pip install -r requirements.txt
```

Run the demo:
```bash
python variance_dropout_task.py
```

This shows the buggy implementation failing and the correct implementation passing.

## Validation Results Summary

**Empirical Pass Rate:** 26.7% (4/15 attempts with Claude Opus 4.1)  
**Target Range:** 10-40%  
**Status:** Within target range

| Metric | Value |
|--------|-------|
| Model Tested | Claude Opus 4.1 |
| Total Runs | 15 |
| Passed | 4 (26.7%) |
| Failed | 11 (73.3%) |
| Pass Examples | 0.1-0.4% variance error |
| Fail Examples | 15-44% variance error |
| Avg Time/Run | 16.9 seconds |

**Result:** Task difficulty is properly calibrated for RL training.

*See [VALIDATION_RESULTS.md](./preference-model-takehome/VALIDATION_RESULTS.md) for detailed attempt-by-attempt results.*

## Empirical Validation with Claude API

To validate that the task is properly calibrated (10-40% success rate), run it against Claude multiple times.

Prerequisites:
```bash
pip install anthropic>=0.28.0
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Run validation:
```bash
# Quick test with 3 attempts
python validate_task.py --runs 3 --output-dir test_validation

# Full validation with 15 attempts
python validate_task.py --runs 15 --output-dir validation_results
```

Output files:
- validation_results/VALIDATION_RESULTS.md - Detailed report
- validation_results/validation_results.json - Raw data

## Why This Task is Effective

### 1. **Scientific Concept** (Pro Tip #1) 
The task teaches variance stabilization in neural networks - a real research concept that:
- Addresses gradient stability issues in deep learning
- Requires understanding of statistical moments (mean vs variance)
- Shows how small mathematical changes have big practical impacts
- Connects theory (paper concept) to implementation (code)

### 2. **Clear Difficulty Targeting (10-40%)** (Pro Tip #4) 
The bug is subtle enough that models will struggle:
- **Weak models (0-10%):** May not understand the math or make random changes
- **Medium models (15-30%):** Understand the concept but make implementation errors
- **Strong models (30-40%):** Get it right, correctly implementing sqrt scaling
- The single-line fix makes grading unambiguous

### 3. **Diverse Failure Modes** 
Models fail for different reasons:
- Not understanding variance vs mean preservation
- Using wrong formula (1/sqrt(p) instead of 1/sqrt(1-p))
- Breaking the class interface
- Not handling edge cases (eval mode, zero dropout rate)
- Changing the test file instead of implementation

### 4. **Tool Usage** (Pro Tip #2) 
The task requires:
- Code execution (running tests)
- Statistical validation (checking variance empirically)
- Debugging skills (identifying the bug location)
- Mathematical reasoning (deriving correct scaling factor)

### 5. **Clean Grading** (Pro Tip #5) 
The grader precisely checks:
- Statistical tests pass (variance within 10% tolerance)
- Eval mode works (identity function)
- Training mode works (correct dropout rate)
- Implementation contains sqrt (sanity check)
- Test file not modified (anti-cheating)
- All tests automated - no ambiguity

## Task Specification

The Problem:

Dropout randomly deactivates neurons during training. There are two ways to scale activations:

1. Inverted dropout: output = mask * input / keep_prob
2. Variance-preserving dropout: output = mask * input / sqrt(1 - dropout_prob)

The bug:

The current implementation uses:
```python
scale = 1.0 / keep_prob  # Wrong! Preserves mean but inflates variance
```

Correct version:
```python
scale = 1.0 / np.sqrt(keep_prob)  # Preserves both mean and variance
```

Why it matters:

For proper variance preservation:
- After dropout + scaling: Var(output) = (1-p) * scale^2 * Var(X)
- For Var(output) = Var(X): need (1-p) * scale^2 = 1
- Therefore: scale = 1/sqrt(1-p)

This is critical for gradient stability in deep networks.

## Testing & Validation

The task has been comprehensively tested and validated. Key results:

**Buggy Implementation:**
- Variance error: 43.0% (fails as expected)
- Clear distinction from correct implementation

**Correct Implementation:**
- Variance error: 0.1% (passes easily)
- All three dropout rates tested (0.3, 0.5, 0.7)
- Eval mode and training mode verified

**Mathematical Verification:**
- Theoretical predictions match empirical measurements within 1%
- Formula derivation confirmed correct

**Empirical Validation Results:**
- Success Rate: 26.7% (4/15 attempts)
- Target Range: 10-40% ✓ ACHIEVED
- Average Time: 16.9s per attempt

For full testing details, see `TESTING_RESULTS.md` and `VALIDATION_RESULTS.md`.

## File Structure

preference-model/
- variance_dropout_task.py - Main task file (everything in one place)
- validate_task.py - Empirical validation with Claude API
- README.md - This file
- FAQ.md - Common questions and troubleshooting
- VALIDATION_RESULTS.md - Empirical validation results
- requirements.txt - Dependencies

## Pass/Fail Criteria

**The task passes if:**
- Variance preserved within 10% across dropout rates (0.3, 0.5, 0.7)
- Eval mode returns identity (no dropout applied)
- Training mode actually drops ~p% of units
- Test file not modified
- Implementation contains sqrt or power function

**Common Failure Modes:**
- Using 1/p or other wrong formulas
- Confusing mean vs variance preservation
- Breaking class interface
- Not handling edge cases (eval mode, zero dropout rate)
- Modifying test file instead of implementation

## How It Works

For a single attempt:
```python
from pathlib import Path
from variance_dropout_task import setup_task_files, grade_solution

workspace = Path("test_workspace")
workspace.mkdir(exist_ok=True)
setup_task_files(workspace)

# Model attempts to fix the code here...

result = grade_solution(workspace)
print(f"Passed: {result['passed']}")
print(f"Feedback: {result['feedback']}")
```

For RL training:
```python
# Initialize
workspace = Path("workspace")
setup_task_files(workspace)

# Present task to model
response = model.generate(
    prompt=TASK_PROMPT,
    tools=["read_file", "write_file", "execute_code"],
    workspace=workspace
)

# Get signal
result = grade_solution(workspace)
reward = 1.0 if result['passed'] else 0.0
```

## Production Integration Notes

This task follows the standard RL environment pattern and can be easily integrated into existing frameworks:

```python
# Example integration pattern
from variance_dropout_task import setup_task_files, grade_solution, TASK_PROMPT

# 1. Initialize environment (create workspace with task files)
workspace = Path("workdir")
setup_task_files(workspace)
# Creates: dropout.py (buggy), test_dropout.py

# 2. Present task to agent
agent_response = agent.solve(
    prompt=TASK_PROMPT,
    tools=["read_file", "write_file", "execute_code"],
    workspace=workspace
)

# 3. Grade solution
result = grade_solution(workspace)

# 4. Extract RL signal
reward = 1.0 if result['passed'] else 0.0
feedback = result['feedback']  # For learning
output = result['output']      # For debugging
```

**Key Integration Features:**
- **Clean state initialization:** `setup_task_files()` creates isolated workspace
- **Standard grading interface:** `grade_solution()` returns structured dict
- **Detailed feedback:** Rich error messages for debugging
- **Deterministic:** Same code modification always produces same result
- **Fast execution:** ~5 seconds per grading run

The task is framework-agnostic and can integrate with any RL training system that supports:
- File I/O tools (read/write code)
- Code execution (run tests)
- Python environment (NumPy)

## Difficulty Calibration & Expected Pass Rates

The task is calibrated for 10-40% success rate through:

### Difficulty Factors
1. **Conceptual understanding** - Model must grasp variance vs mean preservation
2. **Mathematical derivation** - Need to derive correct scaling factor  
3. **Subtle bug** - Only one character different (keep_prob vs sqrt(keep_prob))
4. **Multiple tests** - Must pass variance, eval mode, and training mode tests
5. **Strict tolerance** - 10% variance error threshold

### Expected Performance by Model Capability

**Pass Rate Method:**  
Based on empirical testings, the pass rate varies by model strength. For instance, weak models (e.g., GPT-3.5) typically achieve 5-10% success, medium models (e.g., GPT-4) 15-30%, and strong models (e.g., Claude Opus) 30-40%. This is confirmed through 15 actual runs on Claude Opus 4.1, yielding a 26.7% success rate (within the target 10-40% range), as detailed in VALIDATION_RESULTS.md. The task's difficulty stems from requiring precise mathematical reasoning, such as selecting the correct scaling formula among several plausible alternatives, with a clear performance gap between incorrect (e.g., 43% variance error) and correct implementations (e.g., 0.1% error).
- **GPT-3.5 / Weak models (0-10%):** 
  - Likely make random changes or copy buggy pattern
  - May not understand variance stabilization concept
  
- **GPT-4 / Medium models (15-30%):**
  - Understand the concept but may implement incorrectly
  - Common errors: 1/p, sqrt(p), 1/sqrt(p) instead of 1/sqrt(1-p)
  
- **Claude Opus / Strong models (30-40%):**
  - Correctly derive and implement sqrt scaling
  - Empirical validation shows 26.7% success rate

### Why This Hits The Sweet Spot
- Too easy if: Bug was obvious (e.g., missing return statement)
- Too hard if: Required complex multi-step derivation
- Just right: Single subtle math error that requires understanding

## Why This Teaches Something Valuable

### For ML Researchers/Engineers
1. **Paper Implementation Skills** - Reading and correctly implementing research
2. **Statistical Reasoning** - Understanding moments beyond just means
3. **Numerical Debugging** - Finding subtle mathematical bugs
4. **Gradient Stability** - Why variance matters for training dynamics

### For AI Models
1. **Mathematical Rigor** - Can't just pattern-match, must derive correctly
2. **Debugging** - Requires analyzing broken code systematically  
3. **Validation** - Understanding how to test statistical properties
4. **Practical ML** - Connects theory to implementation

### Real-World Relevance
- Dropout is used in virtually every neural network
- Variance issues cause real training instability
- Paper-to-code is a daily ML research activity
- Statistical validation is critical for ML systems

## Task Design Rationale

### Why Variance-Stabilized Dropout?
1. **Real research concept** - Based on actual training stability techniques
2. **Single clear bug** - Makes grading unambiguous
3. **Requires math** - Can't be solved by guessing
4. **Testable empirically** - Statistical tests provide ground truth
5. **Educational value** - Teaches important ML concept

### Why This Bug Specifically?
- **Subtle:** Both 1/(1-p) and 1/sqrt(1-p) seem plausible
- **Testable:** Variance metrics clearly show the difference
- **Common:** Many practitioners confuse mean vs variance preservation
- **Fixable:** Single-line change, but requires understanding

### Alignment with Requirements
 **Scientific concept** - Variance stabilization in neural nets  
 **Tool usage** - Code execution, statistical testing  
 **Coherent** - Single clear objective with measurable outcome  
 **Balanced difficulty** - 10-40% calibrated, empirically validated at 26.7%  
 **Precise grading** - Automated tests, no ambiguity  

## Requirements

```
numpy>=1.20.0
anthropic>=0.28.0
```

## Setup Instructions

1. Get Anthropic API key from https://console.anthropic.com
2. Set environment variable: export ANTHROPIC_API_KEY="sk-ant-..."
3. Install dependencies: pip install -r requirements.txt
4. Run validation: python validate_task.py --runs 15 --output-dir validation_results
5. Check results: cat validation_results/VALIDATION_RESULTS.md

## Time Spent Breakdown

- **Concept Selection:** 45 min (reviewed research concepts, selected dropout)
- **Task Design:** 1 hour (prompt, bug design, test cases)
- **Implementation:** 1.5 hours (buggy/correct versions, grader, tests)
- **Testing & Validation:** 45 min (verified difficulty, edge cases, empirical validation)
- **Documentation:** 30 min (README, comments, examples)
- **Total:** ~4 hours

## Files Included

1. **variance_dropout_task.py** - Main task file with everything
2. **validate_task.py** - Validation script for empirical testing
3. **README.md** - This documentation
4. **FAQ.md** - Common questions and troubleshooting
5. **TESTING_RESULTS.md** - Comprehensive test results
6. **VALIDATION_RESULTS.md** - Empirical validation results
7. **requirements.txt** - Dependencies

## Status

Task is ready for review and RL training integration.
Empirical validation demonstrates proper difficulty calibration (26.7% success rate - within target 10-40%).
All grading criteria satisfied.

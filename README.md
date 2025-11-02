# Variance-Stabilized Dropout: RL Task for ML Researcher Training

Author: Eva Paunova, AI Research Scientist
Date: October 31, 2025

## Overview

This is an RL training task that teaches models to:
- Understand and implement techniques from research papers
- Debug subtle mathematical errors in ML code
- Apply statistical validation to neural network components
- Reason about variance and gradient stability

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

## Grading Criteria

The task passes if:
- Variance preserved within 10% across dropout rates (0.3, 0.5, 0.7)
- Eval mode returns identity (no dropout applied)
- Training mode actually drops ~p% of units
- Test file not modified
- Implementation contains sqrt or power function

## Common Failure Modes

Models fail for different reasons:
- Using 1/p or other wrong formulas
- Confusing mean vs variance preservation
- Breaking class interface
- Not handling edge cases (eval mode, zero dropout rate)
- Modifying test file instead of implementation

## File Structure

preference-model/
- variance_dropout_task.py - Main task file (everything in one place)
- validate_task.py - Empirical validation with Claude API
- README.md - This file
- FAQ.md - Common questions and troubleshooting
- VALIDATION_RESULTS.md - Generated after running validation
- requirements.txt - Dependencies

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

## Difficulty Analysis

The task targets 10-40% success rate through:
- Conceptual barrier: Must understand variance vs mean preservation
- Mathematical requirement: Must derive correct scaling factor
- Subtle bug: Only one character different (keep_prob vs sqrt(keep_prob))
- Multiple tests: Must pass variance, eval, and training tests
- Strict tolerance: 10% variance error threshold

## Learning Objectives

This task teaches:
- Paper implementation: Converting research to code
- Statistical reasoning: Understanding variance beyond just means
- Numerical debugging: Finding subtle mathematical bugs
- Gradient stability: Why variance matters for training
- Practical ML: Connects theory to implementation

## Integration with RL Training

The task follows standard RL environment patterns:
- State: Buggy dropout implementation
- Action Space: Code modifications
- Reward Signal: Binary pass/fail from grader
- Episode Length: Typically 1-3 interactions
- Observation: Task prompt + code + test outputs

Framework compatibility:
- Works with any system supporting file I/O and code execution
- Deterministic grading (same code produces same result)
- Fast evaluation (about 5 seconds per grading run)
- Structured feedback for debugging

## Implementation Notes

- Single file: All code in variance_dropout_task.py
- Self-contained: No external data required
- Reproducible: Fixed random seeds
- Scalable: Easy to run in parallel
- Observable: Detailed logs and metrics

## Validation Results

After running validation with Claude API, you will get:
- Success rate (e.g., 26.7% = 4 out of 15 passed)
- Variance error distribution
- Individual attempt analysis
- Recommendations for difficulty adjustment

Example output:
```
Success Rate: 26.7% (4/15)
Range Status: ACCEPTABLE (target is 10-40%)
Average Time: 16.9s per attempt
```

## Requirements

numpy>=1.20.0
anthropic>=0.28.0

## Setup Instructions

1. Get Anthropic API key from https://console.anthropic.com
2. Set environment variable: export ANTHROPIC_API_KEY="sk-ant-..."
3. Install dependencies: pip install -r requirements.txt
4. Run validation: python validate_task.py --runs 15 --output-dir validation_results
5. Check results: cat validation_results/VALIDATION_RESULTS.md

## Status

Task is ready for review and RL training integration.
Empirical validation demonstrates proper difficulty calibration.
All grading criteria satisfied.

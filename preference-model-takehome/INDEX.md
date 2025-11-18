# File Navigation Guide

## Core Files

**Essential for the task:**
- **variance_dropout_task.py** - Complete task implementation (the actual submission)
- **requirements.txt** - Dependencies (numpy, anthropic)

## Documentation Files

**Getting started:**
- **README.md** - Full documentation with rationale and design decisions
- **QUICKSTART.md** - How to run and integrate (5 min read)

**For understanding the task:**
- **FAQ.md** - Common questions and answers
- **INDEX.md** - This file (navigation guide)

**For verification:**
- **CHECKLIST.md** - Verification against all requirements

## Validation & Testing

**Empirical validation:**
- **validate_task.py** - Script to test with Claude API (10+ runs)
- **VALIDATION_RESULTS.md** - Results and analysis (26.7% success rate)

**Detailed analysis:**
- **TESTING_RESULTS.md** - Comprehensive test results and validation
- **TECHNICAL_DESIGN.md** - Deep technical documentation

## Quick Navigation

### I want to...

**Run the task demo**
```bash
pip install numpy
python variance_dropout_task.py
```
See: README.md, QUICKSTART.md

**Understand the task**
Read: README.md (15 min)
Then: TECHNICAL_DESIGN.md (optional, 20 min)

**Integrate into RL training**
Read: QUICKSTART.md (5 min)
See: validate_task.py for API testing

**Verify all requirements are met**
Read: CHECKLIST.md

**See test results**
Read: TESTING_RESULTS.md

**Ask common questions**
Read: FAQ.md

**Understand design decisions**
Read: TECHNICAL_DESIGN.md

## File Summary

| File | Purpose | Size | Read Time |
|------|---------|------|-----------|
| variance_dropout_task.py | Task implementation | ~13 KB | - |
| validate_task.py | Claude API validation | ~4 KB | - |
| README.md | Full documentation | ~9 KB | 15 min |
| QUICKSTART.md | Integration guide | ~3 KB | 5 min |
| FAQ.md | Common questions | ~2 KB | 5 min |
| CHECKLIST.md | Requirements verification | ~4 KB | 10 min |
| TESTING_RESULTS.md | Test results | ~8 KB | 10 min |
| TECHNICAL_DESIGN.md | Technical deep-dive | ~15 KB | 20 min |
| VALIDATION_RESULTS.md | Validation results | ~4 KB | 5 min |
| requirements.txt | Dependencies | ~0.2 KB | 1 min |
| INDEX.md | This file | ~2 KB | 5 min |

**Total package:** ~65 KB (lightweight!)

## For Reviewers

### 5-Minute Review
1. Run: `python variance_dropout_task.py`
2. Check output shows both buggy (43% error) and correct (0.1% error)
3. Read: README.md summary

### 15-Minute Review
1. Read: README.md (full)
2. Skim: CHECKLIST.md
3. Check: variance_dropout_task.py structure

### Deep Review (30 min)
1. Read: README.md
2. Read: TECHNICAL_DESIGN.md
3. Review: variance_dropout_task.py code
4. Check: TESTING_RESULTS.md
5. Verify: CHECKLIST.md

## Key Concepts

**The Task:** Fix a subtle bug in dropout implementation

**The Bug:** Using `1/(1-p)` instead of `1/sqrt(1-p)` for scaling

**Why Hard:** Requires understanding variance vs mean preservation

**Difficulty:** Calibrated for 10-40% success rate

**Validation:** Empirical statistical tests

**Tool Usage:** NumPy for computation and testing

**Scientific Concept:** Neural network training stability

## Integration Example

```python
from pathlib import Path
from variance_dropout_task import TASK_PROMPT, setup_task_files, grade_solution

# 1. Create workspace
workspace = Path("model_workspace")
workspace.mkdir(exist_ok=True)

# 2. Set up task
setup_task_files(workspace)

# 3. Present to model
# Give model TASK_PROMPT and tools to edit/execute

# 4. Grade
result = grade_solution(workspace)
print(f"Passed: {result['passed']}")
```

See QUICKSTART.md for more details.

## Questions?

- **How do I run this?** → QUICKSTART.md
- **What does this task do?** → README.md
- **Why this difficulty?** → TECHNICAL_DESIGN.md
- **What are the results?** → VALIDATION_RESULTS.md
- **Is it production ready?** → Yes! See CHECKLIST.md

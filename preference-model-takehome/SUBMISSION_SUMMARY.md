# Quick Submission Summary

## Task: Variance-Stabilized Dropout Implementation

**One-line pitch:** Debug a subtle mathematical error in a dropout implementation based on a research paper concept.

### Why This Task Excels

| Requirement | How Task Satisfies It | Evidence |
|-------------|----------------------|----------|
| **Scientific Concept** | Variance stabilization in neural networks | Mathematical derivation in prompt, real research technique |
| **Tool Usage** | Code execution + statistical testing | Model must run tests, validate empirically |
| **Clear Grading** | Automated test suite | Binary pass/fail, variance within 10% tolerance |
| **10-40% Difficulty** | Subtle one-character bug | sqrt(keep_prob) vs keep_prob - easy to miss |
| **Teaches Value** | Paper→code + debugging + statistics | Core ML research skill |

### The Bug (Spoiler)

**Wrong:**
```python
scale = 1.0 / keep_prob  # Preserves mean, NOT variance
```

**Right:**
```python
scale = 1.0 / np.sqrt(keep_prob)  # Preserves variance ✓
```

### Test Results

```
BUGGY VERSION:
✗ Variance ratio: 1.43 (target: 1.0) - 43% error
✗ Tests FAILED

CORRECT VERSION:  
✓ Variance ratio: 1.001 (target: 1.0) - 0.1% error
✓ All tests PASSED
```

### Files Delivered

1. **variance_dropout_task.py** (300 lines)
   - Task prompt with paper explanation
   - Buggy implementation
   - Test suite
   - Grader
   - Correct solution (for validation)

2. **README.md** (comprehensive docs)
   - Task rationale
   - Difficulty calibration
   - Integration guide
   - Expected pass rates by model strength

3. **requirements.txt** (numpy only)

### 

Run `python variance_dropout_task.py` to see it in action!

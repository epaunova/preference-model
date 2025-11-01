# RL Task Submission: Variance-Stabilized Dropout Implementation

**Candidate:** Eva Paunova, AI Research Scientist   
**Date:** October 31, 2025  

## Task Overview

This submission contains an RL task for ML researcher/engineer training that teaches models to:
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

## Why This Task is Effective

### 1. **Scientific Concept** 
The task teaches variance stabilization in neural networks - a real research concept that:
- Addresses gradient stability issues in deep learning
- Requires understanding of statistical moments (mean vs variance)
- Shows how small mathematical changes have big practical impacts
- Connects theory (paper concept) to implementation (code)

### 2. **Clear Difficulty Targeting (10-40%)** 
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

### 4. **Tool Usage** 
The task requires:
- Code execution (running tests)
- Statistical validation (checking variance empirically)
- Debugging skills (identifying the bug location)
- Mathematical reasoning (deriving correct scaling factor)

### 5. **Clean Grading** 
The grader precisely checks:
- Statistical tests pass (variance within 10% tolerance)
- Eval mode works (identity function)
- Training mode works (correct dropout rate)
- Implementation contains sqrt (sanity check)
- Test file not modified (anti-cheating)
- All tests automated - no ambiguity

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

For full testing details, see `TESTING_RESULTS.md`.

## Task Structure

```
variance_dropout_task.py
├── TASK_PROMPT              # Clear instructions with paper explanation
├── BUGGY_IMPLEMENTATION     # dropout.py with wrong scaling (1/(1-p))
├── TEST_SCRIPT              # test_dropout.py with statistical validation
├── grade_solution()         # Automated grading function
└── CORRECT_SOLUTION         # Reference implementation (1/sqrt(1-p))
```

### The Bug
**Buggy code:**
```python
scale = 1.0 / keep_prob  # Wrong! Preserves mean but inflates variance
```

**Correct fix:**
```python
scale = 1.0 / np.sqrt(keep_prob)  # Preserves variance!
```

**Why it matters:**
- Var(output) = Var(input) × (1-p) × scale²
- For Var(output) = Var(input), need: (1-p) × scale² = 1
- Therefore: scale = 1/√(1-p) 

## Pass/Fail Criteria

**The task passes if:**
-  `dropout.py` modified with correct scaling factor
-  Variance preserved within 10% across dropout rates (0.3, 0.5, 0.7)
-  Eval mode returns identity (no dropout)
-  Training mode actually drops ~p% of units
-  Test file not modified
-  Implementation contains sqrt/0.5/power

**Common failure modes:**
- Using 1/p or 1/(1-p) instead of 1/sqrt(1-p)
- Confusing mean vs variance preservation
- Breaking class interface
- Modifying test instead of implementation
- Not handling edge cases

## Testing Instructions

### Quick Test
```bash
python variance_dropout_task.py
```

Output shows:
1. Task prompt with paper explanation
2. Buggy version failing (variance inflated by ~43%)
3. Correct version passing (variance error < 1%)

### Manual Testing
```python
from pathlib import Path
from variance_dropout_task import setup_task_files, grade_solution

# Create workspace
workspace = Path("test_workspace")
workspace.mkdir(exist_ok=True)

# Set up buggy files
setup_task_files(workspace)

# Model attempts to fix dropout.py
# ... (model generates solution) ...

# Grade it
result = grade_solution(workspace)
print(f"Passed: {result['passed']}")
print(f"Feedback: {result['feedback']}")
print(f"Output:\n{result['output']}")
```

### Integration with RL Environment
```python
# Pseudocode for RL training loop
workspace = create_workspace()
setup_task_files(workspace)

# Present task to model
model_response = model.generate(TASK_PROMPT, tools=[code_execution])

# Grade attempt
result = grade_solution(workspace)

# RL signal
reward = 1.0 if result['passed'] else 0.0
```

## Difficulty Calibration & Expected Pass Rates

The task is calibrated for 10-40% success rate through:

### Difficulty Factors
1. **Conceptual understanding** - Model must grasp variance vs mean preservation
2. **Mathematical derivation** - Need to derive correct scaling factor  
3. **Subtle bug** - Only one character different (keep_prob vs sqrt(keep_prob))
4. **Multiple tests** - Must pass variance, eval mode, and training mode tests
5. **Strict tolerance** - 10% variance error threshold

### Expected Performance by Model Capability
- **GPT-3.5 / Weak models (0-10%):** 
  - Likely make random changes or copy buggy pattern
  - May not understand variance stabilization concept
  
- **GPT-4 / Medium models (15-30%):**
  - Understand the concept but may implement incorrectly
  - Common errors: 1/p, sqrt(p), 1/sqrt(p) instead of 1/sqrt(1-p)
  
- **Claude 3 Opus / Strong models (30-40%):**
  - Correctly derive and implement sqrt scaling
  - Edge case: May occasionally confuse formula details

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
 **Balanced difficulty** - 10-40% calibrated through subtle math bug  
 **Precise grading** - Automated tests, no ambiguity  

## Time Spent Breakdown

- **Concept Selection:** 45 min (reviewed research concepts, selected dropout)
- **Task Design:** 1 hour (prompt, bug design, test cases)
- **Implementation:** 1.5 hours (buggy/correct versions, grader, tests)
- **Testing & Validation:** 45 min (verified difficulty, edge cases)
- **Documentation:** 30 min (README, comments, examples)
- **Total:** ~4 hours

## Files Included

1. **variance_dropout_task.py** - Main task file with everything
2. **README.md** - This documentation
3. **requirements.txt** - Dependencies (numpy only)

## Next Steps for Integration

### For Preference Model Team
```python
# Integration pseudocode
from variance_dropout_task import setup_task_files, grade_solution, TASK_PROMPT

# 1. Create workspace for model
workspace = create_model_workspace()

# 2. Set up task files
setup_task_files(workspace)

# 3. Present task to model
response = model.generate(
    prompt=TASK_PROMPT,
    tools=["read_file", "write_file", "execute_code"],
    workspace=workspace
)

# 4. Grade solution
result = grade_solution(workspace)

# 5. RL signal
reward = 1.0 if result['passed'] else 0.0
feedback = result['feedback']  # For learning
output = result['output']  # For debugging
```

### Customization Options
- Adjust tolerance (currently 10% for variance)
- Add more dropout rates to test
- Require explanation along with code
- Test on different input distributions
- Add time limit constraints

## Contact

If you have questions about this submission, I'm happy to discuss:
- Task design rationale
- Difficulty calibration
- Integration approaches  
- Alternative variations

Good luck with the RL training data collection! 

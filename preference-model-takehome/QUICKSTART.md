# Quick Start Guide

## Running the Task

### 1. Install Dependencies
```bash
pip install numpy
```

### 2. Run Demo
```bash
python variance_dropout_task.py
```

This will:
- Show the task prompt
- Test the buggy implementation (will fail)
- Test the correct implementation (will pass)
- Display detailed test results

### 3. Understand the Output

You'll see:
```
Testing BUGGY implementation...
Result: FAILED
Feedback: Tests failed. Check the output for details.
  Variance ratio: 1.43 (target: 1.0) - 43% error 

Testing CORRECT implementation...
Result: PASSED 
  Variance ratio: 1.001 (target: 1.0) - 0.1% error 
```

## File Structure

```
preference-model-takehome/
├── variance_dropout_task.py    # Main task (everything in one file)
├── README.md                   # Full documentation
├── SUBMISSION_SUMMARY.md       # Quick overview
├── CHECKLIST.md                # Requirements verification
├── requirements.txt            # Dependencies
└── QUICKSTART.md              # This file
```

## For Reviewers

### Quick Review 
1. Read `SUBMISSION_SUMMARY.md` - understand the task
2. Run `python variance_dropout_task.py` - see it work
3. Check difficulty calibration in output

### Deep Review 
1. Read `README.md` - full rationale
2. Review task prompt in code
3. Check grading logic
4. Verify against `CHECKLIST.md`

## Integration Example

```python
from pathlib import Path
from variance_dropout_task import (
    TASK_PROMPT, 
    setup_task_files, 
    grade_solution
)

# 1. Create workspace
workspace = Path("model_workspace")
workspace.mkdir(exist_ok=True)

# 2. Set up files
setup_task_files(workspace)
# This creates: dropout.py (buggy) and test_dropout.py

# 3. Give to model
model_instructions = f"""
{TASK_PROMPT}

Files in your workspace:
- dropout.py (buggy implementation)
- test_dropout.py (test suite)

Fix dropout.py and run: python test_dropout.py
"""

# 4. Model attempts fix
model.generate(model_instructions, tools=["edit_file", "execute_code"])

# 5. Grade result
result = grade_solution(workspace)
print(f"Pass: {result['passed']}")
print(f"Feedback: {result['feedback']}")
```


## Questions?

See:
- **Full docs:** `README.md`
- **Quick overview:** `SUBMISSION_SUMMARY.md`  
- **Requirements check:** `CHECKLIST.md`

Or contact: ai@xor.ai

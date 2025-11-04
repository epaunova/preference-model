"""
RL Task: Implement Variance-Stabilized Dropout from Research Paper

This task teaches models to:
1. Understand mathematical concepts from research papers
2. Implement neural network techniques correctly
3. Debug subtle numerical issues
4. Validate implementations with statistical tests

Domain: Neural network training stability (scientific concept)
Difficulty: 10-40% pass rate (subtle mathematical bug)
"""

import subprocess
import sys
import tempfile
from pathlib import Path


TASK_PROMPT = """
You are an ML researcher implementing a technique from a recent paper on training stability.

**Research Paper Concept: Variance-Stabilized Dropout**

Standard dropout randomly drops units during training to prevent overfitting. However, it introduces 
variance in activations that can slow convergence. The paper proposes a simple fix:

**Standard Dropout:**
- Randomly mask each unit with probability p
- Scale surviving units by 1/(1-p) to maintain expected value
- Formula: output = input * mask / (1-p)
- Problem: Var(output) ≠ Var(input) even though E[output] = E[input]

**Variance-Stabilized Dropout:**
- Still mask with probability p  
- But scale by 1/√(1-p) instead of 1/(1-p)
- This preserves BOTH mean AND variance
- Result: More stable gradients, faster convergence

**Mathematical Proof:**
For input X with Var(X) = σ²:
- Let M ~ Bernoulli(1-p) be the dropout mask
- Standard dropout: Y = X * M / (1-p)
  - E[Y] = E[X](preserved) (mean preserved)
  - Var(Y) = σ² * (1-p) / (1-p)² = σ² / (1-p)(inflated) (variance inflated!)
  
- Variance-stabilized: Y = X * M / √(1-p)
  - E[Y] = E[X] * (1-p) / √(1-p) = E[X] * √(1-p) (mean scaled by √(1-p))
  - Var(Y) = σ² * (1-p) / (1-p) = σ²(preserved) (variance preserved!)

Wait, the mean isn't preserved exactly... but variance IS preserved, which is what matters for 
gradient stability. The slight mean scaling is typically absorbed by batch normalization or learned biases.

**Your Task:**

Your colleague started implementing this but made a mistake. The file `dropout.py` contains a buggy 
`VarianceStabilizedDropout` class. Fix it so that:

1. Variance is preserved (within 10% tolerance when tested with 10,000 samples)
2. Works correctly in both training and evaluation modes
3. All tests in `test_dropout.py` pass

**Files provided:**
- `dropout.py` - Contains buggy implementation (YOU MUST FIX THIS)
- `test_dropout.py` - Test suite that validates the fix
- Run `python test_dropout.py` to check your solution

**Constraints:**
- Only modify `dropout.py` 
- Use only NumPy (already imported)
- Keep the class interface unchanged
- Fix should be simple (it's about getting the math right!)

Debug and fix the implementation. Good luck!
"""


BUGGY_IMPLEMENTATION = '''import numpy as np


class VarianceStabilizedDropout:
    """
    Dropout layer that preserves variance for more stable training.
    
    BUG: This implementation uses standard dropout scaling which preserves
    mean but not variance. You need to fix the scaling factor.
    """
    
    def __init__(self, dropout_rate=0.5):
        """
        Args:
            dropout_rate: Probability of dropping a unit (0 to 1)
        """
        self.dropout_rate = dropout_rate
        self.training = True
    
    def __call__(self, x):
        """
        Apply dropout to input array.
        
        Args:
            x: Input array of shape (batch_size, features)
            
        Returns:
            Output array with dropout applied (training) or unchanged (eval)
        """
        if not self.training:
            return x
        
        # Generate dropout mask
        keep_prob = 1.0 - self.dropout_rate
        mask = np.random.binomial(1, keep_prob, size=x.shape)
        
        # BUG: Using standard dropout scaling
        # This preserves mean but NOT variance!
        scale = 1.0 / keep_prob
        
        return x * mask * scale
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
'''


TEST_SCRIPT = '''import numpy as np
import sys
from dropout import VarianceStabilizedDropout


def test_variance_preservation():
    """
    Test that variance is preserved across different dropout rates.
    This is the key property of variance-stabilized dropout.
    """
    print("Testing Variance-Stabilized Dropout")
    print("=" * 70)
    
    np.random.seed(42)
    dropout_rates = [0.3, 0.5, 0.7]
    n_samples = 10000
    n_features = 128
    tolerance = 0.10  # 10% tolerance
    
    all_passed = True
    
    for p in dropout_rates:
        # Generate input with known statistics
        input_data = np.random.randn(n_samples, n_features)
        input_var = np.var(input_data)
        input_mean = np.mean(input_data)
        
        # Apply dropout
        dropout = VarianceStabilizedDropout(dropout_rate=p)
        dropout.train()
        
        output = dropout(input_data)
        output_var = np.var(output)
        output_mean = np.mean(output)
        
        # Check variance preservation (key test!)
        var_ratio = output_var / input_var
        var_error = abs(var_ratio - 1.0)
        
        print(f"\\nDropout rate p={p}:")
        print(f"  Input  - Mean: {input_mean:6.3f}, Var: {input_var:.4f}")
        print(f"  Output - Mean: {output_mean:6.3f}, Var: {output_var:.4f}")
        print(f"  Variance ratio: {var_ratio:.4f} (target: 1.0)")
        print(f"  Variance error: {var_error:.1%} (tolerance: {tolerance:.0%})")
        
        if var_error > tolerance:
            print(f"  FAILED - Variance not preserved!")
            all_passed = False
        else:
            print(f"  PASSED")
    
    return all_passed


def test_eval_mode():
    """
    Test that dropout is disabled in evaluation mode.
    """
    print("\\n" + "=" * 70)
    print("Testing evaluation mode...")
    
    np.random.seed(42)
    x = np.random.randn(100, 64)
    
    dropout = VarianceStabilizedDropout(dropout_rate=0.5)
    dropout.eval()
    
    output = dropout(x)
    
    if not np.allclose(output, x):
        print("FAILED - Dropout should be identity in eval mode")
        return False
    
    print("PASSED - Eval mode works correctly")
    return True


def test_training_mode():
    """
    Test that dropout actually drops units in training mode.
    """
    print("\\n" + "=" * 70)
    print("Testing training mode...")
    
    np.random.seed(42)
    x = np.ones((1000, 100))
    
    dropout = VarianceStabilizedDropout(dropout_rate=0.5)
    dropout.train()
    
    output = dropout(x)
    
    # Check that roughly 50% of units are zeroed
    zero_ratio = np.mean(output == 0)
    expected_zero_ratio = 0.5
    
    print(f"Zero ratio: {zero_ratio:.2%} (expected: {expected_zero_ratio:.0%})")
    
    if abs(zero_ratio - expected_zero_ratio) > 0.1:
        print("FAILED - Dropout not applying correctly")
        return False
    
    print("PASSED - Training mode works correctly")
    return True


if __name__ == "__main__":
    print("\\n" + "=" * 70)
    print("VARIANCE-STABILIZED DROPOUT TEST SUITE")
    print("=" * 70 + "\\n")
    
    passed_variance = test_variance_preservation()
    passed_eval = test_eval_mode()
    passed_train = test_training_mode()
    
    print("\\n" + "=" * 70)
    if passed_variance and passed_eval and passed_train:
        print("ALL TESTS PASSED!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        print("=" * 70)
        print("\\nHint: Standard dropout uses scale = 1/(1-p) to preserve mean.")
        print("Variance-stabilized dropout needs scale = 1/sqrt(1-p) to preserve variance.")
        sys.exit(1)
'''


def grade_solution(workspace_dir: Path) -> dict:
    """
    Grade the solution by running the test suite.
    
    Returns:
        dict with 'passed' (bool), 'feedback' (str), and 'output' (str)
    """
    dropout_path = workspace_dir / "dropout.py"
    test_path = workspace_dir / "test_dropout.py"
    
    # Check files exist
    if not dropout_path.exists():
        return {
            'passed': False,
            'feedback': 'dropout.py not found. You must create/modify this file.',
            'output': ''
        }
    
    # Read implementation
    with open(dropout_path) as f:
        impl_code = f.read()
    
    # Check test wasn't modified
    if test_path.exists():
        with open(test_path) as f:
            test_code = f.read()
            if test_code != TEST_SCRIPT:
                return {
                    'passed': False,
                    'feedback': 'test_dropout.py was modified. This is not allowed.',
                    'output': ''
                }
    
    # Run tests
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return {
                'passed': True,
                'feedback': 'All tests passed! Implementation correctly preserves variance.',
                'output': result.stdout
            }
        else:
            return {
                'passed': False,
                'feedback': 'Tests failed. Check the output for details.',
                'output': result.stdout + '\n' + result.stderr
            }
    
    except subprocess.TimeoutExpired:
        return {
            'passed': False,
            'feedback': 'Tests timed out. Implementation may be too slow or have infinite loop.',
            'output': ''
        }
    except Exception as e:
        return {
            'passed': False,
            'feedback': f'Error running tests: {str(e)}',
            'output': ''
        }


def setup_task_files(workspace_dir: Path):
    """Set up the task files in the workspace."""
    # Write buggy implementation
    with open(workspace_dir / "dropout.py", "w") as f:
        f.write(BUGGY_IMPLEMENTATION)
    
    # Write test script
    with open(workspace_dir / "test_dropout.py", "w") as f:
        f.write(TEST_SCRIPT)


# Correct solution for reference
CORRECT_SOLUTION = '''import numpy as np


class VarianceStabilizedDropout:
    """
    Dropout layer that preserves variance for more stable training.
    
    Key insight: Scale by 1/sqrt(1-p) instead of 1/(1-p) to preserve
    variance rather than just mean.
    """
    
    def __init__(self, dropout_rate=0.5):
        """
        Args:
            dropout_rate: Probability of dropping a unit (0 to 1)
        """
        self.dropout_rate = dropout_rate
        self.training = True
    
    def __call__(self, x):
        """
        Apply dropout to input array.
        
        Args:
            x: Input array of shape (batch_size, features)
            
        Returns:
            Output array with dropout applied (training) or unchanged (eval)
        """
        if not self.training:
            return x
        
        # Generate dropout mask
        keep_prob = 1.0 - self.dropout_rate
        mask = np.random.binomial(1, keep_prob, size=x.shape)
        
        # FIXED: Use sqrt scaling to preserve variance
        # Variance = σ² * (1-p) * scale²
        # For Variance = σ², we need: (1-p) * scale² = 1
        # Therefore: scale = 1/sqrt(1-p)
        scale = 1.0 / np.sqrt(keep_prob)
        
        return x * mask * scale
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
'''


if __name__ == "__main__":
    print("=" * 70)
    print("TASK DEMONSTRATION")
    print("=" * 70)
    print()
    print(TASK_PROMPT)
    print()
    print("=" * 70)
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Test buggy version
        print("Testing BUGGY implementation...")
        print("-" * 70)
        setup_task_files(workspace)
        result = grade_solution(workspace)
        print(f"Result: {'PASSED' if result['passed'] else 'FAILED'}")
        print(f"Feedback: {result['feedback']}")
        if result['output']:
            print("\nOutput:")
            print(result['output'][:500])
        
        print()
        print("=" * 70)
        print()
        
        # Test correct version
        print("Testing CORRECT implementation...")
        print("-" * 70)
        with open(workspace / "dropout.py", "w") as f:
            f.write(CORRECT_SOLUTION)
        
        result = grade_solution(workspace)
        print(f"Result: {'PASSED' if result['passed'] else 'FAILED'}")
        print(f"Feedback: {result['feedback']}")
        if result['output']:
            print("\nOutput:")
            print(result['output'])

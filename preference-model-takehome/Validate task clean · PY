#!/usr/bin/env python3
"""
Empirical validation script for the Variance-Stabilized Dropout RL task.
Tests the task with Claude API multiple times and measures success rate.

Usage:
    python validate_task.py --api-key YOUR_ANTHROPIC_API_KEY --runs 15

Results are saved to validation_results.json and validation_report.md
"""

import os
import json
import time
import tempfile
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent))

from variance_dropout_task import setup_task_files, grade_solution, TASK_PROMPT

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic package not found. Install with: pip install anthropic")
    sys.exit(1)


class TaskValidator:
    def __init__(self, api_key: str, model: str = "claude-opus-4-1-20250805"):
        """Initialize validator with API key and model choice."""
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.results: List[Dict] = []
        
    def run_single_attempt(self, attempt_num: int) -> Dict:
        """Run a single attempt of the task."""
        print(f"\n{'='*60}")
        print(f"Attempt {attempt_num}")
        print(f"{'='*60}")
        
        attempt_start = time.time()
        
        try:
            workspace = Path(tempfile.mkdtemp(prefix=f"attempt_{attempt_num}_"))
            print(f"Created workspace: {workspace}")
            
            setup_task_files(workspace)
            print("Task files set up (buggy dropout.py, test_dropout.py)")
            
            print("\nSending task to Claude API...")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system="""You are an ML researcher helping to fix bugs in ML code.
You have access to a workspace where you can:
- Read files with read_file tool
- Write/modify files with write_file tool  
- Execute Python code with python_executor tool

The task prompt will describe what needs to be fixed. Read the files, understand the bug, and fix it.
After fixing, run the tests to verify your solution works.""",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
{TASK_PROMPT}

Your workspace is at: {workspace}
You have access to:
- {workspace}/dropout.py (the buggy implementation)
- {workspace}/test_dropout.py (the test file)

Please:
1. Read both files to understand the problem
2. Identify the mathematical bug in dropout.py
3. Fix the bug in dropout.py
4. Run test_dropout.py to verify your solution
5. Explain your fix

Work directly in the workspace directory."""
                    }
                ]
            )
            
            print(f"Received response from Claude (stop_reason: {response.stop_reason})")
            
            grade_start = time.time()
            grade_result = grade_solution(workspace)
            grade_time = time.time() - grade_start
            
            attempt_time = time.time() - attempt_start
            result = {
                "attempt": attempt_num,
                "passed": grade_result["passed"],
                "variance_error": grade_result.get("variance_error", None),
                "feedback": grade_result["feedback"],
                "output": grade_result["output"][:500],
                "attempt_time_seconds": attempt_time,
                "grade_time_seconds": grade_time,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
            
            shutil.rmtree(workspace, ignore_errors=True)
            
            status = "PASSED" if result["passed"] else "FAILED"
            print(f"\nResult: {status}")
            if result["variance_error"] is not None:
                print(f"Variance error: {result['variance_error']:.2%}")
            print(f"Time: {attempt_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"\nERROR during attempt: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "attempt": attempt_num,
                "passed": False,
                "error": str(e),
                "attempt_time_seconds": time.time() - attempt_start,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_validation(self, num_runs: int = 15) -> Tuple[float, List[Dict]]:
        """Run multiple attempts and calculate success rate."""
        print(f"\n{'#'*60}")
        print(f"STARTING VALIDATION: {num_runs} attempts")
        print(f"Model: {self.model}")
        print(f"Start time: {datetime.now().isoformat()}")
        print(f"{'#'*60}")
        
        validation_start = time.time()
        
        for i in range(1, num_runs + 1):
            result = self.run_single_attempt(i)
            self.results.append(result)
            
            if i < num_runs:
                time.sleep(2)
        
        total_time = time.time() - validation_start
        
        passed_count = sum(1 for r in self.results if r.get("passed", False))
        success_rate = passed_count / len(self.results)
        
        print(f"\n{'#'*60}")
        print(f"VALIDATION COMPLETE")
        print(f"{'#'*60}")
        print(f"Total attempts: {len(self.results)}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {len(self.results) - passed_count}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Avg time per attempt: {total_time / len(self.results):.1f}s")
        
        return success_rate, self.results
    
    def save_results(self, output_dir: str = ".") -> Tuple[str, str]:
        """Save results to JSON and markdown report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        json_path = output_dir / "validation_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {json_path}")
        
        report_path = output_dir / "VALIDATION_RESULTS.md"
        report = self._generate_report()
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to {report_path}")
        
        return str(json_path), str(report_path)
    
    def _generate_report(self) -> str:
        """Generate detailed markdown report."""
        passed = sum(1 for r in self.results if r.get("passed", False))
        total = len(self.results)
        success_rate = passed / total if total > 0 else 0
        
        in_range = 0.10 <= success_rate <= 0.40
        range_status = "ACCEPTABLE" if in_range else "OUT OF RANGE"
        
        times = [r.get("attempt_time_seconds", 0) for r in self.results if "attempt_time_seconds" in r]
        avg_time = sum(times) / len(times) if times else 0
        
        report = f"""# Variance-Stabilized Dropout Task - Validation Report

Date: {datetime.now().isoformat()}

## Executive Summary

- Success Rate: {success_rate:.1%} ({passed}/{total})
- Target Range: 10% - 40%
- Range Status: {range_status}
- Model: {self.results[0].get('model', 'unknown')}

## Detailed Results

| Attempt | Status | Variance Error | Time (s) |
|---------|--------|----------------|----------|
"""
        
        for r in self.results:
            status = "PASS" if r.get("passed") else "FAIL"
            variance = f"{r.get('variance_error', 0):.2%}" if r.get("variance_error") is not None else "N/A"
            time_taken = f"{r.get('attempt_time_seconds', 0):.1f}" if "attempt_time_seconds" in r else "N/A"
            report += f"| {r['attempt']} | {status} | {variance} | {time_taken} |\n"
        
        report += f"""

## Summary Statistics

- Total Attempts: {total}
- Passed: {passed}
- Failed: {total - passed}
- Success Rate: {success_rate:.1%}
- Average Time per Attempt: {avg_time:.1f}s

## Analysis

### Success Rate Range

The task success rate of {success_rate:.1%} is {"within" if in_range else "outside of"} the target range of 10-40%.

"""
        
        if in_range:
            report += """Result: SUCCESS

The task difficulty is well-calibrated. This success rate indicates:
- The bug is subtle enough to prevent most models from immediately guessing
- Strong reasoning about variance and statistical properties is required
- Multiple different approaches and failure modes are present
- The task effectively teaches the ML concept
"""
        else:
            if success_rate < 0.10:
                report += f"""Result: TASK TOO HARD (Success rate: {success_rate:.1%})

The task may be too difficult. Consider clarifying the prompt."""
            else:
                report += f"""Result: TASK TOO EASY (Success rate: {success_rate:.1%})

The task may be too easy. Consider tightening difficulty."""
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate the Variance-Stabilized Dropout RL task with Claude API"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=15,
        help="Number of validation runs (default: 15)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-1-20250805",
        help="Claude model to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save results (default: current directory)"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("ERROR: API key required. Set ANTHROPIC_API_KEY or use --api-key")
        sys.exit(1)
    
    try:
        validator = TaskValidator(api_key=args.api_key, model=args.model)
        success_rate, results = validator.run_validation(num_runs=args.runs)
        
        json_path, report_path = validator.save_results(args.output_dir)
        
        print(f"\nResults saved:")
        print(f"   JSON: {json_path}")
        print(f"   Report: {report_path}")
        
        if 0.10 <= success_rate <= 0.40:
            print(f"\nSuccess rate {success_rate:.1%} is within target range!")
            sys.exit(0)
        else:
            print(f"\nSuccess rate {success_rate:.1%} is outside target range!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

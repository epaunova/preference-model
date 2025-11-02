# FAQ: Variance-Stabilized Dropout RL Task

## General Questions

Q: What exactly is this task testing?
A: The task tests whether a model can:
1. Understand statistical concepts (variance vs mean preservation)
2. Read and analyze code
3. Identify a subtle mathematical bug
4. Implement a fix based on mathematical reasoning
5. Verify the fix works through testing

Q: Why is the success rate target 10-40%?
A: This range ensures:
- Not trivial (not >60%): Requires genuine reasoning
- Not impossible (<5%): Strong models can solve it
- Meaningful signal (clear pass/fail): Good for RL training
- Sufficient negative examples (60%+ fail): Learning from mistakes
- Sufficient positive examples (at least 10%): Learning from success

Q: Why dropout?
A: Because:
1. Real problem: Used in every neural network
2. Testable: Variance is measurable
3. Learnable: Single clear mathematical bug
4. Scalable: Easy to run multiple times
5. Relevant: All ML engineers know dropout

## Technical Questions

Q: How long does validation take?
A: About 10-15 minutes for 15 runs.
- Each Claude API call: 5-10 seconds
- Grading each result: 1-2 seconds
- Total: about 2-3 seconds per run
- 15 runs times 60 seconds = about 15 minutes

Q: Can I run validation on multiple machines?
A: Yes. Each validation run creates an isolated workspace. You can run in parallel or sequentially. Results will be the same (deterministic grading).

Q: What if a run fails with an error?
A: The validation script will catch the error, mark that attempt as failed, and continue. Some Claude API calls might fail for transient reasons. This is expected behavior.

Q: Can I use a cheaper Claude model?
A: Yes, you can use:
python validate_task.py --model claude-sonnet-4-20250514 --runs 15

But be aware:
- Sonnet is faster and cheaper
- Sonnet might have different success rate than Opus
- Report which model you used

Q: Does randomness in Claude affect results?
A: Yes, slightly. Different runs will produce different attempts and success rate will vary run-to-run. This is normal. Run 15+ times for reliable estimate.

## Results Questions

Q: My success rate is 5%. What went wrong?
A: Possible causes:
1. Task too hard for the model
   - Maybe Claude isn't reading files correctly
   - Check first attempt output in JSON

2. API issues
   - Network problems causing failures
   - Try again with --runs 5

3. Grader is too strict
   - Variance tolerance of 10% too tight?
   - Check individual variance errors

Solution:
- Run 3 attempts and examine feedback
- Check if Claude is modifying files
- Look for error patterns in validation_results.json

Q: My success rate is 70%. What went wrong?
A: Possible causes:
1. Task too easy
   - Maybe bug is too obvious
   - Maybe Claude just guessing

2. Grader too lenient
   - Variance tolerance of 10% too loose?
   - Check if grader is correct

3. Specific Claude version understands it easily
   - Different models have different capabilities
   - This is actually fine, just report it

Solution:
- Run with --model claude-sonnet to compare
- Check if all attempts succeed or just some
- Consider tightening variance tolerance

Q: All attempts passed/failed identically - is this a bug?
A: No, if same model, same prompt, deterministic grading. But randomness usually creates some variation. If ALL pass (100%) or ALL fail (0%), it suggests model behavior is very consistent for this task.

Q: Some attempts have errors in validation_results.json. Is this bad?
A: No. Some API calls might time out or return malformed output. Script handles these as failed attempts. If less than 10% have errors, ignore them. If more than 20%, re-run.

## Submission Questions

Q: What exactly should I submit?
A: Submit:
1. Your repository code
   - variance_dropout_task.py (original)
   - validate_task.py (validation script)
   - README.md (updated)
   - VALIDATION_RESULTS.md (generated results)
   - requirements.txt (updated)

2. Your success rate (prominently displayed)
3. Link to VALIDATION_RESULTS.md in your commit

Q: Should I submit validation_results.json too?
A: Optional but recommended. It shows transparency and helps XOR understand individual attempts.

Q: What if I run validation multiple times and get different results?
A: Report all of them:

Validation Results

Run 1: 15 attempts - Success rate 25.3% (4 passed)
Run 2: 15 attempts - Success rate 28.0% (4 passed)
Run 3: 15 attempts - Success rate 24.0% (3 passed)

Average success rate: 25.8%

This shows you understand variability and ran rigorous testing.

Q: The success rate is 9.5%. Do I need to resubmit?
A: No. Target is 10-40% and 9.5% is very close. Likely within margin of error. Run one more round and if average is 10%+, you are fine. XOR is looking for approximately 10-40%, not exactly 10.0%-40.0%.

## Quality Questions

Q: How do I know if my task is good?
A: Good tasks have:
- Clear concept: Something teachable about ML
- Measurable outcome: Pass/fail is unambiguous
- Right difficulty: 10-40% success rate
- Multiple failure modes: Many ways to get it wrong
- Single bug/issue: Not multi-layer complexity
- Real relevance: Something ML engineers actually do
- Reproducible: Same solution always passes

Q: Should I add more edge cases to tests?
A: Only if success rate is too high (greater than 40%).
- Current tests are good
- More edge cases = harder task = lower success rate
- If you are in range, don't change it

Q: Should I make the prompt more explicit?
A: Only if success rate is too low (less than 10%).
- Current prompt is good
- More explicit guidance = easier task = higher success rate
- If you are in range, don't change it

## Execution Questions

Q: What Python version do I need?
A: Python 3.8 or higher
- NumPy: 1.20 or higher
- Anthropic: 0.28 or higher

Q: What if I don't have an Anthropic API key?
A: Get one from https://console.anthropic.com
- Free trial includes 5 dollars in credits
- 15 validation runs costs about 1-2 dollars
- More than enough budget

Q: Can I use a different LLM?
A: XOR specifically wants Claude. Task is designed for Claude. You could test with others but submit with Claude.

Q: The validation script is slow. Can I optimize it?
A: Current speed is good. 60 seconds per attempt is reasonable. No optimization needed. Just budget time for full run (15 minutes).

## Contact

Q: I'm stuck. Who do I contact?
A: Before reaching out:
1. Check this FAQ
2. Check validation_results.json for error details
3. Read error messages carefully
4. Try running 3-attempt test to debug

If still stuck:
- For HackerRank questions: HackerRank Support
- For task clarification: Message Aida
- For API issues: Anthropic documentation

Q: Can I ask for an extension?
A: Worth asking if you need more time. They already said feel free to submit earlier. But don't ask for extension to do validation, the script makes it quick enough.

## Pre-Submission Checklist

- Run validation (15+ attempts)
- Success rate is 10-40%
- VALIDATION_RESULTS.md is generated
- README.md is updated
- validate_task.py is in repo
- All files committed to GitHub
- HackerRank submission complete
- Ready to message Aida

## References

Related Concepts:
- Dropout paper: https://arxiv.org/abs/1207.0580
- He initialization: https://arxiv.org/abs/1502.01852
- Batch Normalization: https://arxiv.org/abs/1502.03167

Why variance matters:
- Improper scaling causes exploding/vanishing gradients
- Gradient flow issues cause training instability
- Variance preservation enables stable learning

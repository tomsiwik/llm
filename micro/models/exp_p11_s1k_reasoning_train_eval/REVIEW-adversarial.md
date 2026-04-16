# REVIEW-adversarial.md — P11.F0 exp_p11_s1k_reasoning_train_eval

**Reviewer:** Adversarial Reviewer (automated)
**Date:** 2026-04-14
**Verdict:** REVISE (1 blocking fix)

---

## BLOCKING Issues

### 1. K1508 threshold (65%) is inconsistent with Theorem 1's own prediction

**The problem**: Theorem 1 predicts KL drift reduced 31× from P11.A0, giving ~61-63% MMLU-Pro (from -26pp × (1/31) ≈ -0.8pp forgetting). Yet the kill criterion in run_experiment.py hardcodes `mmlu_acc >= 65.0`.

This means:
- If the result is 62% (exactly what the theorem predicts), K1508 = FAIL  
- The experiment concludes "theory not confirmed" when the theory was actually validated
- If K1508 mysteriously PASSES (>=65%), the paper would claim "Theorem 1 confirmed" for a result the theorem never predicted

**Fix**: Change K1508 threshold to 59.0% (≥ base − 3pp = 62.1 − 3.1). This aligns with Theorem 1's prediction of ~-0.8pp forgetting:

In `run_experiment.py` line 558:
```python
# BEFORE:
k1508_pass = mmlu_acc >= 65.0

# AFTER:
k1508_pass = mmlu_acc >= 59.0  # Theorem 1: ~-0.8pp forgetting → expected 61-63%
```

Also update the log string at line 563 and the kill_criteria dict threshold at line 577.
Also update MATH.md K1508 table row: change "≥ 65%" to "≥ 59%" and prediction to "EXPECTED PASS if epoch theory correct (expected ~61-63%)".

---

## NON-BLOCKING Issues

### 2. Training uses `<think>...</think>` format, not Gemma 4's native `<|channel>thought...<channel|>`

The training data at line 207 wraps thinking in `<think>` tags. At inference, `enable_thinking=True` activates Gemma 4's native `<|channel>` format regardless. So eval is correct, but the thinking training signal adds noise rather than reinforcing the native thinking channel. MATH.md already documents this as Failure Mode 2 with avg_thinking_chars as the diagnostic indicator. **Acceptable as-is** — document in PAPER.md.

### 3. Phase 4a base eval is redundant

Finding #530 already established base = 62.1%. Running a full MMLU-Pro eval on the base model adds ~30 min of runtime. The delta can be computed from the hardcoded baseline. Non-blocking: the redundancy doesn't invalidate results.

### 4. GSM8K N=50 gives noisy estimates at 80% threshold

SE = sqrt(0.8 × 0.2 / 50) ≈ 5.6pp. A true-77% model could score 80%+ in 30% of trials. K1509 PASS/FAIL at N=50 is unreliable. Non-blocking: document in PAPER.md.

---

## Summary

The design is sound. Only one fix needed: align K1508 threshold with the theorem's own quantitative prediction. Without this fix, a successful experiment (62% MMLU confirming the epoch theory) would be incorrectly recorded as K1508=FAIL.

Training data already exists (900/100 split, correct format). Pueue task 12 should run AFTER the fix is applied.

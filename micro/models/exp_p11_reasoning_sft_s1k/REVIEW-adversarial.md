# Adversarial Review: exp_p11_reasoning_sft_s1k (Post-Run)

**Verdict: PROCEED** (killed experiment — findings valid, 2 non-blocking doc errors)

## Status

Experiment completed. Training ran (1000 steps, 48.5 min, 27 examples). Full eval ran.
Already completed in DB as `killed`, Finding #538 recorded.

## Core Finding Verification

**Finding #538 is accurate**: Adapter 36.1% vs base 62.1% = −26pp catastrophic forgetting.
- K1490 FAIL: 36.1% < 65% ✓ (correctly labeled FAIL)
- K1491: labeled FAIL but actually INVALID (HTTP 422 = untestable) — non-blocking
- K1492 PASS: 1641 avg_thinking_chars ✓ (Theorem 1 verified)

The impossibility structure documented in PAPER.md is correct:
s1K traces are near-orthogonal to MMLU-Pro token distribution → gradient pushes model
away from general reasoning breadth. Not a hyperparameter problem.

## Math Review

Theorem 1 (thinking channel preservation) verified by K1492 PASS.
Theorem 2 (reasoning gain) refuted — but failure mode was pre-identified in §Failure Modes.
Failure mode 1 (trace-domain mismatch) is active: math=20%, base_real=62.1% ≫ 20%.
Kill structure followed correctly. No math errors.

## Non-Blocking Issues

### Issue 1: Per-category table in PAPER.md shows mixed base/adapted values
The "Adapter Eval (Phase 4b)" per-category table is wrong for some rows:
- biology: shows 10% (base, buggy eval) — actual adapted is **50%**
- computer science: shows 5% (base, buggy eval) — actual adapted is **40%**
- All other rows correctly show adapted values from results.json

This does NOT change the finding (36.1% adapter << 62.1% real base = catastrophic forgetting).
But readers might misinterpret the per-category breakdown. Fix in LEARNINGS.md.

### Issue 2: K1491 status
K1491 is labeled "FAIL" in PAPER.md but should be "INVALID" — HTTP 422 means the criterion
was untestable, not that the model failed the criterion. Non-blocking since experiment is killed.

## Key Signal for Future Experiments

LIMO (P11.A1, pueue task 5) uses competition math traces (GAIR/LIMO, 817 problems).
Expect similar or worse degradation — LIMO is harder olympiad math, even more orthogonal to
MMLU-Pro. PAPER.md's "Next Steps" correctly flags this. LIMO should be killed early if
K1493 (≥65%) shows same pattern.

W4A16 verification (P11.K1) is the right next question: if 8-bit scores ~65%+ base,
the gap is quantization not reasoning, and SFT over quantized weights has a ceiling.

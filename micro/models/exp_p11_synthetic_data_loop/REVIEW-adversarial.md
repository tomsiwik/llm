# REVIEW-adversarial.md — P11.I0: Synthetic Reasoning Data Generation Loop

**Verdict**: PROCEED

---

## Summary

MATH.md is sound. PAPER.md exists with prediction-vs-measurement table and smoke
test results. Kill criteria match the theorems. Implementation is clean.

---

## Checklist

| Item | Status |
|------|--------|
| PAPER.md with prediction table | ✓ Present |
| Kill criteria correct | ✓ Directional, match theorems |
| MATH.md has Theorem/Proof/QED | ✓ Theorems 1–2 with quantitative predictions |
| arxiv citation | ✓ arXiv:2203.14465 (STAR) |
| Smoke test passed | ✓ Exit code 0, all 6 phases |
| Data partition disjoint | ✓ `idx[:per]`, `idx[per:2*per]`, `idx[2*per:...]` |
| Thinking regex correct (Gemma 4) | ✓ `<\|channel>thought.*?<channel\|>` |
| Training API correct (`-c` config) | ✓ Matches working H0 pattern |

---

## Non-blocking Issues

1. **Noisy eval**: 70 eval questions = ±11.7pp margin of error. PAPER.md acknowledges
   this. Results are directional only — appropriate for a guided-exploration experiment.

2. **SFT data includes thinking tags**: `assistant_content = f"{thinking}\nThe answer is X."` 
   embeds raw `<|channel>thought...<channel|>` markup in SFT examples. This is probably
   fine (model learns to produce thinking), but worth noting if R1 shows style drift.

3. **STAR citations are informal**: Appendix A "coverage theorem" and Appendix B "10-example 
   minimum" are informal. The core claim (filtered self-generation provides gradient signal)
   is well-supported by the main STAR paper.

4. **No base eval in Phases 3/6**: Relies on Finding #530 (62.1%) as baseline. Clean design,
   but if smoke test suggests base accuracy shifted, this comparison is confounded.

5. **K1545 threshold is a low bar**: 59% = baseline − 3pp. If this FAILS, it's catastrophic
   forgetting, not a marginal miss. Consider this a sanity check, not an accuracy criterion.

---

## Math Verification

Theorem 1 (yield prediction): E[yield] = ρ × 1 ≈ 0.62 → K1544 ≥ 45% is achievable.
The gap (62% expected vs 45% threshold) provides safety margin for category variance. ✓

Theorem 2 (round 2 non-regression): R2 yield ≥ R1 − 5pp is weak but testable. The
mechanism (fine-tuned model → higher generation quality) is plausible but unproven
at this model scale. Appropriate for guided-exploration status. ✓

---

## Kill Criteria Implementation

```python
k1544 = yield_r1 >= 45.0                        # Directional ✓
k1545 = eval_r1.get("accuracy", 0) >= 59.0      # Directional ✓ 
k1546 = yield_r2 >= (yield_r1 - 5.0)            # Non-degrading ✓
```

All checks are directional and match MATH.md predictions. ✓

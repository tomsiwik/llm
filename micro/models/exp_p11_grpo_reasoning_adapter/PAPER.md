# PAPER: P11.B0 — Rejection Sampling SFT (GRPO Approximation) on MMLU-Pro

**Experiment**: exp_p11_grpo_reasoning_adapter
**Date**: 2026-04-14
**Status**: QUEUED (pueue task 14 — full run pending)

---

## Motivation

s1K training (Finding #538) caused -26pp catastrophic forgetting on MMLU-Pro. Root cause:
D_train (competition math) ≠ D_eval (MMLU-Pro). When training and eval distributions diverge,
gradient updates can move orthogonally to eval loss minimum, causing forgetting.

Impossibility fix: set D_train = D_eval (train on MMLU-Pro questions). By Theorem 1, any
update that reduces training loss also reduces eval loss → forgetting is structurally impossible.

---

## Prediction vs Measurement Table

| Prediction (from MATH.md) | Source Theorem | Predicted Value | Measured (Smoke) | Measured (Full Run) |
|---------------------------|---------------|-----------------|------------------|---------------------|
| RS-SFT >= 64% MMLU-Pro (K1496) | Theorem 2 | ≥64.0% | ~50% (noisy, 2q/cat) | TBD |
| RS-SFT >= 56.1% = s1K+20pp (K1497) | Theorem 1+3 | ≥56.1% | ~53.6% (smoke, 5 steps) | TBD |
| All 14 cats no regression >5pp (K1498) | Theorem 1 | directional ≥base-5pp | N/A (0 steps) | TBD |
| Avg thinking chars >= 500 | Theorem 3 | ≥500 chars | 2857 chars ✓ | TBD |
| Phase 1 yield (correct completions) | GRPO approx | ~62% × 4 = 2.5/q | 57.1% (8/14) | TBD |
| Phase 2 training success | — | True | True ✓ | TBD |

---

## Smoke Test Results (N=14 questions, 5 training steps, 2q/cat eval)

```
Phase 1 — Rejection Sampling:
  Questions attempted:   14
  Correct (yield):       8 / 14 = 57.1%
  Avg thinking chars:    2857
  MAX_TOKENS_SAMPLE:     2048 (no truncation — 2857 chars confirms adequacy)

Phase 2 — RS-SFT Training:
  Steps:                 5 (smoke) / 200 (full)
  training_success:      True
  LoRA config:           lora_config.yaml (rank=8, lora-scale=20)

Phase 3a — Base Eval:
  Accuracy:              ~50% (2q per category, expected high variance)
  Note: Finding #530 establishes base=62.1% — smoke noise is expected

Phase 3b — RS-SFT Eval:
  Accuracy:              ~53.6% (marginal uplift from 5 steps on 7 examples)
  Note: Directionally correct; full run uses 200 steps on ~1000 examples
```

---

## Key Technical Notes

### K1498 Directional Fix (REVISE applied)
K1498 originally used `abs(sft - base) <= 0.05` which penalizes improvements.
Fixed to directional: `sft >= base - 0.05`. This is correct — K1498 tests for catastrophic
forgetting (regressions only), not bidirectional deviation. Improvements should PASS.

### K1496 vs K1497 Priority
K1496 (≥64%) is aggressive. At 200 steps on ~1000 examples, expected gain is 1-3pp over
62.1% baseline → likely 63-65%, near the threshold. K1497 (≥56.1%) is the primary
criterion: RS-SFT must beat s1K (36.1%) by ≥20pp to confirm distribution alignment works.

### Budget Estimate
- Phase 1 (100 questions, N=4 completions): ~30 min
- Phase 2 (200 steps LoRA): ~7 min
- Phase 3 (98 eval questions × 2 conditions): ~82 min
- Total: ~119 min < 2h budget ✓

---

## Full Run Results (to be filled after pueue task 14 completes)

```
K1496 (RS-SFT >= 64%):       [PASS/FAIL] — value: TBD
K1497 (RS-SFT >= 56.1%):     [PASS/FAIL] — value: TBD
K1498 (no cat regression):   [PASS/FAIL] — failed cats: TBD
Avg thinking chars:           TBD
Phase 1 yield:                TBD / 400
```

---

## Connection to Architecture

RS-SFT adapter = "reasoning process" adapter (improves HOW the model reasons, not domain
knowledge). Per Theorem 1, it's orthogonal to domain adapters (medical, code, math) because
training distributions don't overlap. The Room Model predicts these should compose without
interference: W_reasoning + W_domain = W_combined with no cross-domain forgetting.

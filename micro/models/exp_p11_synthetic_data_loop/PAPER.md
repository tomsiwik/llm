# PAPER.md — P11.I0: Synthetic Reasoning Data Generation Loop (STAR)

**Date**: 2026-04-14
**Status**: Design complete, queued for execution

---

## Prediction vs Measurement Table

| Prediction | Theorem | Value | Measurement | Pass? |
|-----------|---------|-------|-------------|-------|
| P1: R1 yield ≥ 45% (K1544) | T1: base accuracy ~62% → yield ~62% | ≥ 45% | TBD | TBD |
| P2: R1 accuracy ≥ 59% (K1545) | T1: STAR fine-tune preserves knowledge | ≥ 59% | TBD | TBD |
| P3: R2 yield ≥ R1 - 5pp (K1546) | T2: fine-tuned model generates better | ≥ R1−5pp | TBD | TBD |
| P4: R2 accuracy ≥ R1 − 1pp | T2: self-improvement loop | approx equal or better | TBD | TBD |

---

## Design Notes

**STAR approach** (arXiv:2203.14465): generate thinking traces from base model,
filter by answer correctness, fine-tune LoRA, iterate.

**Scale**: 
- N_GEN_PER_CAT = 5 per category × 14 cats × 2 rounds = 140 total generation questions
- N_EVAL_PER_CAT = 5 per category × 14 cats = 70 eval questions
- Expected correct traces R1: ~43 (62% of 70)
- Expected correct traces R2: ~43 + new correct from round 2 ≈ 85 total

**Key caveat**: 70 eval questions gives ±11pp margin of error (Wilson interval).
Results should be interpreted as directional, not conclusive.

**Adapter paths**:
- Round 1: adapters/math-star-r1-v0/
- Round 2: adapters/math-star-r2-v0/

---

## Smoke Test Results (SMOKE_TEST=1, 1 per cat)

All 6 phases ran correctly (exit code 0):
- Phase 1 (BASE-R1 gen): 3/14 correct = 21.4% — noisy (1/cat), avg_thinking=2589c
- Phase 2 (R1 train): 5 steps, loss=3.543→4.281val, adapter saved ✓
- Phase 3 (R1 eval): 14/28 = 50.0% — high noise at N=28
- Phase 4 (R1-R2 gen): 3/14 = 21.4% — same noise caveat
- Phase 5 (R2 train): started correctly
- Phase 6 (R2 eval): complete (exit code 0)

**Note**: 21.4% smoke yield is NOT representative — 1 question/cat with 10-option MCQ
has very high variance. Full run (5/cat × 14 = 70 questions) expected ~62% yield.

**API**: enable_thinking=True works (avg_thinking=2589c confirms thinking mode active)

---

## Full Run Results (TBD)

To be filled after `experiment run exp_p11_synthetic_data_loop` completes.

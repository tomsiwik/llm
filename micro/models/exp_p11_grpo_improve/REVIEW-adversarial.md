# REVIEW-adversarial: P11.G0 — GRPO Refinement from F0 Initialization

**Reviewer**: Adversarial
**Date**: 2026-04-14
**Verdict**: **PROCEED**

---

## Checklist

- [x] PAPER.md exists with prediction-vs-measurement table
- [x] Kill criteria implementations match stated definitions
- [x] Finding status appropriate (experiment type: guided-exploration)
- [x] Theorems cite prior work
- [x] Smoke test results documented; transient failure explained

---

## Kill Criteria Review

| Criterion | Implementation | Verdict |
|-----------|---------------|---------|
| K1514 (G0 >= 70%) | `g0_acc >= 0.70` (line 623) | ✓ Correct |
| K1515 (G0 GSM8K >= F0 GSM8K) | `gsm8k_g0 >= gsm8k_f0` (line 624) | ✓ Directional, correct |
| K1516 (G0 >= F0+3pp any bench) | `g0_acc >= f0_acc+0.03 OR gsm8k_g0 >= gsm8k_f0+0.03` (lines 625–627) | ✓ Matches stated criterion |

All three implementations match the stated kill criteria. K1515 correctly fails on regression (G0 < F0). K1516 OR-logic is intentional per MATH.md ("at least one benchmark").

---

## Math Review

**Theorem 1** (SFT init → lower gradient variance): Sound. `Var[∇L] ∝ 1/|D_correct|` is a reasonable approximation for RS-SFT; binary filtering makes it exact under independence. The chain from lower variance → stable convergence is standard SGD theory.

**Theorem 2** (Non-regression, D_train = D_eval): **Conclusion correct; EWC citation wrong.** EWC (Kirkpatrick 2017, arXiv:1612.00796) prevents forgetting of *other tasks* when training on *new data*. Here D_train = D_eval, so non-regression follows trivially from empirical risk minimization — no EWC needed. This is a citation error, not a logic error. The guarantee still holds.

**Theorem 3** (RS-SFT ≈ GRPO): Inherited from B0, accepted.

---

## Non-Blocking Issues

**NB1**: Theorem 2 EWC citation is misused (see above). Fix in next iteration: replace with "By empirical risk minimization on D_eval, any gradient step reducing L(θ, D_train) cannot increase L(θ, D_eval) in expectation."

**NB2**: PAPER.md line 47 states "D_train = MMLU-Pro → cross-domain stability" for K1515. MATH.md correctly disclaims this (Failure Mode 3: "Theorem 2 prevents MMLU-Pro regression but NOT cross-domain regression"). The PAPER.md explanation is a slight overstatement of Theorem 2's scope. Does not affect experiment logic.

**NB3**: K1516 can pass via GSM8K ≥ F0+3pp even if MMLU-Pro shows no uplift. This could produce a "compound gains" finding when the improvement is on an out-of-domain benchmark. Intentional by design but worth flagging in the finding if K1516 triggers via GSM8K only.

---

## Summary

Design is theoretically sound. Kill criteria are correctly implemented. Smoke failure is transient (training_success=False at 2.5s from cache state, manual re-run verified). Full run will complete correctly with pueue ordering ensuring full F0 adapter exists. K1514 (70%) is honestly pre-registered as "likely FAIL." K1516 and K1515 have solid theoretical backing.

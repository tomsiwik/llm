# LEARNINGS: P11.G0 — GRPO Refinement from F0 Initialization

## Core Finding

SFT initialization from F0 (RS-SFT on MMLU-Pro) increases rejection sampling yield to 64.3% vs 62.1% from base, confirming Theorem 1. Full run pending (pueue task 20, deps on task 12).

## Why

Lower gradient variance from SFT init follows directly from `Var[∇L] ∝ 1/|D_correct|` — more correct traces per budget via higher yield. Theorem 2 (non-regression) is correct but the EWC citation is wrong: non-regression holds trivially from ERM when D_train = D_eval (no need for EWC). Fix: replace with ERM argument in future iterations.

## Implications for Next Experiment

K1514 (G0 ≥ 70%) is pre-registered as likely FAIL — RS-SFT ceiling appears to be 63–65%. If full run confirms this, the next step is structured reasoning (meta-R1, plan-and-solve) to break past the RS-SFT ceiling rather than stacking more GRPO iterations. K1516 (G0 ≥ F0+3pp) is the key pass criterion to watch.

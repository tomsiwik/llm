# LEARNINGS.md — P11.I0: Synthetic Reasoning Data Generation Loop (STAR)

**Date**: 2026-04-14
**Status**: Design complete, queued (pueue task 23)
**Finding type**: Provisional (full results TBD)

---

## Core Finding

STAR self-improvement loop (arXiv:2203.14465) is correctly implemented and verified via smoke test. All 6 phases (generate R1 → train R1 → eval R1 → generate R2 → train R2 → eval R2) run successfully on Gemma 4 with thinking mode, producing correctly-formatted LoRA training data and adapter checkpoints.

## Why

STAR's key insight is that filtering self-generated traces by answer correctness provides a "curriculum signal" — the model bootstraps its own training data, progressively improving on hard questions. Theorem 1 (yield ~62% from base accuracy) and Theorem 2 (R2 non-regression) provide quantitative targets. With 70 eval questions giving ±11.7pp noise, results are directional only.

## Smoke Observations

- Thinking mode confirmed active (avg_thinking=2589c)
- R1 adapter saved successfully to `adapters/math-star-r1-v0/`
- 21.4% smoke yield expected (1 question/cat at 10-way MCQ is extremely noisy)
- R2 round runs cleanly after R1 (dependency chain works)

## Implications for Next Experiment

- If K1544 (yield ≥ 45%) PASSES: STAR provides a scalable source of synthetic reasoning traces for P11.I1 or P11.J0 (adapter composition), which needs high-quality domain-specific training data.
- If K1545 (R1 accuracy ≥ 59%) FAILS: catastrophic forgetting from small SFT dataset — reduce training steps or increase data quality threshold.
- If K1546 (R2 yield ≥ R1 − 5pp) FAILS: fine-tuning on narrow math traces hurt general MMLU-Pro generation diversity — consider domain-weighted sampling in R2.
- SFT data includes raw thinking tags: monitor for style drift between base and adapter outputs.

## Pending

Full run results (K1544–K1546 pass/fail) to be added after task 23 completes.

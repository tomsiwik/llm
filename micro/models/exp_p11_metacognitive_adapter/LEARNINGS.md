# LEARNINGS.md — P11.H1: thinking-metar1-metacognitive-v0

**Date**: 2026-04-14  
**Status**: Design review complete — awaiting full run (pueue task 18, after H0 task 17)

---

## Core Finding

Sequential composition (H0 → H1) is structurally sound: fine-tuning from H0's checkpoint
provides a superior initialization, and format-injected PLAN/CHECK traces avoid the
accuracy penalty that direct metacognitive instruction causes (confirmed in P11.D0 smoke test).

## Why

Injecting structured metacognitive format post-hoc (after generation, before SFT) is the
only reliable way to add PLAN/CHECK structure without confusing the model's answer prediction
(arXiv:2508.17291, Meta-R1). Direct instruction in the prompt caused systematic "A" answers
in P11.D0. Format injection sidesteps this by not touching the inference-time prompt.

## Critical Fix: Data Leakage in Eval Design

When Phase 1 (trace generation) and Phase 3 (eval) use the same numpy RNG seed on the
same parquet file, `choice(size=7)` returns indices that are a prefix of `choice(size=14)`.
This makes eval questions a subset of training questions, invalidating K1521.
Fix applied: Phase 3 uses `np.random.default_rng(SEED + 1000)` for disjoint sampling.

## Theorem Framing Note

Theorem 1 frames H1 as W + ΔW_H0 + ΔW_meta (Room Model composition). The implementation
is actually sequential fine-tuning from H0's weights, which is better described as transfer
learning. The orthogonality bound doesn't strictly apply, but the experimental claim (H1 ≥ H0
accuracy) remains valid under the transfer-learning framing.

## Implications for Next Experiment

If K1520 fails (H1 does NOT reduce thinking tokens), it suggests that PLAN/CHECK structure
in training data does not compress the model's internal reasoning chain — the model learns
the format but not the efficiency. This would motivate an explicit token-budget reward signal
(GRPO with token penalty) rather than SFT-based metacognition.

If K1520 passes, the H0+H1 adapter stack becomes the basis for the universal metacognitive
adapter — a drop-in "think efficiently" module for any domain adapter composition.

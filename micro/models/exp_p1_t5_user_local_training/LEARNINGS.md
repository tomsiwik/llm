# LEARNINGS.md — T5.1: User Local Training

**Status:** SUPPORTED (Finding #436)
**Date:** 2026-04-10

---

## Core Finding

Personal stylistic adapters are viable on consumer hardware: rank-4 LoRA on Gemma 4 E4B
trained for 1.2 min on 40 synthetic examples achieves 76% compliance with a user-specific
phrase vs 0% base — within the predicted 60-80pp range.

## Why It Works

Low-rank sufficiency (Hu et al. 2021 + Aghajanyan et al. 2020): suffix injection is
effectively rank-1 in task space, so rank-4 provides ample capacity. The adapter captures
the stylistic direction with minimal parameters (3.67MB, ~64 float32 matrices).

## Key Numbers

| Metric | Value |
|--------|-------|
| Training time | 1.2 min (cold: ~2-3 min) |
| Compliance gain | 76pp (0% → 76%) |
| Adapter size | 3.67MB |
| Script size | 127 lines |
| Peak GPU memory | 4.885 GB |

## Confound to Resolve in T5.2

K1097's 76pp gain conflates two effects: (1) style injection AND (2) thinking suppression
(base model fills MAX_TOKENS=120 with `<thought>` traces, never reaching the suffix).
T5.2 must test with MAX_TOKENS >> 120 to isolate pure style injection from format change.
Expected: compliance will drop from 76% but still show clear lift over base.

## Implications for Next Experiment

T5.2 (User Adapter Validation) should: (a) use MAX_TOKENS=512+ to allow full responses,
(b) evaluate on held-out style targets the adapter wasn't trained on, and (c) test
coexistence with domain adapters to confirm T3.6/T3.7 hot-add/remove invariance holds
for user-trained (non-Grassmannian) adapters. The size budget ($0, < 2 min, < 4MB) is
proven viable — the next question is behavioral isolation and multi-adapter coexistence.

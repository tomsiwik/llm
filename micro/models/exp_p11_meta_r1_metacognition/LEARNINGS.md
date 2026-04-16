# LEARNINGS: P11.D0 — Meta-R1 Metacognition

**Status**: Full run queued (pueue task 16). These learnings are from smoke test + design.

---

## Core Finding

Metacognitive instruction injection during generation is broken — prepending PLAN/CHECK
instructions drops model accuracy from ~57% to 14.3% because the model echoes template
text into the answer slot. Fix: post-hoc format injection (generate normally, inject structure
into correct traces only).

## Why

Gemma 4's instruction-following causes it to treat prefix-injected metacognitive templates
as part of the response to generate, not as behavioral instructions. This confirms a general
principle: metacognitive scaffolding must be in the training signal (correct-trace fine-tuning),
not in the inference prompt. Aligns with arXiv:2508.17291 which trains on structured traces, not
prompted at inference time.

## Implications for Next Experiment

K1502 (≥30% token reduction) is likely to fail because format injection trains on traces that
are ~150 chars LONGER than base (PLAN prefix + CHECK suffix added to already-full thinking traces).
The model learns structure (K1504 trivially passes) but not brevity. If K1502 fails, next step
is to curate SHORT correct traces (< 800 chars thinking) and train exclusively on those — forcing
the model to learn concise solution paths before adding PLAN/CHECK scaffolding.

## Key Numbers

- Smoke yield: 2/14 (14.3%) — statistically noisy at N=14, not alarming
- Base thinking chars (smoke): 1864.5 — lower than GRPO baseline (3086), favorable for K1502
- K1502 boundary: 2160 chars. Predicted trace length post-training: ~2000-2200 chars (at-risk)
- K1504 pattern `r'First,?\s+I'` is too broad; use PLAN:/CHECK: counts for reliable measurement

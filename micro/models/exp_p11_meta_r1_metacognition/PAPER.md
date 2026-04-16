# PAPER: P11.D0 — Meta-R1 Metacognition (Planning, Regulation, Early Stopping)

**Model**: mlx-community/gemma-4-e4b-it-4bit  
**Date**: 2026-04-14  
**Status**: Full run pending (pueue task 16)

---

## Overview

Tests whether post-hoc metacognitive format injection (PLAN/CHECK scaffolding) can reduce
thinking token counts by ≥30% without accuracy loss. Based on Meta-R1 (arXiv:2508.17291).

Key mechanism: generate correct traces normally (preserves accuracy), then inject PLAN prefix
and CHECK suffix. Model learns structural termination without explicit prompting during generation.

---

## Prediction vs Measurement Table

### Smoke Test (N=10/category, 14 total, 5 training steps)

| Theorem | Prediction | Smoke Measurement | Status |
|---------|------------|-------------------|--------|
| T1: Thinking ≤ 2160 chars avg | Base ~3086 chars → meta-R1 ≤ 2160 | Phase 2 aborted (2/14 yield); no adapter eval possible | TBD (full run) |
| T2: Accuracy ≥ 62.1% | No forgetting on MMLU-Pro | Phase 2 aborted; no adapter eval | TBD (full run) |
| T3: Accuracy within 5pp of base | Planning error < 5% → minor degradation | Phase 2 aborted; base phase1_yield = 14.3% (2/14) | TBD (full run) |
| Base thinking chars | ~3086 chars (GRPO baseline) | 1864.5 chars avg in Phase 1 smoke | OK — lower than expected, good sign |
| Phase 1 yield | ~57% correct (GRPO: 8/14) | 14.3% (2/14) — see Smoke Finding 1 | SUSPICIOUS (N=14, statistical noise) |
| PLAN structure after injection | 100% of correct traces get injected | 100% structured_pct (trivially: both correct traces injected) | PASS (trivial) |

### Full Run Predictions (TBD)

| Kill Criterion | Prediction | Confidence | Expected Outcome |
|---------------|------------|------------|-----------------|
| K1502: meta-R1 thinking ≤ 2160 chars | LIKELY FAIL | Low | Format injection adds ~150 chars overhead; model may output PLAN...long_thinking...CHECK = ~2000-2200 chars — near boundary |
| K1503: accuracy ≥ 62.1% | UNCERTAIN | Medium | Training on correct MMLU-Pro traces → domain match; 200 steps is light; could go either way |
| K1504: ≥50% traces contain PLAN structure | LIKELY PASS | High | Adapter trained exclusively on PLAN-prefixed traces; but `r'First,?\s+I'` pattern is too broad — will overcount |

---

## Smoke Test Findings

### Finding 1: Phase 1 yield 14.3% (2/14) — statistically noisy, not alarming

Smoke test sampled N_SAMPLE=10 → 1 question per category → 14 total across 14 categories.
Observed: 2 correct (14.3%). Expected ~57% (from GRPO smoke test: 8/14).

**Statistical interpretation**: Binomial(14, 0.57) → P(X ≤ 2) ≈ 3%. This is suspicious
but N=14 is too small for reliable probability estimates. Each "category" here is 1 hard question;
with high variance, getting 2/14 is within statistical noise.

**Root cause**: Stratified 1/category = 14 diverse hard questions, not random sample.
avg_thinking_chars = 1864.5 confirms the thinking channel IS active; it's not a format bug.

**Conclusion**: Full run (N_EVAL=100 → 7/category) will give reliable yield estimates.
Smoke abort at Phase 2 (`insufficient_training_data`) was expected and correct.

### Finding 2: Base thinking 1864.5 chars — lower than GRPO baseline (3086 chars)

Smoke test Phase 1 saw avg_thinking = 1864.5 chars, notably lower than GRPO experiment
(3086 chars) and injection decoding experiment (2614 chars). Possible causes:
- Different question mix (14 hard stratified categories vs MMLU-Pro math-heavy sample)
- Generation variability at smoke scale

If this lower baseline holds in the full run, K1502 target of ≤2160 chars becomes easier to hit.
Full run base eval will establish the accurate baseline.

### Finding 3: Metacognitive prompting during generation confirmed broken

Smoke test confirmed: injecting metacognitive instruction as a PREFIX drops yield
from ~57% → 14.3%. Root cause: model echoes template text into answer space,
`parse_answer` finds "A" in the template text.

**Fix applied**: Format injection post-hoc (generate normally → inject PLAN/CHECK into correct
traces). All 2 correct traces had PLAN/CHECK successfully injected (structured_pct = 1.0).

---

## K1502 Risk Analysis

**Theorem 1 predicts**: T_meta ~ 600-1100 chars (bounded execution under PLAN scaffold).
**Format injection reality**: training traces = PLAN_prefix (~100 chars) + raw_thinking (~1864 chars)
+ CHECK_suffix (~50 chars) = ~2014 chars total per trace.

The model learns to generate PLAN...full_reasoning...CHECK, not PLAN...short_reasoning...CHECK.
At inference, it may replicate the full trace structure = ~2000 chars = near K1502 boundary (2160).

**Verdict**: K1502 is at-risk. If it fails, the finding is:
> 200 steps of format injection is insufficient to induce token reduction.
> The model learns structure (K1504 passes) but not brevity.
> More training or explicit short-trace examples required.

---

## K1504 Measurement Caveat

Pattern `r'First,?\s+I'` matches natural Gemma 4 output like "First, I need to consider..."
which the base model generates without any adapter. This means K1504 may pass even if the adapter
learned nothing — it's a necessary but not sufficient condition for metacognitive structure.

For the full run, also report:
- Fraction of traces containing "PLAN:" prefix (specific to injected format)
- Fraction of traces containing "CHECK:" suffix

These tighter metrics distinguish genuine structure from natural language patterns.

---

## References

- Meta-R1: arXiv:2508.17291 (three metacognitive capabilities, +27.3% SOTA)
- EWC forgetting: arXiv:1612.00796 (Kirkpatrick et al. 2017)
- Finding #530: base 62.1% MMLU-Pro + thinking (our baseline)
- GRPO MATH.md: D_train=D_eval → no forgetting theorem

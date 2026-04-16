# PAPER.md — P11.H1: thinking-metar1-metacognitive-v0 (Sequential Adapter Composition)

**Date**: 2026-04-14  
**Status**: Pre-run (design review only — H0 adapter not yet available, task 17 queued)

---

## Smoke Test Note

No smoke test was possible at design time: H0 adapter (`adapters/thinking-openthoughts-universal-v0/`)
does not yet exist (pueue task 17 in queue). Phase 1 of run_experiment.py fails fast
with a clear error if H0 is missing. This experiment must run AFTER task 17 completes.

---

## Prediction vs Measurement Table

| Metric | Predicted | Measured (full run) | Source |
|--------|-----------|---------------------|--------|
| MMLU-Pro H0 accuracy (Condition B) | ≥ 65.1% | TBD | P11.H0 MATH.md Theorem 1 |
| MMLU-Pro H1 accuracy (Condition C) | ≥ H0 accuracy | TBD | Theorem 2: higher-quality training data |
| H0 thinking chars (Condition B) | ~3202 chars | TBD | P11.H0 smoke test (3202 chars observed) |
| H1 thinking chars (Condition C) | ≤ 2562 chars (80% of 3202) | TBD | Theorem 3: PLAN/CHECK early termination |
| % structured traces (H1) | ≥ 50% contain PLAN | TBD | K1522: SFT learns injected format |
| Training time (total) | < 2h | TBD | Phase1 (~30min) + Phase2 (~7min) + Phase3 (~80min) |
| Phase 1 yield rate (correct traces) | ~65% | TBD | Q_H0 expected from H0 MATH.md |

---

## Kill Criteria

| Kill ID | Criterion | Predicted Outcome | Measured |
|---------|-----------|-------------------|----------|
| K1520 | H1 thinking chars ≤ H0 × 0.80 (≥20% reduction) | UNCERTAIN — format injection may lengthen traces | TBD |
| K1521 | H1 MMLU-Pro ≥ H0 accuracy (composition preserves quality) | LIKELY PASS — sequential fine-tune from better init | TBD |
| K1522 | ≥50% H1 traces contain PLAN structure | LIKELY PASS — all training traces have injected format | TBD |

---

## Data Leakage Fix (applied 2026-04-14)

Phase 3 evaluation changed from `np.random.default_rng(SEED)` to `np.random.default_rng(SEED + 1000)`.
With identical seed, numpy `choice(size=7)` returns the same indices as the first 7 of `choice(size=14)`,
meaning eval questions would be a subset of training questions (K1521 vacuous).
The fix ensures eval uses a disjoint sample.

---

## Theorem Framing Note

Theorem 1 in MATH.md frames H1 as composition of two independently-trained adapters
`f(W + ΔW_H0 + ΔW_meta)`. The actual implementation uses sequential fine-tuning:
H1 is initialized from H0's weights and trained further (single adapter, not two stacked).
The orthogonality bound (r²/d² ≈ 10⁻⁵) applies to independently-random adapters.
The sequential case is better described as transfer learning from a superior initialization.
The experimental claim remains valid; the composition framing is aspirational rather than exact.

---

## References

- Meta-R1 metacognition: arXiv:2508.17291
- LIMO data quality: arXiv:2502.03387
- LoRA+: arXiv:2402.07148
- Room Model: this project (W_combined = Σ ΔW_i)
- P11.H0: exp_p11_thinking_adapter_universal/MATH.md
- P11.D0: exp_p11_meta_r1_metacognition/MATH.md
- Finding #530: base model 62.1% MMLU-Pro + thinking

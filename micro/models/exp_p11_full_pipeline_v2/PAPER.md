# PAPER.md — P11.M0: Full Pipeline v2

## Prediction vs Measurement Table

| Criterion | Theorem Prediction | Smoke Result | Full Run Result | Status |
|-----------|-------------------|--------------|-----------------|--------|
| K1544: MMLU-Pro >= 70% | UNCERTAIN (67-69% expected) | TBD | TBD | TBD |
| K1545: GSM8K >= 85% | UNCERTAIN | TBD | TBD | TBD |
| K1546a: PS adds >= 1pp | LIKELY PASS (T2) | TBD | TBD | TBD |
| K1546b: adapter adds >= 1pp | LIKELY PASS (T1) | TBD | TBD | TBD |
| K1546c: injection adds >= 1pp | EXPECTED FAIL (T3) | TBD | TBD | TBD |

## Notes

**Smoke test**: Pending. Experiment depends on:
- exp_p11_rsd_aligned_traces (pueue task 19): produces adapters/math-rsd-aligned-v0
- exp_p11_grpo_improve (pueue task 20): produces adapters/math-s1k-grpo-v0
- exp_p11_injection_decoding (pueue task 13): validates injection mechanism

**Adapter fallback**: If no dep adapters exist, adapter_only = base model with same prompt.
This is detectable (delta_adapter ≈ 0) and will FAIL K1546b.

**Key caveat**: K1544 threshold of 70% matches Google's reported Gemma 4 MCQ benchmark.
Expected outcome at 35q/condition (small sample) is high variance. The experiment
is valuable for establishing relative component contributions even if absolute accuracy
is uncertain.

**Pre-registered**: K1546c (injection) expected FAIL. Gemma 4 mean thinking = 2614 chars
>> 1500 threshold → injection almost never triggers → δ_inject ≈ 0.

## Component Δ Summary (to fill post-run)

| Component | Delta vs Previous | Evidence |
|-----------|-------------------|---------|
| Adapter | TBD | adapter_only - base_thinking |
| PS prompt | TBD | adapter_ps - adapter_only |
| Injection | TBD (≈ 0 expected) | full_pipeline - adapter_ps |
| Total | TBD | full_pipeline - base_thinking |

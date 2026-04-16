# Adversarial Review: exp_p11_baseline_eval

**Verdict: PROCEED (1 inline fix applied)**

## Status

Experiment QUEUED (pueue task 1, waiting on s1K task 0 to finish).
No results.json yet. This is a design review.

## Math Review

**MATH.md is sound.**

- Theorem 1 (Adapter Evaluation Orthogonality): Trivially correct, well-stated.
  The independence property follows directly from load-one-at-a-time evaluation.
- Theorem 2 (Thinking Mode Orthogonality): Correctly grounded in Finding #536.
  Prediction of partial suppression for knowledge adapters is testable.
- Quantitative predictions table: Has specific numbers with citations. ✓
- Kill criteria K1505/K1506/K1507: Measurable and appropriate. ✓

## Implementation Review

**All adapter paths verified to exist:**
- exp_p1_t2_single_domain_training/adapters/{math,code,medical} ✓
- exp_p1_t2_multi_domain_5/adapters/{legal,finance} ✓

**MMLU-Pro data path verified:** exp_bench_mmlu_pro/data/test.parquet ✓

**Memory management:** Per-adapter load/unload with cleanup() + mx.clear_cache(). ✓

## Fix Applied (Inline)

### Bug: GSM8K data directory not created before write

`gsm_path = EXPERIMENT_DIR / "data" / "gsm8k_test.jsonl"` would fail with
`FileNotFoundError` because `EXPERIMENT_DIR/data/` does not exist.

**Fix applied:** Added `gsm_path.parent.mkdir(parents=True, exist_ok=True)`
before the existence check. Task is still queued so the fix takes effect.

## Non-Blocking Observations

- HuggingFace API for GSM8K (datasets-server) has occasional rate limits.
  Error handling is present — will return `{"accuracy": null, "error": "..."}`.
  Acceptable since K1505 checks `"error" not in r`, so a failed GSM8K fetch
  would cause K1505 to FAIL rather than silently pass with wrong data.
- K1506 tolerance is ±5pp around 62.1% — wide enough to accommodate sampling
  variance (N=280 total) but tight enough to catch major regressions.
- The experiment does not evaluate GSM8K with thinking=OFF. This is intentional
  (MATH.md specifies thinking=ON only for GSM8K). Not a flaw.

## Prediction Table (Pre-run)

| Metric | Predicted | Source |
|--------|-----------|--------|
| Base MMLU-Pro (thinking=OFF) | ~40% | Estimated |
| Base MMLU-Pro (thinking=ON) | ~62.1% | Finding #530 |
| Base GSM8K (thinking=ON) | ~77% | Finding #536 |
| Math adapter MMLU-Pro | > base | Expected domain lift |
| Knowledge adapters thinking chars | < base | Finding #536 suppression |

Actual measurements will fill this table in PAPER.md when results arrive.

## Summary

Design is correct. One blocking bug fixed inline. Experiment will run as queued.
PAPER.md should be written when results.json appears after pueue task 1 completes.

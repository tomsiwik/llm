# REVIEW-adversarial.md — P11.H1: thinking-metar1-metacognitive-v0

**Reviewer**: Adversarial  
**Date**: 2026-04-14  
**Round**: 2 of 2  
**Verdict**: PROCEED

---

## Round 2 Verification (2026-04-14)

Both blocking fixes from Round 1 are confirmed applied:

1. **PAPER.md exists** — prediction-vs-measurement table present with TBD full-run rows.
   Data leakage fix and Theorem 1 framing caveat documented. ✓

2. **Phase 3 seed fixed** — run_experiment.py line 449 uses `np.random.default_rng(SEED + 1000)`.
   Phase 1 uses `SEED` (line 211), Phase 3 uses `SEED + 1000` — disjoint samples. ✓

Non-blocking issues remain acceptable as documented in PAPER.md.

---

## Original Round 1 Review

**Round**: 1 of 2  
**Verdict**: REVISE (2 blocking fixes)

---

## Summary

The experiment design is sound: start from H0's checkpoint, fine-tune on H0-generated
metacognitive traces, measure thinking token reduction and accuracy preservation.
The MATH.md has reasonable theorems and the implementation is mostly correct.

Two blocking issues prevent proceeding:
1. PAPER.md is missing (required by proof-first protocol)
2. Data leakage: training and eval use the same seed, causing eval questions to be a
   subset of training questions — K1521 (H1 >= H0 accuracy) is therefore invalid

---

## Blocking Issues

### Fix 1: PAPER.md missing

Protocol requires PAPER.md with prediction-vs-measurement table before the experiment
is marked ready to run. Write the skeleton now (TBD rows for full-run measurements):

Required columns: Metric | Predicted | Measured (full run) | Source  
Rows: MMLU-Pro H0 acc, MMLU-Pro H1 acc, H0 thinking chars, H1 thinking chars,
structured trace %, training time

Also record smoke test status: "No smoke test possible — H0 adapter not yet available
(task 17 in queue). Design-only review."

### Fix 2: Data leakage — same RNG seed for training and eval samples

**Bug location**: run_experiment.py lines 212 and 447

Both `phase1_generate_traces()` and `phase3_evaluate()` call:
```python
rng = np.random.default_rng(SEED)  # SEED=42
```
and iterate over the same `df["category"].unique()` in the same order, sampling
from the same parquet file.

Phase 1 samples `per_cat = 14` per category; Phase 3 samples `EVAL_PER_CAT = 7`.

With numpy's `choice(..., replace=False)`, the partial Fisher-Yates shuffle means the
first 7 indices selected for size=7 ARE the same as the first 7 of the 14 selected for
size=14 when starting from the same seed. The model is trained on exactly the questions
it is evaluated on — K1521 (H1 >= H0) becomes vacuous.

**Fix**: Change Phase 3 to use `np.random.default_rng(SEED + 1000)`, OR explicitly
exclude Phase 1 indices from the eval pool.

---

## Non-Blocking Issues

### Theorem 1 notation mismatch

Theorem 1 is framed as composition of two independently-trained adapters
`f(W + ΔW_H0 + ΔW_meta)`, but the implementation does sequential fine-tuning:
H1 is initialized from H0's weights and trained further. These are not the same
operation, and the orthogonality bound (r²/d² ≈ 10⁻⁵) applies to independently
random adapters, not to a fine-tuned extension of H0.

The experiment is still valid — this is transfer learning theory (fine-tuning from
a better initialization → better convergence). But the composition framing overstates
the math. Acceptable as-is if annotated; does not block the experiment.

### K1521 circular dependency

K1521 requires knowing H0's accuracy at inference time, which is unknown until H0
completes (task 17). The check at inference time is correct — K1521 compares H1 to
the empirically measured H0 in the same run. Acceptable.

### No smoke test

H0 adapter doesn't exist yet, so Phase 1 fails fast (exit code 1). No empirical
smoke test is possible at this stage. The design has a clean prerequisite check
(line 540). Acceptable — experiment won't start until H0 is done.

---

## Verdict

**REVISE** — Apply 2 fixes:
1. Write PAPER.md with prediction table + smoke test note
2. Change Phase 3 seed to `SEED + 1000` (avoids leakage into eval set)

After fixes are applied, emit experiment.done and the next review pass should PROCEED.

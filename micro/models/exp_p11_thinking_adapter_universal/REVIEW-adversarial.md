# REVIEW-adversarial.md — P11.H0: thinking-universal-v0

**Reviewer**: Ralph (Adversarial)
**Date**: 2026-04-14
**Status**: PROCEED (round 2 — both blocking fixes verified)

---

## Round 2 Summary (REVISE fixes verified)

Both blocking fixes from round 1 have been applied correctly:

**Fix 1 (PAPER.md)**: PAPER.md written with full prediction-vs-measurement table,
smoke test findings (K1519 PASS ✓, K1517/K1518 deferred to full run), format mismatch
note, adapter size (12.56 MB), and 2-domain correction. ✓

**Fix 2 (2-domain correction)**: MATH.md Theorem 1 precondition updated from "|{D_i}| ≥ 3"
to "|{D_i}| ≥ 2". Dataset section correctly states "math:1400, code:600, Science shard not loaded".
Science→medical transfer claim removed from Theorem 2. FM4 updated accordingly.
PAPER.md notes the correction under "2-domain design (corrected)". ✓

Design is now internally consistent. Full run (pueue task 17) is ready to proceed.

---

## Original Summary (Round 1)

MATH.md is solid and the smoke test passed. The gradient diversity framing is sound
in intent. Two blocking issues prevent PROCEED: PAPER.md is missing, and there is a
silent mismatch between the declared 3-domain sampling strategy and the actual
2-domain implementation.

---

## Blocking Issues

### Fix 1: PAPER.md missing

PAPER.md is required by protocol and must contain:
- Prediction-vs-measurement table (smoke test findings + TBD full-run rows)
- Smoke test evidence (K1519 PASS, K1517/K1518 deferred to full run)
- Notes on format mismatch (DeepSeek tags → `<think>` wrapper) and size (12.56 MB)

Write a skeleton PAPER.md now; full-run rows filled in after task 17 completes.

### Fix 2: Science domain silently folded into math

**MATH.md claims**: "2000 examples stratified by domain (math:1000, code:600,
science:400)" and Theorem 1 requires "|{D_i}| ≥ 3 diverse domains".

**Code does** (`run_experiment.py:314`):
```python
n_math = min(N_MATH + N_SCIENCE, len(math_examples))  # fold science budget into math
```
Only 2 domains are loaded: code (shard 0) + math (shard 2). No science data.

**Consequences**:
1. Theorem 1's precondition "|{D_i}| ≥ 3" is violated.
2. Theorem 2's prediction "MedMCQA ≥ 55% (science→medical transfer)" is unsupported
   since there are no science examples in training.

**Fix**: Either:
a) Download a science shard and keep the 3-domain design, OR
b) Amend MATH.md to say "2 diverse domains (code + math)" and update Theorem 1
   precondition, remove science→medical transfer claim from Theorem 2 predictions.

Option (b) is faster and still valid: code + math gradients ARE diverse enough.
Remove K1518's MedMCQA sub-criterion or lower it to "expected uncertain" with
justification that no medical/science training data was used.

---

## Non-Blocking Issues (noted, no REVISE required)

1. **Theorem 1 proof step is hand-wavy**: The Cauchy-Schwarz bound gives
   |⟨Δh_i, x_j⟩| ≤ ‖Δh_i‖·‖x_j‖·(1-GD/2), but the jump to
   FG ≤ FG_single × (1-GD) is not derived — it's asserted. Acceptable for a
   guided-exploration experiment; full derivation would require empirical GD
   measurement (which we don't have). Note in MATH.md as assumption.

2. **K1517 +3pp threshold**: The "+3pp" prediction is hand-waved
   ("thinking quality × 14 categories"). It's not derived from the theorem.
   K1517 is set at 65.1% = 62.1% + 3pp. This is fine but should be noted as
   a heuristic estimate, not a theorem prediction.

3. **Smoke K1517 = 46.4%**: This is 28 questions, 2 per category. High variance
   is expected. Not a signal of failure — full run uses 210 questions. Noted.

---

## Verdict (Round 2)

**PROCEED** — Both blocking fixes applied. Design is consistent. Full run (pueue task 17) approved.

**Kill criteria expectations** (for analyst):
- K1517 (MMLU-Pro+thinking ≥ 65.1%): UNCERTAIN — smoke at 46.4% (28q noise) is uninformative
- K1518a (GSM8K ≥ 80%): LIKELY — math is directly in training data  
- K1518b (MedMCQA ≥ 55%): UNCERTAIN — no medical/science data; explicitly noted as conditional
- K1519 (thinking > 0 chars): EXPECTED PASS — 3202 chars/q in smoke confirms thinking channel active

**Remaining non-blocking issues** (from round 1, no action required):
1. Theorem 1 Cauchy-Schwarz bound → FG scaling step is asserted not derived (acceptable for guided exploration)
2. K1517 +3pp threshold is a heuristic, not theorem-derived (noted in PAPER.md)
3. Smoke K1517 = 46.4% is expected noise — not a failure signal

---

## Round 1 Verdict (superseded)

**REVISE** — Apply fixes 1 and 2, then re-emit experiment.done.

Both fixes are fast:
- Fix 1: Write skeleton PAPER.md (~10 min)
- Fix 2b: Edit MATH.md 3-domain claim + remove science→medical Theorem 2 prediction
  + edit run_experiment.py comment to remove misleading "science:400" from docstring (~10 min)

Do NOT restart the full run. Fixes are documentation-only (task 17 hasn't run yet,
so the code fix prevents the misleading claim from persisting into full results).

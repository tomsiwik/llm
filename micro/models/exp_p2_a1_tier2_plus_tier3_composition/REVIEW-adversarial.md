# REVIEW-adversarial.md — exp_p2_a1_tier2_plus_tier3_composition

**Verdict: PROCEED (KILLED — no issues with kill verdict)**

## Summary

Smoke test (n=5) reveals categorical failure sufficient to kill. K2 (personal style) drops
100pp (100% → 0%), K3 (B-matrix cosine) = 0.1607 >> 0.1 threshold. Kill verdict is correct.

## Adversarial Checks

### 1. Is the kill evidence solid at n=5?
**Yes.** Style compliance went from 100% → 0% (not 65% → 60%). This is a categorical failure,
not noise. Even at n=5, a 100pp swing is conclusive. A full run would not reverse this.

### 2. Does the impossibility derivation hold?
**Yes.** The formal bound is clean:
- Required: ε_B × (S_D / S_P) < compliance_threshold / personal_only_rate
- Measured: 0.1607 × 2.96 = 0.476 >> 0.132 (threshold)
- 3.6× violation — not close to the boundary

The derivation correctly identifies that the violation has TWO independent structural causes
(non-orthogonal B-matrices AND power imbalance), either of which alone could cause failure.

### 3. Is K1 PASS (math accuracy) meaningful?
**Marginally.** n=5 gives ±20pp variance. However, K1 is consistent with the power-dominance
analysis — math dominates, so math output is preserved. This is expected, not a positive result.

### 4. Is the fix pathway valid?
**Yes, with caveats.** Three fixes proposed:
- Grassmannian re-orthogonalization: Finding #428 confirms max_cos ≈ 2e-8 achievable ✓
- Scale normalization: Straightforward ✓
- Sequential activation (exclusive routing): T3.6 hot-add already supports this ✓

The simplest fix is sequential activation (personal adapter applied AFTER domain via hot-add),
which requires no new math — just using the existing T3.6 result.

### 5. Is PAPER.md complete?
**Yes.** Prediction-vs-measurement table ✓. Impossibility structure derived ✓. Fix proposed ✓.

## Non-blocking issues

- The "merged_adapter" directory in the experiment suggests the composition was done via
  safetensors merge, not runtime composition. This is fine for the algebraic test but means
  the "composed_rate=0%" measures rank-10 merged weights, not hot-add runtime composition.
  The sequential hot-add path (existing T3.6 mechanism) is the correct production approach.

## Decision

**PROCEED with KILLED status.** Finding #460 is accurate and the impossibility structure is
well-derived. No full run needed — the structural violation (3.6×) is too large to be noise.

**Next experiment**: Implement sequential activation using T3.6 hot-add (personal applied
AFTER domain) — this is the simplest fix and doesn't require new experiments on orthogonalization.

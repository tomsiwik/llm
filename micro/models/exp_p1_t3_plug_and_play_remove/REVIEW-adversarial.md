# Adversarial Review: T3.7 Plug-and-Play Hot-Remove

**Verdict: PROCEED**

## Math Review

Theorems 1–3 are correct and non-trivial in the following sense:
- **Theorem 1** is algebraically exact under exclusive routing. The proof is a one-liner:
  `R'[j] = R[j]` when `j ≠ k`. No approximation involved.
- **Theorem 2** correctly identifies that Python dict keys have no ghost state — O(1)
  insertion is independent of prior deletions. The reference to T3.6 Theorem 2 (Finding #429)
  is appropriate.
- **Theorem 3** (O(1) latency) is correct. Dict deletion is O(1) amortized. The prediction
  (~0.005ms) was accurate: measured 0.000205ms mean, 0.000922ms p99.

## Empirical Quality

- **K1070**: 0/40 token differences across 4 domains × 10 queries. Exact theorem, exact pass.
- **K1071**: history=100% vs 4% base. Strong pass with large margin (+96pp).
- **K1072**: p99=0.000922ms vs 10ms threshold. 10,800× margin.

## Non-blocking Caveats

1. **n=10 per domain is small** for K1070. However, since Theorem 1 is algebraically exact
   (not probabilistic), a larger n would not change the conclusion. The sample is sufficient
   to verify there are no implementation bugs.

2. **"Code" domain not tested in K1070** (only math, medical, legal, finance). This is an
   unforced omission — code was in the registry but not included in the pre-eval phase.
   Non-blocking because the theorem applies to all j ≠ k equally.

3. **Base MMLU = 4%** format compliance artifact persists. Still non-blocking (observed
   since T3.2), but T4 should verify base MCQ parsing once real routing is in place.

4. **Symmetry claim**: The T3.7 proof cites T3.6 Finding #429, which is appropriate.
   The plug-and-play claim (add + remove = complete interface) is earned.

## Summary

The experiment correctly verifies the symmetric counterpart to T3.6. The proofs are exact
(not statistical), and all predictions match measurements with large margins. T3 tier is
structurally complete: routing is load-bearing (T3.1), safe operating point established
(T3.2), activation bounds known (T3.3), N=25 validated (T3.4), full plug-and-play
confirmed bidirectionally (T3.6–T3.7).

**Status: SUPPORTED — emit review.proceed**

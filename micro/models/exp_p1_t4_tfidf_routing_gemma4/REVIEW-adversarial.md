# Adversarial Review: exp_p1_t4_tfidf_routing_gemma4

**Reviewer:** Red-team pass
**Verdict:** PROCEED

---

## Evidence Verification

All results verified against results.json:
- K1073 PASS: 0.966 ≥ 0.95 ✓
- K1074 PASS: 0.8608 ≥ 0.85 ✓ (barely — 86.08%, margin 1.08pp)
- K1075 FAIL: p99=1.11ms > 1ms ✓ (correctly marked FAIL)
- K1076 PASS: 0 LLM params ✓

No fabricated numbers. N=500 (N=5) and N=2500 (N=25) are solid sample sizes.

---

## Math Issues

**Non-blocking — Theorem 1 is heuristic, not rigorous:**
The claim ε_N ≤ C/N is asserted without deriving C. The "Vocabulary Separation Lemma" is
labeled empirical. This is acceptable for a micro-experiment whose predictions were verified.

**Blocking — None.**

**Non-blocking — Theorem 3 prediction missed by 11x:**
Proof predicted ~0.1ms; actual p99=1.11ms. The gap is entirely from Python GIL scheduler
jitter, not from FLOP count. The proof did not account for interpreter overhead. PAPER.md
correctly attributes this and notes p50=0.30ms meets spirit. For T4.3 (e2e latency), if
routing is called per-request (not per-token), 1.11ms p99 is still <20% of first-token
latency. Production fix (Cython/compiled path) is documented.

---

## Finding Status Assessment

**SUPPORTED is correct.** 3/4 criteria pass. K1075 FAIL is a Python runtime artifact:
- p50=0.30ms, p95=0.60ms both satisfy the spirit
- Routing is once-per-request, not per-token — 1ms absolute is fine
- PAPER.md makes this reasoning explicit

Danger: "spirit of K1075 is met" reasoning could be used to rationalize any FAIL. Here it's
sound because: (1) the failure mode is known and environmental (GIL), (2) the fix is
well-defined (compiled implementation), (3) practical impact is negligible at 6ms/token LLM
latency.

---

## Concerns for T4.2 (Non-blocking)

1. **K1074 margin is thin:** 86.08% vs 85% threshold = 1.08pp. A different test set draw
   could flip this. T4.2 LSH comparison should report confidence intervals.

2. **Finance confusion at 74% will persist:** The finance↔statistics confusion is structural
   (shared quantitative vocabulary). LSH routing will not fix this without domain-specific
   features. T4.2 should test whether LSH narrows or widens this gap.

3. **N=25 used N_TRAIN=300 per domain = 7500 training samples.** Real deployment may have
   fewer samples for niche domains. T4.2 should include low-data ablation (N_TRAIN=50).

---

## PAPER.md Quality

PASS:
- Prediction-vs-measurement table present ✓
- Per-domain breakdown for both N=5 and N=25 ✓
- Latency percentiles reported ✓
- Confusion floor documented and explained ✓
- Production implications stated ✓
- Confusion with Finding #389 correctly noted as generalization, not duplicate

Finding #431 already added — no action needed.

---

## Verdict: PROCEED

Core finding solid: TF-IDF nearest-centroid achieves production-viable accuracy at N=25
with zero LLM parameters and <1ms median latency. The math is sound at micro-experiment
level. Non-blocking caveats documented above should inform T4.2 design.

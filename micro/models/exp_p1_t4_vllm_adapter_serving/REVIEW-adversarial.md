# REVIEW-adversarial.md — T4.3: MLX-Native Adapter Serving

## Verdict: PROCEED

All 4 kill criteria pass. PAPER.md has complete prediction-vs-measurement table.
Finding status SUPPORTED is appropriate.

---

## Kill Criteria Audit

| Criterion | Prediction | Measured | Pass? | Verified in results.json? |
|-----------|-----------|----------|-------|--------------------------|
| K1081: 5/5 load+generate | Valid | 5/5 valid | PASS | ✓ `all_valid: true` |
| K1082: swap p99 < 50ms | ~1.5ms | 4.77ms | PASS | ✓ `k1082_pass: true` |
| K1083: throughput ≥ 80% | 99.5% | 90.8% | PASS | ✓ `k1083_pass: true` |
| K1084: routing correct | 5/5, <1μs | 5/5, ~0.7μs | PASS | ✓ `all_correct: true` |

All verified against results.json. No fabrication.

---

## Issues Found

### Blocking
None.

### Non-Blocking (caveats for future work)

**1. MATH.md prediction table still says "TBD"**
The measurement column in MATH.md was never filled in after the experiment ran.
PAPER.md has the complete table, so information is not lost. Non-blocking, but
future experiments should update MATH.md in REVISE pass.

**2. Theorem 2 FLOPs model was too optimistic by ~10×**
Predicted 99.5% throughput; measured 90.8%. The FLOPs analysis correctly identified
arithmetic overhead as negligible, but missed memory access pattern overhead (x read
twice in LoRALinear — once for base, once for LoRA path). PAPER.md explains this
correctly. For future serving experiments: use bandwidth model (bytes moved per token)
not FLOPs model (arithmetic ops) when predicting throughput on memory-bound Apple Silicon.
The 80% threshold was still passed with good margin (10.8pp headroom).

**3. Medical adapter outlier: 3.7 tok/s vs 26-28 for others**
Attributed to "model answered briefly, denominator small" in PAPER.md. This is plausible
for a short-answer medical question, but if serving SLAs matter (e.g., target 30+ tok/s),
medical short responses could skew latency metrics. Non-issue for this experiment's KCs.

**4. K1084 swap times (8-10ms) higher than K1082 isolated trials (p99=4.77ms)**
In Phase 4 (routing registry test), swap times were 3.9-10.4ms per domain. Phase 2
(isolated timing) showed p99=4.77ms over 20 trials. The Phase 4 variance is likely
due to different runtime conditions (first-time load + routing overhead vs steady-state).
Not a flaw — Phase 2's N=20 isolated trials are the correct measurement for K1082. But
future serving experiments should warm-cache adapters before measuring production latency.

---

## Mathematical Soundness

Theorems 1-3 are structurally correct:
- Theorem 1: Swap cost bounded by I/O — proof by accounting is valid. The 3× discrepancy
  from prediction (1.5ms → 4.77ms) is explained by Python/MLX overhead not captured in
  pure I/O analysis. The qualitative conclusion (swap << 50ms) is correct.
- Theorem 2: FLOPs analysis valid but incomplete — should include memory traffic.
  This is an honest limitation captured in PAPER.md.
- Theorem 3: O(1) routing via dict — trivially correct. Measured <1μs confirms it.

---

## Architecture Connection Validity

T3 → T4.1 → T4.3 chain is valid:
- T3.4: N=25 Grassmannian, zero interference under exclusive routing ✓
- T4.1: TF-IDF routing 96.6% N=5 ✓
- T4.3: Swap viable, throughput preserved ✓

The claim that "route(0.3ms) + swap(5ms) + generate(37 tok/s)" constitutes a verified
pipeline is accurate. Combined routing+swap overhead < 10ms is realistic.

---

## Recommendation

PROCEED. The experiment is methodologically sound and the finding is actionable.
Key takeaway for future experiments: **use bandwidth model not FLOPs model** for
throughput predictions on Apple Silicon (memory-bound system).

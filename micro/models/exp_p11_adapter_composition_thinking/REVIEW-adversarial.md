# REVIEW-adversarial.md — P11.J0: Adapter Composition via Exclusive Routing

**Reviewer**: Adversarial Reviewer
**Date**: 2026-04-14
**Verdict**: PROCEED (with 3 non-blocking notes)

---

## BLOCKING ISSUES: None

---

## NON-BLOCKING NOTES

### 1. Theorem 2 doesn't actually use JL-lemma — intuitive justification only

MATH.md cites arXiv:1904.10480 and says "from JL-lemma" but the derivation is:
> ~90% of questions contain ≥1 domain-specific token → each contributes directional signal

This is plausible intuition, not a JL-derived bound. JL-lemma governs dimension-preserving
projections; it doesn't directly bound centroid separation from vocabulary differences.
The 85% threshold is empirically motivated.

**Impact**: K1528 is testable regardless of whether the theorem is formally tight.
Accept the 85% threshold as empirical, not derived.

### 2. "Domain_only" condition uses base model for physics (no physics adapter)

`DOMAIN_ADAPTER_MAP["physics"] = None` → physics questions in domain_only condition
run on the unmodified base model. Theorem 1 assumes each adapter is best-in-class for its
domain, but base model is not a "domain adapter." The label "domain_only" is slightly
misleading — it's "best available domain adapter, or base if none."

**Impact**: K1526 comparison (routed vs domain_only) is still valid and tests the right
question: "does our routing approach beat the best-available domain-specific model?"
The base fallback for physics is a reasonable design choice given adapter availability.

### 3. Smoke test pending — PAPER.md has TBD results

The experiment depends on exp_p11_thinking_adapter_universal (task 17, queued).
No smoke test has run. PAPER.md has the prediction table skeleton with all results TBD.
This is acceptable given the dependency, but means design flaws will only surface at
runtime.

**Potential runtime failure**: If ADAPTER_THINKING does not exist, sys.exit(1) fires.
Domain adapters (math, medical, legal) are not pre-checked — they silently degrade to base
model. The experiment should log adapter existence early (cosmetic fix, not blocking).

---

## OVERALL ASSESSMENT

The design is coherent. Key strengths:
- Kill criteria (K1526/K1527/K1528) are cleanly implemented and match MATH.md predictions
- Pre-computing routing decisions before model cleanup avoids requiring two live models
- Acknowledged caveat about domain adapter MCQ degradation (Finding #517) is honest and
  appropriately scopes the expected outcomes
- PAPER.md exists with complete prediction table (TBD rows are fine pre-run)
- Room Model interference argument (Theorem 3) is correct: exclusive routing guarantees
  zero adapter cross-contamination by construction

**Expected outcome** (pre-registered): K1528 PASS, K1527 FAIL (domain adapters degrade MCQ
per Finding #517), K1526 ambiguous. This is itself a useful finding — it motivates training
MCQ-format domain adapters (P11.L0).

The experiment can proceed to the queue.

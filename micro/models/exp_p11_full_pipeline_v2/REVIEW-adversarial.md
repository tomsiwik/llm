# REVIEW-adversarial.md — P11.M0: Full Pipeline v2

**Verdict**: PROCEED

**Round**: 1 of max 2

---

## Summary

Design is sound. PAPER.md exists with prediction table and honest TBDs. Kill criteria
implementations are correct. Theorems are appropriately informal (frontier extension type).
Pre-registration of K1546c failure (T3) is scientifically honest and well-grounded.

No blocking fixes required.

---

## Non-Blocking Concerns (informational only)

**NB1: K1546_all omnibus always fails by design**
- Code: `k1546_pass = k1546a_pass and k1546b_pass and k1546c_pass` (line 490)
- K1546c (injection >= 1pp) is expected FAIL per T3 → K1546_all is vacuously false
- Intent stated in docstring: "adapter >= 1pp, PS >= 1pp (injection expected ~0pp)"
- Fix if desired: `k1546_pass = k1546a_pass and k1546b_pass`
- **Impact**: Only affects logging label, not `experiment complete` reporting
  (which uses K1546a/K1546b/K1546c individually)

**NB2: PS_PROMPT missing answer constraint**
- DIRECT_PROMPT says "Answer with ONLY the letter of the correct option (A through X)"
- PS_PROMPT has no such constraint — model may produce verbose justification without
  clean final letter
- `parse_answer()` handles this via "last standalone letter" fallback
- Risk: parse failures increase in PS conditions → bias against PS accuracy
- Mitigation: the "answer is X" and last-letter patterns are robust in practice

**NB3: GSM8K mislabeled as "full_pipeline"**
- `evaluate_gsm8k()` uses plain generation (no PS, no injection)
- Results key `gsm8k_full_pipeline` is misleading — it's actually "adapter_thinking"
- K1545 threshold (85%) is UNCERTAIN regardless; label doesn't affect finding

**NB4: Memory cleanup uses deprecated API**
- Line 429: `mx.metal.clear_cache()` deprecated → lines 430-432 `mx.clear_cache()` handles it
- The `hasattr` guard prevents crashes; the correct call follows immediately
- No functional impact

---

## Math Assessment

**Theorem 1** (Additive Independence): Lower/upper bounds are informal but correctly framed.
The independence of weight-space (adapter) and input-space (PS prompt) interventions
is a reasonable first-order approximation. Acknowledged as approximation, not formal proof.

**Theorem 2** (PS as Algorithmic Priming): Informal, correctly cited (arXiv:2209.01510).
"Temperature-0 beam search in thinking space" is an analogy, not a theorem. Acceptable
for frontier extension type.

**Theorem 3** (Injection Irrelevance): The strongest theorem — formally grounded in P11.Z1
pre-registration. D_avg = 2614 >> D_threshold = 1500 → P(trigger) very low. Clean.

---

## Prediction Validity

All kill criterion thresholds are pre-registered with honest uncertainty:
- K1544 (70%): explicitly marked UNCERTAIN with expected range 67-69% — honest
- K1545 (85% GSM8K): UNCERTAIN, no measurement bias — honest
- K1546c pre-registered FAIL — correctly flags the design intention

**No fabricated numbers. PAPER.md skeleton is appropriate (deps not yet run).**

---

## Decision

PROCEED — no blocking issues. Individual K1546a/K1546b pass/fail will be the
operative criteria for `experiment complete`. The omnibus K1546_all will FAIL
(as designed and pre-registered) and should be treated as informational only.

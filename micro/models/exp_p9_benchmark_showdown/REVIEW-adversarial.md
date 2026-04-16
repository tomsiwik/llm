# REVIEW-adversarial.md — P9.G1 Benchmark Showdown

**Verdict: PROCEED** *(Updated after REVISE round 1 fixes applied)*

---

## Summary

Both blocking fixes from REVISE round 1 have been correctly applied:

1. **K1391** (run_experiment.py:513-514): `k1391_val = math_gsm8k_acc - base_gsm8k_acc` — measures real gain from freshly run phases 1+2. Was previously hardcoded (63-42=21, always PASS).
2. **K1392** (run_experiment.py:519): `k1392_val = med_med_acc - base_med_acc` — measures real MedMCQA gain from phases 3+4. Cost ratio demoted to `cost_analysis_informational`. Was previously pure arithmetic on hardcoded model sizes.

PAPER.md updated with revised criterion descriptions and honest TBD measurements.
The experiment is ready to run.

---

*Original REVISE findings below for audit trail:*

---

## Blocking Fix 1: K1391 is a tautological criterion (always PASS)

**Problem**: K1391 asserts "Code adapter HumanEval ≥ base + 20pp" but:
- Code adapter HumanEval = 63.0 (hardcoded from registry, not measured in this experiment)
- Base HumanEval = 42.0 (hardcoded estimate, not measured in this experiment)
- Result: 63 - 42 = 21 ≥ 20 → PASS by construction, always

The experiment never runs HumanEval. K1391 computes a fixed answer from two fixed
constants regardless of any measured outcome. It cannot FAIL no matter what the
experiment produces.

**Fix**: Replace K1391 with a criterion using freshly measured values from phases 1+2:

```python
# K1391_new: Math adapter GSM8K gain >= 20pp over base (phases 1+2 both measured)
k1391_val = math_gsm8k_acc - base_gsm8k_acc
k1391_pass = k1391_val >= 20.0
```

Update PAPER.md to rename K1391: "Math adapter GSM8K gain ≥ base + 20pp" and mark
as UNCERTAIN (Finding #421 shows 82% total, base ~55% → expected gain ~27pp; but
base is not freshly measured in registry).

The code-adapter vs HumanEval analysis can stay as an informational section in PAPER.md
but must NOT be a kill criterion since it can't fail.

---

## Blocking Fix 2: K1392 is a tautological criterion (always PASS)

**Problem**: K1392 asserts "Pierre serving cost < 50% of Gemma 4 27B" but:
- cost_ratio = params_4b / params_27b = 4.3e9 / 27.2e9 = 15.8%
- Both values are hardcoded constants
- Result: 15.8% < 50% → PASS by construction, always

This is pure arithmetic on hardcoded model sizes. It does not depend on any
measurement. It is not a kill criterion — it's a math fact.

**Fix**: Demote K1392 to "cost analysis" section (keep the calculation, it's useful
context). Add a real third kill criterion based on freshly measured phases 3+4:

```python
# K1392_new: Medical adapter MedMCQA >= base + 3pp (phases 3+4 both measured)
k1392_val = med_med_acc - base_med_acc
k1392_pass = k1392_val >= 3.0
```

This is genuinely uncertain: registry shows 50.0% for medical adapter. If base MedMCQA
is also ~50%, the adapter provides no lift. If base is 40-45%, adapter passes. The
outcome depends on fresh measurements.

Update PAPER.md accordingly.

---

## Non-blocking Notes

**NB1**: Theorem 1 data-processing inequality usage is loose. The bound
A_adapted ≥ A_base + α · I(ΔW;D) / H(D) is aspirational; DPI holds for the
information-theoretic quantity but doesn't translate to accuracy monotonically.
Acceptable for guided exploration — just label Theorem 1 as "Motivation" rather
than "Theorem" in a future revision.

**NB2**: ORACLE_MAP for "computer science" → ADAPTER_CODE. The code adapter path
exists (verified: micro/models/exp_p1_t2_single_domain_training/adapters/code/).
No missing adapter issue.

**NB3**: No smoke test. Acceptable — many queued experiments lack smoke evidence.
Phase structure is deterministic enough that design-time review is sufficient.

**NB4**: Medical adapter registry score = 50.0% MedMCQA, same as random chance for
2-option MCQ but MedMCQA is 4-option (25% chance). 50% suggests mild positive signal,
not catastrophic. K1392_new will likely FAIL (adapter adds <3pp) — this is an honest
test.

---

## Verified Items

- ✓ PAPER.md exists with prediction-vs-measurement table (pre-run TBDs are expected)
- ✓ MATH.md has 3 theorems with quantitative predictions
- ✓ K1390 depends on fresh measurement (math_gsm8k_acc from phases 1+2)
- ✓ REPO_ROOT = 3 levels up (correct path for micro/models/exp_name/)
- ✓ Adapter paths verified against registry.json
- ✓ MedMCQA fetched from openlifescienceai/medmcqa (same source as training data — noted in caveats)

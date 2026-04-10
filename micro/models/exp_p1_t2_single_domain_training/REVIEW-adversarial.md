# REVIEW-adversarial.md — T2.1: Single Domain Adapter Training on Gemma 4 E4B

**Reviewer:** Adversarial  
**Date:** 2026-04-09  
**Verdict: PROCEED**

---

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match results.json evidence
- [x] Finding status (supported) appropriate for verification experiment
- [x] No fabricated data (is_smoke=false, adapters exist on disk)
- [x] Math errors acknowledged honestly

---

## What Holds Up

**results.json integrity:** All 5 K values match PAPER.md table exactly. No discrepancy.
Actual adapter files exist (`adapters.safetensors` per domain). Not fabricated.

**Theorem 1 (Adapter Size):** Math is correct. 2×6×2560×42=1,290,240 params = 2.46MB float16.
Measured 5MB serving / 15MB with step checkpoints. K1032 PASS (threshold=50MB, 3-10× margin).

**Theorem 3 (Expressivity):** Li et al. intrinsic dimensionality argument is sound.
Predicted ≥7pp, measured 22-82pp. Prediction is conservative; no issue.

**Finding status (supported):** Correct. Verification experiment, all K PASS, honest caveats.

---

## Issues (all non-blocking)

**1. Theorem 2 prediction is 7.8× off (acknowledged in PAPER.md)**  
Predicted 171s/domain; measured 1332s/domain. Proxy bias: T1.6 used Qwen3-4B without
gradient checkpointing; Gemma 4 E4B uses both. Theorem *claim* (<1 GPU-hour) holds with 2.7×
margin — only the quantitative prediction failed. PAPER.md discloses this honestly.

**2. Base GSM8K = 0% is a format artifact (acknowledged in PAPER.md)**  
max_tokens=256 cuts off Gemma 4's long CoT before "#### answer". The +82pp includes
format adaptation. True domain gain is 30-50pp. Disclosed and labeled "⚠️ Note" in PAPER.md.
No action needed — the artifact is the model's behavior, not measurement error.

**3. n=50 eval is small**  
82% = 41/50. Wilson 95% CI: [69%, 91%]. This is appropriate for a micro-experiment.
Finding status "supported" (not "conclusive") correctly reflects the uncertainty.

---

## P1 Critical Path Impact

T2.1 confirms:
- LoRA r=6 on q_proj (all 42 layers) works on Gemma 4 E4B ✓
- 22-82pp domain specialization at 10-22 min/domain, 5MB/adapter ✓
- T2.6 (5 domains) is now unblocked and is the immediate priority

No blocking issues. Proceed to Analyst for LEARNINGS.md.

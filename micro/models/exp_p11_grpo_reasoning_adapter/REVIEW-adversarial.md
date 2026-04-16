# Adversarial Review: exp_p11_grpo_reasoning_adapter (P11.B0)

**Reviewer:** Adversarial Reviewer
**Date:** 2026-04-14
**Verdict: REVISE** — 2 blocking fixes

---

## Summary

MATH.md is sound. Three theorems correctly derive why RS-SFT on MMLU-Pro prevents the
catastrophic forgetting seen in s1K. Smoke test passed with 57.1% yield and 2857 avg
thinking chars. Design is ready for full run with two fixes below.

---

## Blocking Fixes

### Fix 1: Write PAPER.md (required by protocol)

PAPER.md does not exist. The protocol requires PAPER.md before full results are reviewed.
Write with:
- Full prediction-vs-measurement table (predictions from MATH.md Quantitative Predictions)
- Smoke test findings: Phase1 yield=57.1%, thinking=2857 chars, Phase2 success=true, Phase3 base=50%, RS-SFT=53.6%
- TBD rows for full run (100 sample q, 200 steps, 98 eval q)

### Fix 2: K1498 directional check (penalizes improvements)

**Bug**: `abs(per_cat_sft - per_cat_base) <= 0.05` — this fails even when adapter IMPROVES a
category by >5pp. K1498 is a *catastrophic forgetting* criterion — only regressions matter.

**Fix** in `run_experiment.py` line 532 (inside the `cat_within_5pp` dict comprehension):

```python
# WRONG (current):
cat: abs(per_cat_sft.get(cat, 0) - per_cat_base.get(cat, 0)) <= 0.05

# CORRECT (directional — only penalize drops):
cat: per_cat_sft.get(cat, 0) >= per_cat_base.get(cat, 0) - 0.05
```

Without this fix, K1498 will almost certainly FAIL in the full run even if the adapter helps
every category, because at 7q per category (SD ≈ 18pp), many categories will show improvement
>5pp by random chance. The FAIL will be a false negative and may trigger an incorrect kill.

---

## Non-Blocking Notes

### N1: Statistical power of K1498 at 7q per category

At 7 questions per category and p≈0.62, SD ≈ √(0.62×0.38/7) ≈ 18.4pp. Even after Fix 2,
K1498 with a 5pp directional threshold requires substantial evidence against forgetting.
Acceptable — we just need to ensure regression detection is the right framing, not
improvement detection.

### N2: K1496 (≥64%) aggressive given smoke base of 50%

Smoke base was 50% (2q/cat, very noisy). Full run baseline expected ≈62.1% (Finding #530).
RS-SFT at 200 steps on ~62 training examples may see 1-3pp gain → K1496 will likely FAIL.
K1497 (≥56.1%, no forgetting) is the meaningful criterion here and is likely to PASS.
PAPER.md should note this ordering: K1497 is the primary kill criterion.

### N3: Smoke K1498 categories showing FAIL

"history", "other", "physics" showed >5pp delta in smoke — all due to 2q per category noise.
Non-issue for full run.

---

## Things That Are Correct

- REPO_ROOT = `.parent.parent.parent` ✓ (3 levels)
- Thinking regex: `<\|channel>thought.*?<channel\|>` with `<think>...</think>` fallback ✓
- MAX_TOKENS_SAMPLE = 2048 ✓ (smoke confirmed: 2857 avg thinking chars, no truncation)
- LoRA config via `-c lora_config.yaml` (not `--rank`/`--lora-scale` CLI args) ✓
- `load(MODEL_ID)` and `load(MODEL_ID, adapter_path=...)` ✓
- Stratified sampling across 14 MMLU-Pro categories ✓
- Phase 1 → 2 → 3 flow with proper model cleanup (del model + gc + mx.clear_cache) ✓
- Budget estimate: ~119 min fits 2h window ✓

---

## Verdict

**REVISE** — apply Fix 1 (PAPER.md) and Fix 2 (K1498 directional), then emit experiment.done.
Pueue task 14 is queued; fixes should be applied before it runs.

---

## REVISE Resolution (2026-04-14)

Both blocking fixes confirmed applied:
- Fix 1: PAPER.md written with full prediction-vs-measurement table, smoke findings, TBD rows for full run ✓
- Fix 2: K1498 at run_experiment.py:533 uses directional check `sft >= base - 0.05` ✓

**Updated verdict: PROCEED** — design is sound, fixes applied, pueue task 14 queued for full run.

Analyst: write LEARNINGS.md with literature context (RS-SFT as GRPO approximation,
EWC distribution alignment, DeepSeek-R1 warmup precedent).

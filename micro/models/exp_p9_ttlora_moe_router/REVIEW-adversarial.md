# REVIEW-adversarial.md -- exp_p9_ttlora_moe_router

**Verdict: REVISE** (2 blocking fixes)

## Pre-Review Summary

Experiment is still running in pueue (task 0, training adapter 1/5 at review time). PAPER.md and results.json do not exist yet. This review covers MATH.md methodology and run_experiment.py implementation.

## MATH.md Assessment

**Theorem 1 (Domain Separability):** Acceptable for guided exploration. The JL-lemma argument shows dimensional sufficiency (d=2560 >> 161), but the "proof" is a plausibility argument — it doesn't constructively show that Gemma 4's hidden states separate these 5 MMLU domains. The constant c is undefined. The real evidence is empirical (LLMs encode domain information, and the cited paper reports 99-100% routing accuracy with <=6 experts). For a guided exploration where the unknown IS separability, this framing is appropriate.

**Theorem 2 (Size Bound):** Correct. Direct parameter counting.

**Theorem 3 (MoE Advantage):** The formula Delta >= (alpha - 1/K)(q_bar - q_off) is algebraically correct. The predicted values (q_bar=60%, q_off=35%) are guesses with no grounding — they could be wildly off. The formula itself is the useful contribution; the specific prediction of 17.5pp is aspirational. Acceptable for guided exploration.

## Implementation Assessment

The code is well-structured:
- Sequential adapter training with proper reinit between domains
- Router trained on non-overlapping data (offset from adapter training set)
- Logit-based MCQ eval (correct for MMLU)
- Cross-domain accuracy matrix + oracle routing comparison
- Memory management follows MLX patterns

## Blocking Fixes

**Fix 1: Experiment not complete — PAPER.md required.**
PAPER.md with prediction-vs-measurement table is mandatory per proof-first protocol. Wait for pueue task 0 to finish, then write PAPER.md with:
- Prediction vs measurement for all 3 kill criteria
- Cross-domain accuracy matrix
- Router confusion matrix (which domains get misrouted)
- Comparison to oracle routing (upper bound on MoE quality)

**Fix 2: Address code domain data scarcity in PAPER.md.**
The code domain has only 412 train / 42 eval examples (vs 1000/100 target). This affects:
- Adapter quality for code (fewer training steps with meaningful data)
- Router accuracy for code (fewer hidden state examples)
- Statistical reliability of code eval (42 examples = +-15% confidence interval)
PAPER.md must flag this as a caveat and report code results separately.

## Non-Blocking Notes

- Theorem 1 would be stronger citing a specific probing study (e.g., Conneau et al. "What you can cram into a single $&!#* vector") rather than JL-lemma for the separability claim. The JL-lemma guarantees embedding capacity, not that the model uses it.
- The 17.5pp MoE advantage prediction is likely too optimistic — it assumes q_bar=60% and q_off=35%, but off-domain accuracy on MMLU MCQ with a 4-bit model could be higher (random = 25%, but LLMs have general knowledge that inflates off-domain scores).

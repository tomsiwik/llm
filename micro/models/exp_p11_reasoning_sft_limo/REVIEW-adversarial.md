# Adversarial Review: P11.A1 — LIMO Reasoning SFT

**Reviewer**: Ralph (Reviewer Hat)
**Date**: 2026-04-14
**Status**: PROCEED

---

## Summary

MATH.md is structurally sound for a Type 2 (guided exploration) experiment. Theorem
cites arXiv:2502.03387, has Theorem/Proof/QED, makes quantitative predictions, and
identifies failure modes. Code uses the validated Gemma 4 thinking regex
(`<|channel>thought...<channel|>`). No blocking fixes required.

---

## Issues Found

### 1. Theorem 1 uses binary classification gradient formula for sequence SFT (non-blocking)

**Issue**: The proof writes `E[(∂L/∂θ)²] ∝ p_x(1-p_x)` for binary y ∈ {0,1}. But this
is token-level SFT, not binary classification. The actual gradient is a sum over token
positions with variable marginals p(y_t | y_{<t}, x, θ).

**Severity**: Non-blocking. The intuition is correct — capability-boundary examples at
p_x ≈ 3-9% provide stronger gradient signal than trivial (p_x≈1) or impossible
(p_x≈0) examples. The binary formalism is an analogy that maps cleanly to the
sequence setting. Acceptable for Type 2 exploration.

**Fix**: None required. PAPER.md should note that Theorem 1 is an informal analogy,
not a strict token-level SFT bound.

---

### 2. Training format `<think>…</think>` may not align with Gemma 4 thinking tokens (non-blocking)

**Issue**: Training data uses `<think>{solution}</think>` as assistant format, but
Gemma 4 generates thinking channel as `<|channel>thought...<channel|>` at inference
time. If `<think>` / `</think>` are NOT special tokens in Gemma 4's tokenizer (just
literal text), training teaches the model to output the literal string `<think>` —
not to activate its actual thinking channel.

**Evidence**: The s1K experiment used the same format and its base eval showed 12.5%
MMLU-Pro (thinking tokens were not being stripped correctly, indicating the model
generates `<|channel>…<channel|>`, not `<think>…</think>`).

**Severity**: Non-blocking. Same format used in s1K (already queued/running). The
s1K results will reveal whether this format is compatible. If s1K's adapted eval
shows 0 thinking chars, then the format is wrong and both s1K + LIMO need reformat.

**Fix**: PAPER.md must report `avg_thinking_chars` for both base and adapted. If
adapted shows 0 thinking chars while base shows >0, flag as "thinking channel
suppression by format mismatch" and recommend retraining with correct tokens.

---

### 3. K1493 (≥65% MMLU-Pro) is aggressive for competition-math-only training (non-blocking)

**Issue**: LIMO is 817 AIME/AMC competition math problems. MMLU-Pro spans 14 categories
including biology, law, chemistry, history, economics. Getting +2.9pp on a diverse
benchmark from 817 subject-specific examples is plausible but not guaranteed.

**Expected failure pattern**: Math/physics categories improve (+5-10pp), others flat
or degrade slightly. Net may be < +2.9pp even with strong math improvement.

**Fix**: MATH.md already specifies the correct response to K1493 failure: per-category
analysis. This is correct. PAPER.md must include per-category breakdown.

---

## Strengths

- Correct thinking regex validated by p10/p11 (not the buggy s1K Phase 4a approach)
- REPO_ROOT = `.parent.parent.parent` — correct 3-level hop confirmed
- Failure Mode 2 (capability ceiling) well-identified: LIMO curation model >> E4B
- `lora_config.yaml` + `-c` flag fix applied from s1K lesson — training should succeed
- GSM8K copy from s1K avoids re-download
- K1495 (<1h) has solid analytic backing: 817 steps × ~1s/step ≈ 14 min

---

## Verdict: PROCEED

No blocking fixes. The experiment is ready to run. PAPER.md must include:
1. `avg_thinking_chars` for base and adapted (to detect format mismatch)
2. Per-category MMLU-Pro breakdown (to diagnose domain mismatch)
3. Note that Theorem 1 gradient formula is an informal analogy for sequence SFT

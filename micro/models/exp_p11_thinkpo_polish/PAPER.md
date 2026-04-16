# PAPER: P11.C0 — ThinkPO Polish (DPO: Long Thinking > Short Thinking)

## Prediction vs Measurement Table

| Prediction | Theorem | Predicted Value | Smoke Result | Full Run |
|------------|---------|-----------------|--------------|----------|
| MMLU-Pro (ThinkPO) vs GRPO | Thm 2 (ThinkPO +2-4pp) | +2pp | TBD | TBD |
| avg_thinking_chars (ThinkPO vs GRPO) | Thm 2 (DPO pushes longer) | +10% | TBD | TBD |
| GSM8K regression | Thm 4 (D_train=D_eval) | ≤ -5pp | TBD | TBD |
| DPO training convergence | Thm 1+3 (offline DPO) | val_loss decreasing | TBD | TBD |

## Kill Criteria

| Criterion | Threshold | Smoke | Full Run |
|-----------|-----------|-------|----------|
| K1499: ThinkPO MMLU-Pro ≥ GRPO + 2pp | +2pp over GRPO | — | TBD |
| K1500: avg_thinking_chars ≥ GRPO × 1.10 | +10% thinking length | — | TBD |
| K1501: GSM8K ≥ GRPO GSM8K - 5pp | no regression | — | TBD |

## Status: Awaiting Full Run

**Dependency**: exp_p11_grpo_reasoning_adapter (pueue task 14) must complete first.
The GRPO adapter provides:
1. The reference policy π_ref (for offline DPO log-prob computation)
2. The training signal: multiple completions per question with variable thinking length

**Smoke test**: Cannot run until grpo_reasoning_adapter adapter file exists at:
`micro/models/exp_p11_grpo_reasoning_adapter/adapters/rs_sft/`

Pueue ordering: task 14 (grpo_reasoning_adapter) → task 21 (thinkpo_polish) ✓

## Theoretical Justification

**Theorem 1 (DPO Objective)**: KL-constrained reward maximization has closed-form
solution via DPO reparameterization (Rafailov et al. 2023). Partition function cancels,
no reward model needed.

**Theorem 2 (ThinkPO Signal)**: Longer CoT contains more exploration steps and
self-correction. arXiv:2502.13173 showed +3.8pp on MATH500 using length-based DPO.
Prediction for E4B-4bit: +2pp (conservative due to smaller model).

**Theorem 3 (Offline DPO)**: Reference log-probs π_ref are constants in the DPO gradient.
Precomputing eliminates second model copy. Peak memory: ~5GB (1 model + LoRA).

**Theorem 4 (Distribution Alignment)**: D_train = D_eval (MMLU-Pro) → DPO updates
cannot increase MMLU-Pro loss. Same ERM argument as GRPO non-regression.

## Expected Outcome Analysis

- K1499 (+2pp): **UNCERTAIN** — paper shows +3.8pp on harder MATH500, but 4-bit quantization
  may reduce DPO sensitivity. Also depends on GRPO baseline quality.
- K1500 (+10% thinking): **LIKELY** — DPO objective directly rewards longer traces.
  Risk: quantized model may not have variable-length thinking; std(thinking_chars) may be low.
- K1501 (GSM8K no regression): **LIKELY** — distribution alignment theorem, no domain shift.

## Known Risks

**Risk 1: GRPO reference too weak** (Failure Mode 1 from MATH.md)
- If grpo_reasoning_adapter accuracy < 56.1% (K1497), preference pairs start from a
  weak reference policy. DPO may not be able to improve on top.
- Detection: compare GRPO accuracy from task 14 results.json before running DPO.

**Risk 2: Thinking length variance too low** (Failure Mode 2 from MATH.md)
- E4B-4bit may generate ~2857 chars regardless of question. If std(thinking_chars) < 200
  across completions per question, preference pairs are degenerate.
- Detection: Phase 1 generates 4 completions/question; log std(thinking_chars).
- Response: use base vs GRPO adapter as short/long pair.

**Risk 3: β=0.1 too weak** (Failure Mode 3 from MATH.md)
- KL too small → policy barely moves from reference.
- Detection: monitor |π_θ(y_w) - π_ref(y_w)| during training.

## References

- arXiv:2502.13173 (ThinkPO): Length-based DPO, +3.8pp MATH500
- arXiv:2305.18290 (DPO): Direct Preference Optimization
- arXiv:2501.12948 (DeepSeek-R1): RS-SFT warmup before preference learning
- arXiv:2501.12599 (s1): "Wait" token budget forcing for extended thinking

# LEARNINGS: P11.C0 — ThinkPO Polish

## Core Finding
Design approved (PROCEED). ThinkPO applies length-based DPO (arXiv:2502.13173) on top of the
GRPO reasoning adapter to push longer CoT traces: predicts +2pp MMLU-Pro and +10% thinking chars
vs GRPO baseline. Full run pending (blocked by pueue task 14 — GRPO adapter).

## Why
DPO's KL-constrained objective directly rewards preference pairs where longer, self-correcting
traces are correct and shorter traces are wrong. arXiv:2502.13173 validated +3.8pp on MATH500;
+2pp is the conservative 4-bit-quantized estimate.

## Key Risks to Watch at Full Run
1. **Thinking variance**: if E4B-4bit std(thinking_chars) < 200 across 4 completions/question,
   preference pairs are degenerate → K1500 fails. Fallback: use base vs GRPO as short/long pair.
2. **Save API**: `from mlx_lm import save` saves full merged model, not LoRA weights. Fix at
   smoke: use `mx.savez(ADAPTER_DIR / "adapters.npz", **dict(tree_flatten(model.trainable_parameters())))`.
3. **LoRALinear.from_base**: verify API at smoke time; fallback constructor available.

## Implications for Next Experiment
If K1499 passes (+2pp), ThinkPO establishes a cheap post-GRPO polishing step for the adapter
pipeline. If K1499 fails but K1500 passes (longer thinking, no accuracy gain), the finding
supports the hypothesis that thinking length and accuracy are partially decoupled at 4-bit scale.

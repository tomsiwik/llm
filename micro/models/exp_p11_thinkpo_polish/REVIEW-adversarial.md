# Adversarial Review: P11.C0 — ThinkPO Polish

**Reviewer**: Adversarial  
**Date**: 2026-04-14  
**Verdict**: PROCEED

---

## Verdict Summary

Design is mathematically sound. PAPER.md has prediction table (all TBD — expected, blocked by GRPO dep). Kill criteria implementations match MATH.md. No blocking issues.

---

## Checklist

| Item | Status |
|------|--------|
| PAPER.md has prediction-vs-measurement table | ✓ (TBD values, dependency documented) |
| Kill criteria match MATH.md definitions | ✓ |
| K1499: `thinkpo_acc >= grpo_acc + 0.02` | ✓ |
| K1500: `thinkpo_thinking >= grpo_thinking * 1.10` | ✓ |
| K1501: `thinkpo_gsm >= grpo_gsm - 0.05` | ✓ |
| Math errors or unsupported claims | None found |
| Theorems cited to prior work | T1=arXiv:2305.18290, T2=arXiv:2502.13173, T3 derived |
| Data split: train/eval leakage | Clean — `sample(frac=0.5)` / `drop(train_df.index)` |
| Smoke test | Impossible (GRPO dep pueue task 14) — documented |

---

## Non-Blocking Notes

**NB1: `from mlx_lm import save` saves full model, not LoRA adapter**  
Line 409-410: `from mlx_lm import save; save(ADAPTER_DIR, model, tokenizer)` — this saves the entire merged model+tokenizer to ADAPTER_DIR, not just the DPO LoRA weights. Log message says "ThinkPO adapter saved" but downstream users expecting a loadable adapter path will get a full model directory instead. Evaluation is unaffected (happens in-memory before save). Fix at smoke test time: use `mx.savez(ADAPTER_DIR / "adapters.npz", **dict(tree_flatten(model.trainable_parameters())))` to save only LoRA weights.

**NB2: `LoRALinear.from_base` API compatibility**  
Line 366: `LoRALinear.from_base(proj, r=4, dropout=0.0, scale=1.0)` — the mlx_lm `LoRALinear` API has changed across versions. Verify at smoke test time when GRPO adapter is available. If `from_base` raises AttributeError, fall back to `LoRALinear(proj.input_dims, proj.output_dims, r=4)`.

**NB3: Theorem 2 empirical claim**  
"Longer thinking traces yield higher expected accuracy" is contested in recent literature (longer reasoning doesn't always help). However, citing arXiv:2502.13173 (+3.8pp on MATH500) is sufficient evidence for a grounded hypothesis, and K1499 directly tests this. Acceptable as frontier-extension type.

**NB4: GRPO reference is self-measured**  
The GRPO baseline accuracy used in K1499/K1501 is measured IN this experiment run (grpo_baseline eval), not taken from the GRPO experiment's results.json. This creates a slight inconsistency if the in-run GRPO eval score differs from the actual GRPO training run score. However, this is methodologically correct — it ensures the same eval conditions for both baselines.

---

## Passed Checks

- Theorem 1 (DPO objective): standard Rafailov et al. 2023, correct reparameterization
- Theorem 3 (offline DPO): reference log-probs are constants in gradient — mathematically sound
- Theorem 4 (distribution alignment): same argument as GRPO Theorem 1, valid
- +2pp prediction is conservative vs paper's +3.8pp — appropriate for 4-bit quantized model
- Failure modes 1-3 documented with detection and response strategy
- pueue ordering enforces GRPO dep (task 14 → task 21) — correct

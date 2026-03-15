Both exp_correlated_layer_errors (PROVEN: correlation reduces error, sub-additivity robust) and exp_residual_layernorm_error_dynamics (PROVEN: Pre-RMSNorm 11.5x safer than feedforward, 1/d scaling preserved) completed. Awaiting review.

## Iteration 51 (2026-03-15)

### New results since last iteration
- run_sole_vs_finetune_1773531197: CRASHED — CUDA OOM. Float16 7B model (~14GB) + iterative merge_and_unload exhausted 24GB GPU.
- run_b_repulsion_1773520515: CRASHED — empty param list (already known, code was fixed)
- run_continual_addition_1773531208: ACTIVE (still running)

### Actions taken
1. Fixed run_sole_vs_finetune.py OOM:
   - Switched from float16 to 4-bit quantization (BitsAndBytesConfig, ~4GB VRAM)
   - Replaced iterative merge_and_unload with add_weighted_adapter (matches composition_quality pattern)
   - Reordered phases: base eval first, then SOLE composition, then union QLoRA training
   - For union LoRA phase: reload fresh 4-bit model + QLoRA, evaluate as PeftModel (no merge_and_unload on quantized base)
2. Resubmitted as run_sole_vs_finetune_1773569015
3. Integrated both failure results into HYPOTHESES.yml

### Queue Status
- 29 pending tasks (28 old + 1 resubmission)
- 1 active: run_continual_addition_1773531208
- b_repulsion resubmission (1773567889) already in queue with fixed code

### Next iteration should
- Check if continual_addition and any queued jobs have completed
- Check for run_sole_vs_finetune_1773569015 result
- Look for newly completable macro nodes

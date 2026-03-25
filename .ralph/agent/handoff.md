# Session Handoff

_Generated: 2026-03-25 01:11:08 UTC_

## Git Context

- **Branch:** `main`
- **HEAD:** 76221eb: chore: auto-commit before merge (loop primary)

## Tasks

### Completed

- [x] Submit reasoning expert distillation (P1) to GPU queue
- [x] Integrate completed GPU results: cat_weight_convergence (PROVEN) and attention_layer_orthogonality (SURVIVES)
- [x] Write and submit exp_lte_no_reset_macro GPU experiment script
- [x] Fix rsync --delete destroying RunPod-only adapters, retrain all 50
- [x] Fix stuck GPU queue: kill hung run_all_eval, reorder pending.jsonl
- [x] Script and submit exp_full_base_free_pipeline
- [x] Update exp_distillation_quality_vs_teacher to active and update scratchpad with queue status
- [x] Generate 2-3 new macro hypotheses from recent findings
- [x] Write and submit scale-500 experiment scripts (data prep, training, eval)
- [x] Generate 2-3 new macro hypotheses and write experiment scripts
- [x] Fix GPU queue: kill stuck eval, reorder pending, sync scripts
- [x] Fix teacher comparison scripts missing from main (copy from worktree)
- [x] Fix SFTConfig max_seq_length bug across training scripts
- [x] Write and submit exp_ffn_only_macro_composition experiment script
- [x] Resubmit failed LTE no_reset experiment
- [x] Fix LTE no-reset missing output directory in prepare_domain_data
- [x] Integrate MMLU held-out eval results into HYPOTHESES.yml exp_pilot50_held_out_eval
- [x] Generate 2-3 new macro hypotheses and submit to GPU queue
- [x] Activate 5 submitted experiments in HYPOTHESES.yml (small_n, individual, selective, redundancy, interpolation)
- [x] Fix broken HYPOTHESES.yml deps: exp_cross_domain_composition and exp_cat_weight_convergence nodes don't exist
- [x] Fix composition_quality script — catastrophic PPL values suggest adapter loading bug. Resubmit.
- [x] Fix measure_orthogonality script — grad_fn error in backward pass
- [x] Fix run_clone_compete — dtype mismatch float vs bfloat16
- [x] Fix run_full_base_free — AMP GradScaler bug
- [x] Fix measure_orthogonality.py grad_fn crash and resubmit
- [x] Integrate composition_quality catastrophic failure result into HYPOTHESES.yml
- [x] Integrate answer_conditioned_shadow KILLED result into HYPOTHESES.yml
- [x] Integrate b_repulsion crash + cancel duplicates + generate new macro hypotheses
- [x] Fix 3 failing GPU scripts: b_repulsion (empty params), continual_addition (OOM/fp16), sole_vs_finetune (OOM/50 adapters)
- [x] Fix OOM in run_continual_addition.py and run_sole_vs_finetune.py — use 4-bit quantization and gradient checkpointing
- [x] Fix OOM in run_sole_vs_finetune.py and run_continual_addition.py: CPU-based adapter composition
- [x] Generate 2-3 new macro hypotheses for composition scaling
- [x] Generate 2-3 new macro hypotheses for HYPOTHESES.yml
- [x] Integrate status: no new results, 37 pending, pipeline caught up


## Key Files

Recently modified:

- `.claude/skills/experiment/SKILL.md`
- `.env.example`
- `.gitignore`
- `.ralph/agent/scratchpad.md`
- `.ralph/agent/summary.md`
- `.ralph/current-events`
- `.ralph/current-loop-id`
- `.ralph/current_direction.md`
- `.ralph/events-20260324-002755.jsonl`
- `.ralph/events-20260325-010842.jsonl`

## Next Session

Session completed successfully. No pending work.

**Original objective:**

```
BitNet-SOLE Research Loop — Wave 3: Foundation + Base-Free + Serving + Evolve.

ARCHITECTURE: Ternary base (BitNet-2B-4T) + ternary LoRA adapters (QAT+STE) + Grassmannian skeleton (frozen A, 17x decorrelation filter) + runtime LoRA serving (no merge).
STATE: Read .ralph/current_direction.md for active work. Read HYPOTHESES.yml Wave 3 section.
CONTEXT: VISION.md (rewritten 2026-03-22), FINDINGS.md, references/BITNET_SOLE_RESEARCH.md.

SCALE CONSTRAINT: micro only. LOCAL Apple Silicon via MLX. $...
```

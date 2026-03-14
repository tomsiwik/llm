2026-03-14: 3 consecutive kills → picking high-confidence exp_amplification_factor_scaling (priority 4, micro, depends on proven zero_shot_base_transfer). Straightforward scaling measurement at d=64/128/256. Delegating to experiment-ideator.

2026-03-14: Macro loop — scripted 2 eligible P3 macro experiments.
Queue had dead entries for scripts that never existed (measure_orthogonality, run_composition_quality pointed to non-existent paths). Created both scripts and submitted fresh queue entries.

1. exp_pilot50_composition_quality (P3): run_composition_quality.py
   - Pre-merges adapters at N=5,10,25,50 via PEFT add_weighted_adapter
   - Measures per-domain PPL vs single-expert baselines
   - Kill: >10% degradation at N=50, superlinear scaling, >20% worse than base
   - Submitted as run_composition_quality_1773517791

2. exp_converged_adapter_orthogonality (P3): measure_orthogonality.py
   - Validates micro orthogonality proof with converged production adapters
   - Checks convergence (non-trivial loss), pairwise cosines, gradient-alignment bias
   - Kill: cos >2x micro prediction, gradient bias >0.05
   - Submitted as measure_orthogonality_1773517906

Queue state: 1 active (run_all_eval), 18 pending (16 original + 2 new). Worker running, GPU at 38%.
All eligible P3 macro nodes now scripted and queued. Remaining open macro nodes:
- exp_distillation_quality_vs_teacher (P3) — scripts exist in queue but dirs don't exist locally
- exp_scale_500_experts (P4) — impractical, needs 500 adapters
- exp_composable_merge_pipeline (P5) — blocked by collision_scaling deps
Next: check queue progress, integrate results when active tasks complete.

## 2026-03-14 Macro Loop — Status Check & Housekeeping
Queue check at 20:09 UTC:
- ACTIVE: run_all_eval_1773497259 — running ~1hr (started 19:06 UTC), GPU 34%
- PENDING: 20 tasks in optimal order (training → eval → retrain → adapter-dependent)
- 3 old FAILED tasks still in pending (diagnose, orthogonality, composition_quality from pre-rsync-fix) — re-submissions later in queue will supersede
- Worker: RUNNING

Updated exp_distillation_quality_vs_teacher: open → active. Scripts were already queued (train_70b_missing, train_8b_adapters, eval_teacher_comparison) but status was never updated.

All eligible macro nodes now active or queued:
- P1: exp_reasoning_expert_distillation (active, training scripts queued)
- P2: exp_pilot50_held_out_eval (active, run_all_eval running now)
- P2: exp_reasoning_domain_composition (active, queued after adapters)
- P2: exp_clone_compete_evolution (active, queued after adapters)
- P3: exp_pilot50_composition_quality (active, re-submitted)
- P3: exp_converged_adapter_orthogonality (active, re-submitted)
- P3: exp_distillation_quality_vs_teacher (NOW active, scripts queued)
- P4: exp_automated_correction_pipeline (active, queued)
- P4: exp_lte_no_reset_macro (active, queued)
- P4: exp_full_base_free_pipeline (active, queued)
- P5: exp_composable_merge_pipeline (active, queued)

Remaining OPEN macro with unsatisfied deps:
- exp_cross_domain_semantic_transfer (P3) — blocked by exp_pilot50_composition_quality
- exp_scale_500_experts (P4) — deps satisfied but needs 500 adapters (impractical)
- exp_evolution_convergence (P5) — blocked by exp_clone_compete_evolution

No results to integrate yet. Queue must progress. Next: integrate results when tasks complete.

## 2026-03-14 Macro Loop — New Hypotheses Generated
No practical eligible macro nodes remain (exp_scale_500_experts needs 500 adapters).
Killed orphaned eval_humaneval process (pid 50996) from first killed run_all_eval attempt — was wasting GPU for 5+ hours.
run_all_eval_1773497259 (2nd attempt) is legitimately running — child process eval_mmlu.py at 99% CPU, 3GB RSS.

Generated 3 new macro hypotheses:
1. **exp_answer_conditioned_shadow_scoring** (P3) — validate answer-only PPL as quality metric at macro with pilot50 + MMLU. Evolve enabler. Depends on: exp_answer_conditioned_scoring (proven), exp_distillation_pilot_50 (supported). ELIGIBLE.
2. **exp_b_matrix_interference_regularization** (P4) — train with B-matrix repulsion regularizer to reduce interference. Root cause fix from minimax Grassmannian findings. Depends on: exp_grassmannian_expert_init (proven), exp_distillation_pilot_50 (supported). ELIGIBLE.
3. **exp_expert_pruning_from_composition** (P4) — test if removing weak experts improves composed model. Quality ceiling investigation. Depends on: exp_cat_weight_convergence (proven), exp_pilot50_composition_quality (active). BLOCKED.

Next: script exp_answer_conditioned_shadow_scoring (P3, eligible, directly enables evolve phase).

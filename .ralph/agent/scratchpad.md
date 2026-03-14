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

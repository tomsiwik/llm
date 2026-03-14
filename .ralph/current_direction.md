# Current Direction: GPU queue processing 28+ macro experiments

## Status
All eligible open macro nodes have been activated and submitted to GPU queue.
Worker is processing run_all_eval (held-out evaluation), 28 tasks pending.

## Active Macro Experiments (in GPU queue)
1. **exp_pilot50_held_out_eval** (P2) — Currently running on worker
2. **exp_reasoning_expert_distillation** (P1) — train + eval scripts queued
3. **exp_reasoning_domain_composition** (P2) — composition test queued
4. **exp_pilot50_composition_quality** (P3) — resubmitted after failure
5. **exp_distillation_quality_vs_teacher** (P3) — 8B vs 70B teacher comparison
6. **exp_sole_vs_full_finetune** (P3) — SOLE vs union LoRA baseline
7. **exp_expert_continual_addition** (P3) — add 10 new experts to composed-50
8. **exp_sole_inference_throughput** (P4) — throughput benchmarks
9. **exp_scale_500_experts** (P4) — scale to 500 experts
10. **exp_automated_correction_pipeline** (P4) — teacher correction pipeline
11. **exp_composable_merge_pipeline** (P5) — merge pipeline with quality gates

## Next Steps
- Wait for GPU results to complete
- Integrate findings as experiments finish
- Generate new hypotheses from proven/killed results

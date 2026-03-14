2026-03-14: Reviewed and integrated exp_inference_routing_strategies (KILL confirmed) and exp_lte_parallel_base_construction (PROCEED). Consecutive kills reset to 0. Two new hypotheses generated: LTE no-reset at macro, LTE rank accumulation quality at scale.

## 2026-03-14 Macro Loop Iteration

### State Assessment
- GPU queue: 1 active (pilot50_held_out_eval), 12 pending
- Completed: cat_weight_convergence (PROVEN), attention_layer_orthogonality (PROVEN) — already integrated
- fix_and_retrain: 4/5 adapters INCOMPLETE, diagnose_adapters2 pending in queue
- composition_quality failed (domains_with_data empty — likely due to incomplete adapters)

### Eligible Open Macro Nodes (deps satisfied)
- p3: pilot50_composition_quality (queued), distillation_quality_vs_teacher (queued), converged_adapter_orthogonality (queued)
- p4: **lte_no_reset_macro**, scale_500_experts, automated_correction_pipeline, full_base_free_pipeline
- p5: composable_merge_pipeline

### Task: Write & submit exp_lte_no_reset_macro
- Highest priority unqueued eligible macro node (p4, blocks full_base_free_pipeline)
- Dependency proven: lte_parallel_base_construction showed par/seq equivalence
- Micro found: no-reset diverges at d=64 due to 8x scaling bug (alpha/r double-counting in forward pass)
- Hypothesis: at macro d=768+, alpha/r is proportionally smaller → may stabilize
- Script: GPT-2 125M on RunPod, compare reset vs no-reset LTE, measure divergence

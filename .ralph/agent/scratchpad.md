Launching two parallel micro experiments: exp_correlated_layer_errors and exp_residual_layernorm_error_dynamics. Both follow up on proven multilayer_removal_cascade. Independent experiments, different dirs.

## Iteration 50 (2026-03-15)

### New results since last iteration
- run_b_repulsion_1773520515: FAILED — empty parameter list in AdamW optimizer. Code bug, not experiment failure. Local code already fixed (explicit requires_grad loop + RuntimeError guard). Resubmissions pending in queue (1773567889).

### Actions taken
1. Cancelled duplicate queue entries: run_composable_merge_1773566242 and run_b_repulsion_1773567480 (kept newer of each pair)
2. Queue reduced from 29 → 27 pending tasks
3. Generated 2 new macro hypotheses with satisfied deps:
   - exp_ppl_probe_macro_composition (p3): PPL-probe weighted composition at macro. Deps: distillation_pilot_50 (supported) + cross_domain_dilution_vs_k (proven). Submitted as run_ppl_probe_composition_1773568575.
   - exp_adaptive_rank_macro_validation (p4): Spectral rank heuristic validation at Qwen2.5-7B. Deps: distillation_pilot_50 (supported) + adaptive_rank_selection (proven). Submitted as run_adaptive_rank_validation_1773568588.
4. Both set to active in HYPOTHESES.yml

### Queue Status
- 29 pending tasks (27 old + 2 new submissions)
- 1 active: run_sole_vs_finetune_1773531197
- Key bottleneck: GPU throughput (single worker)
- All open macro nodes still blocked on exp_pilot50_composition_quality or exp_clone_compete_evolution

### Next iteration should
- Check if sole_vs_finetune has completed
- Check for any new completed results
- Consider generating more hypotheses if pipeline still stalled

Integrated exp_gamma_perturbation_correlation (PROVEN) and exp_removal_position_sensitivity (SUPPORTED). Both adversarial reviews: PROCEED. Safety story complete. 2 new low-priority micro hypotheses generated (silu_gamma_correction, gs_random_permutation_validation).

## Iteration 52 (2026-03-15)

### Queue status
- ACTIVE: eval_math500_1773512952 (reasoning expert distillation eval)
- PENDING: 37 tasks (was 35, added 2 new)
- Worker: RUNNING, GPU 10%

### New results since iteration 50
- run_inference_throughput_1773531219: KILLED (K3: 12.9% degradation at bs=32, threshold 10%)
  - K1 SURVIVES: 97% of base at bs=1 (5.3 vs 5.4 tok/s)
  - K2 SURVIVES: first-token latency ratio 1.03x
  - Already integrated into HYPOTHESES.yml (status: killed)
- train_scale_adapters_1773530181: OK but nothing to train (0 adapters needed — prepare_scale_data failed)
- prepare_scale_data_1773530168: FAILED — KeyError: 'question' (old code on worker). Local code is fixed.
  Resubmission prepare_scale_data_1773575271 will rsync fixed code.
- eval_teacher_comparison_1773501574: FAILED — no 8B adapters exist. Script needs adapters_8b/ directory.
  No resubmission needed until 8B adapters are trained.

### Key insight
All open macro nodes with satisfied deps are already queued. Per objective, generated 2 new macro hypotheses:

1. **exp_leave_one_out_expert_ranking** (p3): Leave-one-out PPL ranking to identify harmful vs helpful experts. Label-free diagnostic. Submitted as run_leave_one_out_1773576048.

2. **exp_composition_dropout_robustness** (p3): Bootstrap test with 20 random 80% subsets to measure composition fragility. Submitted as run_dropout_robustness_1773576059.

Both complement existing experiments:
- LOO ranking complements expert_pruning_from_composition (which is blocked on composition_quality)
- Dropout robustness complements greedy_expert_selection (which finds optimal subset)

### Next iteration should
- Check if eval_math500 completed
- Check for new results from the 37 pending tasks
- Look for newly completable macro nodes

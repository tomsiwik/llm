exp_composition_weight_sensitivity: REVISE verdict from adversarial review. 6 fixes needed: (1) SNR~O(N) is prediction not finding (+0.57 contradicts -1.0), (2) increase noise samples 3→20, (3) fix signal/noise asymmetry, (4) qualify ortho measurement scope, (5) downgrade to "supported K2 only", (6) add trivial-K2 caveat to FINDINGS.md.

## 2026-03-15 Iteration: Fix retrain_all_adapters timeout

### Problem
retrain_all_adapters_1773514547 FAILED with subprocess.TimeoutExpired.
The per-adapter timeout was 1800s (30min), but training 300 steps at ~29s/step
takes ~8700s (~2.4hr). Script crashed at step 58/300 of "academic-writing" domain.

### Fix
Increased timeout from 1800 to 10800 (3hr) in scripts/retrain_all_adapters.py.
Committed: 6f891ed. Resubmitted as retrain_all_adapters_1773536208.

### Queue Status
- Active: train_reasoning_expert_1773532315 (micro)
- Pending: 26 tasks (25 prior + retrain resubmit)
- 3 new hypotheses from last iteration already queued (small_n_eval, redundancy_analysis, interpolation)

### Next Steps
1. Wait for retrain_all_adapters to complete (should take many hours with all missing adapters)
2. After retrain: re-run MMLU held-out eval to check if retrained adapters pass kill criteria
3. Monitor HumanEval (run_humaneval_only_1773533006) and other pending tasks
4. All 3 new hypothesis scripts are already queued — nothing more to submit

## 2026-03-15 Iteration: New Diagnostic Hypotheses for MMLU Regression

### Context
- All eligible open macro nodes already submitted (small_n_eval, redundancy_clustering, interpolation)
- Queue: 1 active + 26 pending. All known script bugs fixed.
- Per objective: generate new hypotheses when no eligible nodes remain

### Critical Gap
The -3.67pp MMLU regression is the biggest threat to SOLE. Existing experiments test
N-dependent dilution (small_n_eval) and adapter redundancy, but don't isolate the
root cause: is it distillation quality or composition interference?

### New Hypotheses (2)

1. **exp_individual_expert_held_out** (priority 2)
   - Tests each adapter INDIVIDUALLY on held-out MMLU (no composition)
   - Separates distillation quality from composition interference
   - If individuals regress → distillation memorized, fix training data/approach
   - If individuals neutral → composition creates interference, fix with selective composition
   - Submitted: run_individual_eval_1773536662

2. **exp_selective_composition_mmlu** (priority 2)
   - Composes only k=1,3,5 relevant experts per MMLU subject (domain matching)
   - Tests if targeted selection avoids dilution from irrelevant experts
   - Compares against random-k baseline to validate domain matching
   - Submitted: run_selective_composition_1773536672

### Queue Status
- Active: train_reasoning_expert_1773532315
- Pending: 28 tasks (26 prior + 2 new diagnostic)

### Next Steps
1. Wait for GPU results — 28 pending tasks will take many hours
2. Priority results to watch: individual_expert_eval + selective_composition (MMLU diagnosis)
3. retrain_all_adapters will retrain with rank-16 all-modules — then re-run held-out eval
4. After diagnosis experiments complete, either pivot distillation approach or add routing layer

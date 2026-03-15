exp_rmsnorm_gamma_nonuniformity integrated as PROVEN. Micro safety story complete (all macro transfer risks resolved). 2 new hypotheses generated: silu_vs_gelu_gamma_correction (p5), gamma_perturbation_correlation (p4).

## Iteration 51 (2026-03-15)

### Situation
- No open macro nodes with all deps satisfied (4 open macro nodes blocked by active deps)
- 35 pending + 1 active (run_inference_throughput) in GPU queue
- No new results to integrate (only the old continual_addition OOM failure, already handled)
- Pipeline is saturated — 31 active macro nodes all have scripts submitted

### Actions taken
1. Generated 3 NEW macro hypotheses targeting the critical gap: Evolve phase has no quality signal
   - exp_task_accuracy_evolve_signal (p2): 10-question held-out benchmark as adapter ranking signal
   - exp_relative_ppl_within_domain (p3): within-domain PPL ranking (salvaging PPL for clone-and-compete)
   - exp_composition_health_kl_divergence (p3): label-free KL divergence composition health monitor
2. Wrote experiment scripts for all 3
3. Submitted to GPU queue: run_task_accuracy_signal_1773574632, run_relative_ppl_1773574642, run_kl_health_1773574653
4. Committed as c4c9951

### Key insight
The Evolve phase is completely blocked without a macro-scale quality signal. Answer-only PPL was killed (r=-0.63 cross-domain). Two approaches to fix:
- **Task accuracy**: more expensive but reliable (10 held-out questions per domain)
- **Within-domain PPL**: PPL may still work for within-domain comparisons (clone-and-compete only needs within-domain ranking)
- **KL divergence**: label-free health monitoring, doesn't solve ranking but detects catastrophic composition

### Queue status
- ACTIVE: run_inference_throughput_1773531219
- 38 pending tasks (35 + 3 new submissions)
- Key blocker: retrain_all_adapters_1773536208 (must complete before composition experiments are meaningful)

### Next iteration should
- Check if run_inference_throughput has completed
- Check if any other queued tasks have results
- If no new results, consider whether pipeline is stalled and whether to investigate retrain_all_adapters progress

## Iteration 52 (2026-03-15)

### New results since last iteration
- run_inference_throughput_1773531219: KILLED by K3 (12.9% degradation at bs=32, threshold 10%)
  - K1 SURVIVED: throughput ratio 0.97 at bs=1 (5.3 vs 5.4 tok/s)
  - K2 SURVIVED: first-token latency ratio 1.03x (206.5ms vs 201.4ms)
  - K3 KILLED: 12.9% degradation at max batch (threshold 10%)
  - Narrowly killed — composition overhead is minimal for single requests
- train_scale_adapters_1773530181: OK but vacuous (no training data exists for scale_500 domains)
- prepare_scale_data_1773530168: FAILED (instant fail, 0s runtime — data gen failed)
- eval_teacher_comparison_1773501574: FAILED (0s runtime, instant fail)
- ACTIVE: eval_math500_1773512952 (reasoning expert eval)

### Actions taken
1. Integrated exp_sole_inference_throughput as KILLED in HYPOTHESES.yml
2. Updated meta: consecutive_kills=1, last_killed_node=exp_sole_inference_throughput
3. No downstream nodes blocked by this kill (blocks: [])

### Queue status
- ACTIVE: eval_math500_1773512952
- 34 pending tasks
- Key blocker: retrain_all_adapters_1773536208 still pending

### Next iteration should
- Check for new results (eval_math500 may complete)
- Investigate why prepare_scale_data and eval_teacher_comparison instantly failed
- Check if any open macro nodes now have satisfied deps
